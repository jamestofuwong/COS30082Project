import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
import os
import json
import pickle
from pathlib import Path
import timm
import torch.nn.functional as F

# ============================================================================
# HELP FUNCTION
# ============================================================================

def load_species_names(species_list_path):
    """Load species names from the text file and create mapping from class_num to name"""
    class_to_name = {}
    try:
        with open(species_list_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ';' in line:
                    class_num, name = line.split(';', 1)
                    class_num = class_num.strip()
                    name = name.strip()
                    class_to_name[class_num] = name
        print(f"Loaded {len(class_to_name)} species names from {species_list_path}")
    except FileNotFoundError:
        print(f"Warning: Species list file {species_list_path} not found. Using class numbers as names.")
    except Exception as e:
        print(f"Error loading species names: {e}. Using class numbers as names.")
    
    return class_to_name

def convert_class_to_name(class_id, class_to_name_map):
    """Convert class ID to species name, fallback to class ID if name not found"""
    class_id_str = str(class_id)
    if class_id_str in class_to_name_map:
        return class_to_name_map[class_id_str]
    else:
        # Try direct lookup
        for class_num, name in class_to_name_map.items():
            if class_num == class_id_str:
                return name
        return f"Class {class_id}"

def format_prediction(class_id, species_name):
    """Shared function to format predictions consistently for both models"""
    return f"{class_id}: {species_name}"

def get_herbarium_image(class_id, herbarium_base_path="herbarium"):
    """
    Get the herbarium image for a given class ID.
    Returns the image path if found, None otherwise.
    """
    class_folder = os.path.join(herbarium_base_path, str(class_id))
    
    if not os.path.exists(class_folder):
        print(f"Herbarium folder not found for class {class_id}: {class_folder}")
        return None
    
    # Look for any image file in the class folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    for ext in image_extensions:
        for file in os.listdir(class_folder):
            if file.lower().endswith(ext):
                image_path = os.path.join(class_folder, file)
                print(f"Found herbarium image for class {class_id}: {image_path}")
                return image_path
    
    print(f"No herbarium image found for class {class_id} in {class_folder}")
    return None

# ============================================================================
# Model Base Class
# ============================================================================

class BaseModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.idx_to_class = None
        
    def predict(self, image, class_to_name_map):
        raise NotImplementedError("Subclasses must implement predict method")

# ============================================================================
# Novel3 Model Components (DANN+Contrastive+SNN+Hybrid)
# ============================================================================

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class StyleNormalizationNetwork(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=512):
        super(StyleNormalizationNetwork, self).__init__()
        
        self.style_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # Project to output dimension for gamma/beta
        self.style_gamma = nn.Linear(hidden_dim // 2, output_dim)  
        self.style_beta = nn.Linear(hidden_dim // 2, output_dim)  
        
        # Project input to output dimension before normalization
        self.input_projection = nn.Linear(input_dim, output_dim)
        
        self.feature_transform = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.output_dim = output_dim
        
    def forward(self, x):
        """
        x: shape (B, input_dim)
        returns: (B, output_dim)
        """
        # Project input to desired output dimension first
        x_projected = self.input_projection(x)       # (B, output_dim)
        
        # Extract style features from original input
        style_features = self.style_encoder(x)       # (B, hidden_dim//2)
        
        # Generate style normalization parameters for projected features
        gamma = torch.sigmoid(self.style_gamma(style_features)) + 0.5   # (B, output_dim)
        beta = self.style_beta(style_features)                         # (B, output_dim)
        
        # Normalize projected features per-sample over the feature dimension
        mean = x_projected.mean(dim=1, keepdim=True)   # (B, 1)
        std = x_projected.std(dim=1, keepdim=True) + 1e-5  # (B, 1)
        normalized = (x_projected - mean) / std         # (B, output_dim)
        
        # Apply style-specific scaling and shifting elementwise
        style_normalized = normalized * gamma + beta   # (B, output_dim)
        
        # Final transformation expects (B, output_dim)
        output = self.feature_transform(style_normalized)
        
        return output

class EnhancedHybridModel(nn.Module):
    def __init__(self, resnet_backbone, num_classes, dinov2_dim=768, adapter_dim=256):
        super().__init__()
        self.resnet = resnet_backbone
        self.resnet_dim = 2048
        
        # Style Normalization Network for CNN features
        self.snn = StyleNormalizationNetwork(
            input_dim=self.resnet_dim,
            hidden_dim=512,
            output_dim=512
        )
        
        # Feature fusion: ViT + CNN + SNN
        self.feature_fusion = nn.Sequential(
            nn.Linear(dinov2_dim + self.resnet_dim + 512, adapter_dim),  # Added SNN dimension
            nn.BatchNorm1d(adapter_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim // 2),
            nn.BatchNorm1d(adapter_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(adapter_dim // 2, num_classes)
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(adapter_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, resnet_input, dinov2_features, domain_alpha=1.0):
        # Extract ResNet features
        resnet_features = self.resnet(resnet_input)
        resnet_features = resnet_features.squeeze(3).squeeze(2)
        
        # Apply Style Normalization Network to CNN features
        snn_features = self.snn(resnet_features)
        
        # Combine all features: ViT + CNN + SNN
        combined_features = torch.cat([dinov2_features, resnet_features, snn_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        class_logits = self.classifier(fused_features)
        rev_features = grad_reverse(fused_features, domain_alpha)
        domain_logits = self.domain_classifier(rev_features)
        
        return class_logits, domain_logits, fused_features

# DINOv2 model for feature extraction
class DINOv2FeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize(518),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._load_dinov2()
    
    def _load_dinov2(self):
        """Load DINOv2 model for feature extraction"""
        try:
            # Try to load the specific architecture used in training
            self.model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True, num_classes=0)
            print("Loaded DINOv2 model successfully")
        except:
            # Fallback to standard DINOv2
            try:
                self.model = timm.create_model('vit_base_patch14_dinov2', pretrained=True, num_classes=0)
                print("Loaded standard DINOv2 model as fallback")
            except Exception as e:
                print(f"Failed to load DINOv2 model: {e}")
                self.model = None
                return
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self, image):
        if self.model is None:
            # Return random features as fallback (shouldn't happen in normal operation)
            return torch.randn(1, 768).to(self.device)
        
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            features = self.model(img_tensor)
            if isinstance(features, (tuple, list)):
                features = features[0]
            return features.flatten(1)

# ============================================================================
# Novel3 Model Implementation
# ============================================================================

class Novel3(BaseModel):
    def __init__(self, model_path='saved_models/Novel3/best_enhanced_model.pth', 
                 label_map_path='saved_models/Novel3/label_mappings.json'):
        super().__init__()
        
        print(f"Loading Novel3 (DANN+Contrastive+SNN+Hybrid) model from {model_path} on {self.device}...")
        
        try:
            # Load label map
            print(f" Loading label map from {label_map_path}...")
            if not os.path.exists(label_map_path):
                raise FileNotFoundError(f"Label map not found at {label_map_path}")
                
            with open(label_map_path, 'r') as f:
                label_map_data = json.load(f)
                print(f" Label map structure: {list(label_map_data.keys())}")
            
            # Extract the mapped_to_original mapping
            if 'mapped_to_original' in label_map_data:
                mapped_to_original = label_map_data['mapped_to_original']
                print(f" Found mapped_to_original with {len(mapped_to_original)} entries")
                print(f" Sample: {list(mapped_to_original.items())[:5]}")
                
                # Build the mapping from index to class ID
                self.idx_to_class = {}
                for idx_str, class_id in mapped_to_original.items():
                    try:
                        idx = int(idx_str)
                        self.idx_to_class[idx] = str(class_id)
                    except (ValueError, TypeError) as e:
                        print(f" Warning: Could not convert index {idx_str}: {e}")
                        self.idx_to_class[idx_str] = str(class_id)
            else:
                raise KeyError("Expected 'mapped_to_original' key in label mapping file")
            
            num_classes = len(self.idx_to_class)
            print(f" Final label map: {num_classes} classes")
            print(f" Index range: {min(self.idx_to_class.keys())} to {max(self.idx_to_class.keys())}")

            # Load model
            model_path_full = model_path
            if not os.path.exists(model_path_full):
                raise FileNotFoundError(f"Model file not found at {model_path_full}")
                
            print(f" Loading model from {model_path_full}")
            print(f" Creating Novel3 model with {num_classes} classes")
            
            # Initialize ResNet backbone
            resnet_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            resnet_backbone = nn.Sequential(*list(resnet_backbone.children())[:-1])
            
            # Initialize the enhanced hybrid model
            self.model = EnhancedHybridModel(resnet_backbone, num_classes=num_classes)
            
            # Load DINOv2 feature extractor
            self.dinov2_extractor = DINOv2FeatureExtractor(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path_full, map_location=self.device)
            print(f" Checkpoint type: {type(checkpoint)}")
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Training checkpoint format
                    state_dict = checkpoint['model_state_dict']
                    print(f" Loading from model_state_dict with {len(state_dict)} parameters")
                    
                    # Load state dict with strict=False to handle any minor mismatches
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        print(f" Missing keys: {missing_keys}")
                    if unexpected_keys:
                        print(f" Unexpected keys: {unexpected_keys}")
                    print(" Loaded from model_state_dict with strict=False")
                else:
                    # Assume it's the model state dict directly
                    try:
                        self.model.load_state_dict(checkpoint, strict=False)
                        print(" Loaded direct state dict with strict=False")
                    except Exception as e:
                        print(f" Error loading state dict: {e}")
                        raise
            else:
                # Direct state dict
                try:
                    self.model.load_state_dict(checkpoint, strict=False)
                    print(" Loaded direct state dict with strict=False")
                except Exception as e:
                    print(f" Error loading state dict: {e}")
                    raise
            
            self.model.to(self.device)
            self.model.eval()
            
            # Verify model
            with torch.no_grad():
                # Test with dummy inputs
                test_resnet_input = torch.randn(1, 3, 224, 224).to(self.device)
                test_dinov2_features = torch.randn(1, 768).to(self.device)
                test_output, _, _ = self.model(test_resnet_input, test_dinov2_features)
                print(f" Model output shape: {test_output.shape}")
                print(f" Model has {test_output.shape[1]} output classes")
            
            print(" Novel3 model loaded successfully!")
            
            # Image transforms - ResNet transform (for the main model input)
            self.resnet_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # DINOv2 transform is handled by the DINOv2FeatureExtractor

        except Exception as e:
            print(f" Novel3 model initialization failed: {e}")
            import traceback
            traceback.print_exc()
            print(" Cannot use Novel3 model due to initialization error")
            self.model = None
            self.dinov2_extractor = None

    def predict(self, image, class_to_name_map):
        if image is None or self.model is None or self.dinov2_extractor is None:
            return {"Novel3 model not available": 1.0}, None

        try:
            # Convert image if needed
            if hasattr(image, 'mode') and image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract DINOv2 features
            dinov2_features = self.dinov2_extractor.extract_features(image)
            
            # Process image for ResNet
            resnet_input = self.resnet_transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions (domain_alpha=0.0 for inference - no gradient reversal)
            with torch.no_grad():
                class_logits, _, _ = self.model(resnet_input, dinov2_features, domain_alpha=0.0)
                probabilities = torch.softmax(class_logits, dim=1).cpu().numpy()[0]
            
            # Get top 5 predictions
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            results = {}
            top_class_id = None
            
            print(f"Novel3 - Top 5 indices: {top5_indices}")
            print(f"Novel3 - Probabilities: {[f'{probabilities[i]:.4f}' for i in top5_indices]}")
            
            for i, idx in enumerate(top5_indices):
                if idx in self.idx_to_class:
                    class_id = self.idx_to_class[idx]
                    species_name = convert_class_to_name(class_id, class_to_name_map)
                    score = float(probabilities[idx])
                    
                    # Use shared format function
                    display_name = format_prediction(class_id, species_name)
                    results[display_name] = score
                    
                    # Store the top prediction class ID for herbarium image
                    if i == 0:  # First (top) prediction
                        top_class_id = class_id
                    
                    print(f"  Index {idx} -> Class ID {class_id} -> Species '{species_name}'")
                else:
                    # Fallback if index not in mapping
                    print(f"Warning: Index {idx} not found in idx_to_class. Available indices: {list(self.idx_to_class.keys())}")
                    species_name = f"Class {idx}"
                    score = float(probabilities[idx])
                    results[species_name] = score
            
            # Get herbarium image for top prediction
            herbarium_image_path = get_herbarium_image(top_class_id) if top_class_id else None
            
            return results, herbarium_image_path
            
        except Exception as e:
            print(f" Novel3 prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {f"Prediction error: {str(e)}": 1.0}, None

# ============================================================================
# Novel2 Model Components (DANN+Hybrid+Focal+Adaptive)
# ============================================================================

class EnhancedHybridModel_Novel2(nn.Module):
    def __init__(self, num_classes, dinov2_dim=768, adapter_dim=256):
        super().__init__()
        
        # ResNet-50 backbone
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove final classification layer
        self.resnet_dim = 2048
        
        # DINOv2 backbone - we'll compute features separately
        self.dinov2_dim = dinov2_dim
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.dinov2_dim + self.resnet_dim, adapter_dim),
            nn.BatchNorm1d(adapter_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim // 2),
            nn.BatchNorm1d(adapter_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(adapter_dim // 2, num_classes)
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(adapter_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 2)
        )
        
    def forward(self, resnet_input, dinov2_features, domain_alpha=1.0):
        # Extract ResNet features
        resnet_features = self.resnet(resnet_input)
        resnet_features = resnet_features.squeeze(3).squeeze(2)
        
        # Combine with DINOv2 features
        combined_features = torch.cat([dinov2_features, resnet_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # Classification
        class_logits = self.classifier(fused_features)
        
        # Domain classification with gradient reversal
        rev_features = grad_reverse(fused_features, domain_alpha)
        domain_logits = self.domain_classifier(rev_features)
        
        return class_logits, domain_logits

# ============================================================================
# Novel2 Model Implementation
# ============================================================================

class Novel2(BaseModel):
    def __init__(self, model_path='saved_models/Novel2/best_model.pth', 
                 label_map_path='saved_models/Novel2/label_mapping.json'):
        super().__init__()
        
        print(f"Loading Novel2 (DANN+Hybrid+Focal+Adaptive) model from {model_path} on {self.device}...")
        
        try:
            # Load label map
            print(f" Loading label map from {label_map_path}...")
            if not os.path.exists(label_map_path):
                raise FileNotFoundError(f"Label map not found at {label_map_path}")
                
            with open(label_map_path, 'r') as f:
                label_map_data = json.load(f)
                print(f" Label map format: {len(label_map_data)} entries")
                print(f" Sample of label map: {list(label_map_data.items())[:5]}")
            
            # Build the mapping from index to class ID
            self.idx_to_class = {}
            for class_id, index in label_map_data.items():
                try:
                    idx = int(index)
                    self.idx_to_class[idx] = str(class_id)
                except (ValueError, TypeError) as e:
                    print(f" Warning: Could not convert index {index}: {e}")
                    self.idx_to_class[index] = str(class_id)
            
            num_classes = len(self.idx_to_class)
            print(f" Final label map: {num_classes} classes")
            print(f" Index range: {min(self.idx_to_class.keys())} to {max(self.idx_to_class.keys())}")

            # Load model
            model_path_full = model_path
            if not os.path.exists(model_path_full):
                raise FileNotFoundError(f"Model file not found at {model_path_full}")
                
            print(f" Loading model from {model_path_full}")
            print(f" Creating Novel2 model with {num_classes} classes")
            
            # Initialize model with correct number of classes
            self.model = EnhancedHybridModel_Novel2(num_classes=num_classes)
            
            # Load DINOv2 feature extractor
            self.dinov2_extractor = DINOv2FeatureExtractor(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path_full, map_location=self.device)
            print(f" Checkpoint type: {type(checkpoint)}")
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Training checkpoint format
                    state_dict = checkpoint['model_state_dict']
                    print(f" Loading from model_state_dict with {len(state_dict)} parameters")
                    
                    # Load state dict with strict=False to handle any minor mismatches
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        print(f" Missing keys: {missing_keys}")
                    if unexpected_keys:
                        print(f" Unexpected keys: {unexpected_keys}")
                    print(" Loaded from model_state_dict with strict=False")
                else:
                    # Assume it's the model state dict directly
                    try:
                        self.model.load_state_dict(checkpoint, strict=False)
                        print(" Loaded direct state dict with strict=False")
                    except Exception as e:
                        print(f" Error loading state dict: {e}")
                        raise
            else:
                # Direct state dict
                try:
                    self.model.load_state_dict(checkpoint, strict=False)
                    print(" Loaded direct state dict with strict=False")
                except Exception as e:
                    print(f" Error loading state dict: {e}")
                    raise
            
            self.model.to(self.device)
            self.model.eval()
            
            # Verify model
            with torch.no_grad():
                # Test with dummy inputs
                test_resnet_input = torch.randn(1, 3, 224, 224).to(self.device)
                test_dinov2_features = torch.randn(1, 768).to(self.device)
                test_output, _ = self.model(test_resnet_input, test_dinov2_features)
                print(f" Model output shape: {test_output.shape}")
                print(f" Model has {test_output.shape[1]} output classes")
            
            print(" Novel2 model loaded successfully!")
            
            # Image transforms - ResNet transform (for the main model input)
            self.resnet_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # DINOv2 transform is handled by the DINOv2FeatureExtractor

        except Exception as e:
            print(f" Novel2 model initialization failed: {e}")
            import traceback
            traceback.print_exc()
            print(" Cannot use Novel2 model due to initialization error")
            self.model = None
            self.dinov2_extractor = None

    def predict(self, image, class_to_name_map):
        if image is None or self.model is None or self.dinov2_extractor is None:
            return {"Novel2 model not available": 1.0}, None

        try:
            # Convert image if needed
            if hasattr(image, 'mode') and image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract DINOv2 features
            dinov2_features = self.dinov2_extractor.extract_features(image)
            
            # Process image for ResNet
            resnet_input = self.resnet_transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions (domain_alpha=0.0 for inference - no gradient reversal)
            with torch.no_grad():
                class_logits, _ = self.model(resnet_input, dinov2_features, domain_alpha=0.0)
                probabilities = torch.softmax(class_logits, dim=1).cpu().numpy()[0]
            
            # Get top 5 predictions
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            results = {}
            top_class_id = None
            
            print(f"Novel2 - Top 5 indices: {top5_indices}")
            print(f"Novel2 - Probabilities: {[f'{probabilities[i]:.4f}' for i in top5_indices]}")
            
            for i, idx in enumerate(top5_indices):
                if idx in self.idx_to_class:
                    class_id = self.idx_to_class[idx]
                    species_name = convert_class_to_name(class_id, class_to_name_map)
                    score = float(probabilities[idx])
                    
                    # Use shared format function
                    display_name = format_prediction(class_id, species_name)
                    results[display_name] = score
                    
                    # Store the top prediction class ID for herbarium image
                    if i == 0:  # First (top) prediction
                        top_class_id = class_id
                    
                    print(f"  Index {idx} -> Class ID {class_id} -> Species '{species_name}'")
                else:
                    # Fallback if index not in mapping
                    print(f"Warning: Index {idx} not found in idx_to_class. Available indices: {list(self.idx_to_class.keys())}")
                    species_name = f"Class {idx}"
                    score = float(probabilities[idx])
                    results[species_name] = score
            
            # Get herbarium image for top prediction
            herbarium_image_path = get_herbarium_image(top_class_id) if top_class_id else None
            
            return results, herbarium_image_path
            
        except Exception as e:
            print(f" Novel2 prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {f"Prediction error: {str(e)}": 1.0}, None

# ============================================================================
# Novel1 Model Components (DANN+Triplet+Prototype+ArcFace)
# ============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer from DANN paper"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

class ArcFaceClassifier(nn.Module):
    """ArcFace classifier for improved angular margin"""
    def __init__(self, in_features, num_classes, s=30.0, m=0.50):
        super(ArcFaceClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        # Normalize features and weights
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(features, weight)

        if self.training and labels is not None:
            # Get angles
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            # cos(theta + m)
            phi = cosine * torch.cos(torch.tensor(self.m)) - sine * torch.sin(torch.tensor(self.m))
            # One-hot encoding
            one_hot = torch.zeros(cosine.size(), device=features.device)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
            # Apply margin only to ground truth class
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            output = cosine * self.s

        return output

class PrototypeLearning(nn.Module):
    """Maintains class prototypes using Exponential Moving Average (EMA)"""
    def __init__(self, num_classes, embedding_dim, momentum=0.9):
        super(PrototypeLearning, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.momentum = momentum
        self.register_buffer('prototypes', torch.zeros(num_classes, embedding_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))

    @torch.no_grad()
    def update_prototypes(self, embeddings, labels):
        for class_idx in range(self.num_classes):
            mask = (labels == class_idx)
            if mask.sum() > 0:
                class_embeddings = embeddings[mask]
                class_mean = class_embeddings.mean(dim=0)
                if self.prototype_counts[class_idx] == 0:
                    self.prototypes[class_idx] = class_mean
                else:
                    self.prototypes[class_idx] = (
                        self.momentum * self.prototypes[class_idx] +
                        (1 - self.momentum) * class_mean
                    )
                self.prototype_counts[class_idx] += mask.sum()

    def forward(self, embeddings, labels):
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        distances = torch.cdist(embeddings_norm, prototypes_norm, p=2)
        gt_distances = distances[torch.arange(embeddings.size(0)), labels]
        return gt_distances.mean()

    def get_nearest_prototype(self, embeddings, top_k=5):
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        distances = torch.cdist(embeddings_norm, prototypes_norm, p=2)
        topk_distances, topk_indices = torch.topk(distances, k=top_k, dim=1, largest=False)
        return topk_distances, topk_indices

class CrossDomainPlantIdentifier(nn.Module):
    """Complete model integrating DINOv2, DANN, ArcFace, Triplet Loss, and Prototype Learning"""
    def __init__(self, num_classes, embedding_dim=384, arcface_s=30.0, arcface_m=0.50):
        super(CrossDomainPlantIdentifier, self).__init__()

        # DINOv2 backbone - create from scratch since we'll load the full state dict
        try:
            self.backbone = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=0)
        except:
            # Fallback to standard DINOv2
            self.backbone = timm.create_model('vit_base_patch14_dinov2', pretrained=False, num_classes=0)
        
        self.feature_dim = 768  # Standard DINOv2 feature dimension

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # ArcFace classifier
        self.classifier = ArcFaceClassifier(embedding_dim, num_classes, s=arcface_s, m=arcface_m)

        # Domain classifier for DANN
        self.gradient_reversal = GradientReversalLayer(lambda_=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # Binary: herbarium vs field
        )

        # Prototype learning
        self.prototype_learning = PrototypeLearning(num_classes, embedding_dim, momentum=0.9)

    def forward(self, x, labels=None, return_features=False):
        # Extract features with DINOv2
        features = self.backbone(x)
        
        # Project to embedding space
        embeddings = self.projection(features)

        # Classification with ArcFace
        class_logits = self.classifier(embeddings, labels)

        # Domain classification with gradient reversal
        reversed_embeddings = self.gradient_reversal(embeddings)
        domain_logits = self.domain_classifier(reversed_embeddings)

        output = {
            'class_logits': class_logits,
            'domain_logits': domain_logits,
            'embeddings': embeddings,
            'features': features
        }

        return output

    def set_lambda(self, lambda_):
        """Set lambda for gradient reversal"""
        self.gradient_reversal.set_lambda(lambda_)

# ============================================================================
# Novel1 Model Implementation
# ============================================================================

class Novel1(BaseModel):
    def __init__(self, model_path='saved_models/Novel1/DomainAdapV2_model_weights.pth', 
                 label_map_path='saved_models/Novel1/label_mapping.json'):
        super().__init__()
        
        print(f"Loading Novel1 (DANN+Triplet+Prototype+ArcFace) model from {model_path} on {self.device}...")
        
        try:
            # Load label map - FIXED: Handle the nested structure
            print(f" Loading label map from {label_map_path}...")
            if not os.path.exists(label_map_path):
                raise FileNotFoundError(f"Label map not found at {label_map_path}")
                
            with open(label_map_path, 'r') as f:
                label_map_data = json.load(f)
                print(f" Label map structure: {list(label_map_data.keys())}")
            
            # Extract the idx_to_class_id mapping from the nested structure
            if 'idx_to_class_id' in label_map_data:
                idx_to_class_data = label_map_data['idx_to_class_id']
                print(f" Found idx_to_class_id with {len(idx_to_class_data)} entries")
                print(f" Sample: {list(idx_to_class_data.items())[:5]}")
                
                # Build the mapping from index to class ID
                self.idx_to_class = {}
                for idx_str, class_id in idx_to_class_data.items():
                    try:
                        idx = int(idx_str)
                        self.idx_to_class[idx] = str(class_id)
                    except (ValueError, TypeError) as e:
                        print(f" Warning: Could not convert index {idx_str}: {e}")
                        self.idx_to_class[idx_str] = str(class_id)
            else:
                raise KeyError("Expected 'idx_to_class_id' key in label mapping file")
            
            num_classes = len(self.idx_to_class)
            print(f" Final label map: {num_classes} classes")
            print(f" Index range: {min(self.idx_to_class.keys())} to {max(self.idx_to_class.keys())}")

            # Load model
            model_path_full = model_path
            if not os.path.exists(model_path_full):
                raise FileNotFoundError(f"Model file not found at {model_path_full}")
                
            print(f" Loading model from {model_path_full}")
            print(f" Creating Novel1 model with {num_classes} classes")
            
            # Initialize model with correct number of classes
            self.model = CrossDomainPlantIdentifier(
                num_classes=num_classes,  # This should be 100
                embedding_dim=384,
                arcface_s=30.0,
                arcface_m=0.50
            )
            
            # Load checkpoint
            checkpoint = torch.load(model_path_full, map_location=self.device)
            print(f" Checkpoint type: {type(checkpoint)}")
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Training checkpoint format
                    state_dict = checkpoint['model_state_dict']
                    print(f" Loading from model_state_dict with {len(state_dict)} parameters")
                    
                    # Load state dict with strict=False to handle any minor mismatches
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        print(f" Missing keys: {missing_keys}")
                    if unexpected_keys:
                        print(f" Unexpected keys: {unexpected_keys}")
                    print(" Loaded from model_state_dict with strict=False")
                else:
                    # Assume it's the model state dict directly
                    try:
                        self.model.load_state_dict(checkpoint, strict=False)
                        print(" Loaded direct state dict with strict=False")
                    except Exception as e:
                        print(f" Error loading state dict: {e}")
                        raise
            else:
                # Direct state dict
                try:
                    self.model.load_state_dict(checkpoint, strict=False)
                    print(" Loaded direct state dict with strict=False")
                except Exception as e:
                    print(f" Error loading state dict: {e}")
                    raise
            
            self.model.to(self.device)
            self.model.eval()
            
            # Verify model
            with torch.no_grad():
                test_input = torch.randn(1, 3, 518, 518).to(self.device)
                test_output = self.model(test_input)
                print(f" Model output shape: {test_output['class_logits'].shape}")
                print(f" Model has {test_output['class_logits'].shape[1]} output classes")
            
            print(" Novel1 model loaded successfully!")
            
            # Image transforms - same as validation in training script
            self.transform = transforms.Compose([
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        except Exception as e:
            print(f" Novel1 model initialization failed: {e}")
            import traceback
            traceback.print_exc()
            print(" Cannot use Novel1 model due to initialization error")
            self.model = None

    def predict(self, image, class_to_name_map):
        if image is None or self.model is None:
            return {"Novel1 model not available": 1.0}, None

        try:
            # Convert image if needed
            if hasattr(image, 'mode') and image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(img_tensor)
                class_logits = outputs['class_logits']
                probabilities = torch.softmax(class_logits, dim=1).cpu().numpy()[0]
            
            # Get top 5 predictions
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            results = {}
            top_class_id = None
            
            print(f"Novel1 - Top 5 indices: {top5_indices}")
            print(f"Novel1 - Probabilities: {[f'{probabilities[i]:.4f}' for i in top5_indices]}")
            
            for i, idx in enumerate(top5_indices):
                if idx in self.idx_to_class:
                    class_id = self.idx_to_class[idx]
                    species_name = convert_class_to_name(class_id, class_to_name_map)
                    score = float(probabilities[idx])
                    
                    # Use shared format function
                    display_name = format_prediction(class_id, species_name)
                    results[display_name] = score
                    
                    # Store the top prediction class ID for herbarium image
                    if i == 0:  # First (top) prediction
                        top_class_id = class_id
                    
                    print(f"  Index {idx} -> Class ID {class_id} -> Species '{species_name}'")
                else:
                    # Fallback if index not in mapping
                    print(f"Warning: Index {idx} not found in idx_to_class. Available indices: {list(self.idx_to_class.keys())}")
                    species_name = f"Class {idx}"
                    score = float(probabilities[idx])
                    results[species_name] = score
            
            # Get herbarium image for top prediction
            herbarium_image_path = get_herbarium_image(top_class_id) if top_class_id else None
            
            return results, herbarium_image_path
            
        except Exception as e:
            print(f" Novel1 prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {f"Prediction error: {str(e)}": 1.0}, None

# ============================================================================
# BaseLine1 Model (EfficientNet Mix-stream)
# ============================================================================

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.7):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Multiple dropout layers for stronger regularization
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(in_features)
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.bn(features)
        features = self.dropout1(features)
        features = self.dropout2(features)
        logits = self.classifier(features)
        return logits

class Baseline1(BaseModel):
    def __init__(self, model_path='saved_models/Baseline1/best_efficientnet_model.pth', 
                 label_map_path='saved_models/Baseline1/label_map.json'):
        super().__init__()
        
        print(f"Loading Baseline1 (Mix-stream) model from {model_path} on {self.device}...")
        
        try:
            # Load label map - FIXED: Your format is {"class_id": index}, need to invert it
            print(f" Loading label map from {label_map_path}...")
            if not os.path.exists(label_map_path):
                raise FileNotFoundError(f"Label map not found at {label_map_path}")
                
            with open(label_map_path, 'r') as f:
                label_map_data = json.load(f)
                print(f" Original label map format: {len(label_map_data)} entries")
                print(f" Sample of original label map: {list(label_map_data.items())[:5]}")
            
            # INVERT THE MAPPING: Your format is {class_id: index}, we need {index: class_id}
            self.idx_to_class = {}
            for class_id, index in label_map_data.items():
                self.idx_to_class[int(index)] = str(class_id)  # Convert index to int, keep class_id as string
            
            print(f" Inverted label map: {len(self.idx_to_class)} entries")
            print(f" Sample of inverted label map: {list(self.idx_to_class.items())[:5]}")
            print(f" Index range: {min(self.idx_to_class.keys())} to {max(self.idx_to_class.keys())}")

            # Load model - USE CUSTOM ARCHITECTURE WITH CORRECT NUMBER OF CLASSES
            model_path_full = model_path
            if not os.path.exists(model_path_full):
                raise FileNotFoundError(f"Model file not found at {model_path_full}")
                
            print(f" Loading model from {model_path_full}")
            num_classes = len(self.idx_to_class)
            print(f" Creating EfficientNet-B0 with {num_classes} classes")
            
            # Use the custom architecture that matches your training script
            self.model = EfficientNetClassifier(num_classes=num_classes, dropout=0.7)
            
            # Load checkpoint
            checkpoint = torch.load(model_path_full, map_location=self.device)
            print(f" Checkpoint type: {type(checkpoint)}")
            print(f" Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
            
            # Load model state with better error handling
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Load the complete state dict
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(" Loaded from model_state_dict")
                else:
                    # Assume it's the model state dict directly
                    try:
                        self.model.load_state_dict(checkpoint)
                        print(" Loaded direct state dict")
                    except Exception as e:
                        print(f" Error loading state dict: {e}")
                        # Try with strict=False as last resort
                        self.model.load_state_dict(checkpoint, strict=False)
                        print(" Loaded with strict=False")
            else:
                # Direct state dict
                try:
                    self.model.load_state_dict(checkpoint)
                    print(" Loaded direct state dict")
                except Exception as e:
                    print(f" Error loading state dict: {e}")
                    self.model.load_state_dict(checkpoint, strict=False)
                    print(" Loaded with strict=False")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Verify the model has correct number of output classes
            with torch.no_grad():
                test_input = torch.randn(1, 3, 224, 224).to(self.device)
                test_output = self.model(test_input)
                print(f" Model output shape: {test_output.shape}")
                print(f" Model has {test_output.shape[1]} output classes")
            
            print(" Baseline1 model loaded successfully!")
            
            # Image transforms - match your training
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        except Exception as e:
            print(f" Baseline1 model initialization failed: {e}")
            import traceback
            traceback.print_exc()
            print(" Cannot use Baseline1 model due to initialization error")
            # Don't create fallback - just mark as unavailable
            self.model = None

    def predict(self, image, class_to_name_map):
        if image is None or self.model is None:
            return {"Baseline1 model not available": 1.0}, None

        try:
            # Convert image if needed
            if hasattr(image, 'mode') and image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # Get top 5 predictions
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            results = {}
            top_class_id = None
            
            print(f"Baseline1 - Top 5 indices: {top5_indices}")
            print(f"Baseline1 - Probabilities: {[f'{probabilities[i]:.4f}' for i in top5_indices]}")
            
            for i, idx in enumerate(top5_indices):
                if idx in self.idx_to_class:
                    class_id = self.idx_to_class[idx]  # This now gives us the original class ID like "12254"
                    species_name = convert_class_to_name(class_id, class_to_name_map)
                    score = float(probabilities[idx])
                    
                    # Use shared format function
                    display_name = format_prediction(class_id, species_name)
                    results[display_name] = score
                    
                    # Store the top prediction class ID for herbarium image
                    if i == 0:  # First (top) prediction
                        top_class_id = class_id
                    
                    print(f"  Index {idx} -> Class ID {class_id} -> Species '{species_name}'")
                else:
                    # Fallback if index not in mapping
                    print(f"Warning: Index {idx} not found in idx_to_class. Available indices: {list(self.idx_to_class.keys())}")
                    species_name = f"Class {idx}"
                    score = float(probabilities[idx])
                    results[species_name] = score
            
            # Get herbarium image for top prediction
            herbarium_image_path = get_herbarium_image(top_class_id) if top_class_id else None
            
            return results, herbarium_image_path
            
        except Exception as e:
            print(f" Baseline1 prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {f"Prediction error: {str(e)}": 1.0}, None

# ============================================================================
# BaseLine2 Model - USING SHARED FORMAT FUNCTION
# ============================================================================

class DomainAwareNormalizer:
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scalers = {
            'herbarium': StandardScaler(),
            'photo': StandardScaler()
        }
        self.is_fitted = False
    
    def transform(self, features, domain_labels):
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        normalized = np.zeros_like(features)
        herbarium_mask = domain_labels == 0
        photo_mask = domain_labels == 1
        
        if herbarium_mask.sum() > 0:
            normalized[herbarium_mask] = self.scalers['herbarium'].transform(
                features[herbarium_mask]
            )
        if photo_mask.sum() > 0:
            normalized[photo_mask] = self.scalers['photo'].transform(
                features[photo_mask]
            )
        return normalized

class Baseline2(BaseModel):
    def __init__(self, model_dir='saved_models/Baseline2'):
        super().__init__()
        self.model_dir = Path(model_dir)
        
        print(f"Loading Baseline2 model from {self.model_dir} on {self.device}...")
        
        # 1. Load Metadata
        with open(self.model_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
            
        with open(self.model_dir / 'class_mapping.json', 'r') as f:
            mapping = json.load(f)
            self.idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}

        # 2. Load DINOv2
        self._load_dinov2()
        
        # 3. Load Normalizer
        with open(self.model_dir / 'normalizer.pkl', 'rb') as f:
            self.normalizer = pickle.load(f)
            
        # 4. Load Classifier (SVM)
        with open(self.model_dir / 'classifier.pkl', 'rb') as f:
            self.classifier = pickle.load(f)

        # 5. Define Transforms (Same as training)
        self.transform = transforms.Compose([
            transforms.Resize(560),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("Baseline2 model ready for inference.")

    def _load_dinov2(self):
        arch_name = self.metadata.get('dinov2_arch', 'vit_base_patch14_reg4_dinov2.lvd142m')
        
        # Create architecture
        try:
            self.model = timm.create_model(
                arch_name,
                pretrained=False,
                num_classes=0,
                img_size=518,
                dynamic_img_size=True
            )
        except:
            self.model = timm.create_model(
                arch_name,
                pretrained=False,
                num_classes=0
            )

        # Load weights
        checkpoint = torch.load(self.model_dir / 'dinov2_model.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image, class_to_name_map):
        if image is None:
            return None, None

        # 1. Preprocess Image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 2. Extract Features
        with torch.no_grad():
            features = self.model(img_tensor).cpu().numpy()
            
        # 3. Normalize
        domain_labels = np.ones(1) 
        norm_features = self.normalizer.transform(features, domain_labels)
        
        # 4. Classification & Top 5
        decision_scores = self.classifier.decision_function(norm_features)[0]
        
        # Convert decision scores to probabilities using Softmax
        scores_tensor = torch.tensor(decision_scores)
        probabilities = F.softmax(scores_tensor, dim=0).numpy()
        
        # Get Top 5
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        
        results = {}
        top_class_id = None
        
        for i, idx in enumerate(top5_indices):
            class_id = self.idx_to_class[idx]
            species_name = convert_class_to_name(class_id, class_to_name_map)
            score = float(probabilities[idx])
            
            # Use shared format function
            display_name = format_prediction(class_id, species_name)
            results[display_name] = score
            
            # Store the top prediction class ID for herbarium image
            if i == 0:  # First (top) prediction
                top_class_id = class_id
            
        # Get herbarium image for top prediction
        herbarium_image_path = get_herbarium_image(top_class_id) if top_class_id else None
        
        return results, herbarium_image_path

# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    def __init__(self):
        self.models = {}
        
        # Initialize models in the desired order: Baseline1, Baseline2, Novel1, Novel2, Novel3
        model_initializers = [
            ("Baseline1 (Mix-stream)", Baseline1),
            ("Baseline2 (DINOv2 + SVM)", Baseline2),
            ("Novel1 (DANN+Triplet+Prototype+ArcFace)", Novel1),
            ("Novel2 (DANN+Hybrid+Focal+Adaptive)", Novel2),
            ("Novel3 (DANN+Contrastive+SNN+Hybrid)", Novel3)
        ]
        
        for model_name, model_class in model_initializers:
            try:
                model_instance = model_class()
                if model_instance.model is not None:
                    self.models[model_name] = model_instance
                    print(f"{model_name} model added successfully")
                else:
                    print(f"{model_name} model not available")
            except Exception as e:
                print(f"{model_name} initialization failed: {e}")
        
        # Set default model to the first available model in our ordered list
        available_models = list(self.models.keys())
        if available_models:
            self.current_model = available_models[0]
        else:
            self.current_model = None
        
        print(f"Available models: {available_models}")
        print(f"Default model: {self.current_model}")
        
    def set_model(self, model_name):
        if model_name in self.models:
            self.current_model = model_name
            print(f"Switched to model: {model_name}")
            return True
        else:
            print(f"Model {model_name} not found!")
            return False
    
    def predict(self, image, class_to_name_map):
        if self.current_model and self.current_model in self.models:
            return self.models[self.current_model].predict(image, class_to_name_map)
        return {"No model available": 1.0}, None
    
    def get_available_models(self):
        return list(self.models.keys())

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Initialize system
model_manager = ModelManager()
species_name_map = load_species_names('species_list.txt')

def classify_image(image, model_name):
    # Update model if selection changed
    model_manager.set_model(model_name)
    
    # Perform prediction - now returns both results and herbarium image path
    results, herbarium_image_path = model_manager.predict(image, species_name_map)
    
    # Load herbarium image if available
    herbarium_image = None
    if herbarium_image_path and os.path.exists(herbarium_image_path):
        try:
            herbarium_image = Image.open(herbarium_image_path)
            print(f"Loaded herbarium image: {herbarium_image_path}")
        except Exception as e:
            print(f"Error loading herbarium image: {e}")
            herbarium_image = None
    
    return results, herbarium_image

# Get available models for dropdown
available_models = model_manager.get_available_models()

sorted_species_list = sorted(list(species_name_map.values()))
display_data = [[name] for name in sorted_species_list]

with gr.Blocks(title="Plant Classifier") as demo:
    gr.Markdown("## 🌿 Plant Classification System")
    gr.Markdown("Upload an image of a plant to identify its species. Select your preferred model architecture.")
    
    with gr.Row():
        # Left column: input
        with gr.Column(scale=1):
            # Model selection dropdown - now in the desired order
            model_selector = gr.Dropdown(
                choices=available_models,
                value=available_models[0] if available_models else None,
                label="Select Model Architecture",
                info="Choose which model to use for classification",
                interactive=True
            )

            input_image = gr.Image(
                type="pil", 
                label="Upload Image",
                sources=["upload", "clipboard"],
                height=400
            )
            
            classify_btn = gr.Button(
                "Classify Plant", 
                variant="primary", 
                elem_id="classify_btn",
                size="lg"
            )

        # Right column: output
        with gr.Column(scale=1):
            output_labels = gr.Label(
                num_top_classes=5, 
                label="Top 5 Confidence Scores"
            )
            
            # New section for herbarium image
            gr.Markdown("### 📚 Herbarium Reference")
            herbarium_image = gr.Image(
                label="Herbarium Image of Top Prediction",
                type="pil",
                height=300
            )

    # Event Listener
    classify_btn.click(
        fn=classify_image, 
        inputs=[input_image, model_selector], 
        outputs=[output_labels, herbarium_image]
    )

    # Display class list
    gr.Markdown("---") 
    
    # Accordion allows the user to hide/show the list
    with gr.Accordion("📚 Available Species List (Click to Expand)", open=False):
        
        # Dataframe allows searching and scrolling
        gr.Dataframe(
            headers=["Species Name"],
            value=display_data,
            interactive=False, # Read-only
            wrap=True
        )

if __name__ == "__main__":
    demo.launch()