import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tqdm import tqdm
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DINOV2 MODEL
# ============================================================================

def load_dinov2_from_kagglehub(device):
    """Load DINOv2 plant-pretrained model as frozen feature extractor"""
    import kagglehub
    import timm
    
    print("Loading DINOv2 Feature Extractor from KaggleHub")

    dinov2_path = kagglehub.model_download(
        "juliostat/dinov2_patch14_reg4_onlyclassifier_then_all/PyTorch/default"
    )

    # Find checkpoint file
    ckpt_file = None
    for root, dirs, files in os.walk(dinov2_path):
        for f in files:
            if f.endswith(".pth.tar"):
                ckpt_file = os.path.join(root, f)
                break
        if ckpt_file:
            break

    if not ckpt_file:
        raise FileNotFoundError("No .pth.tar checkpoint found.")

    print(f"Found checkpoint: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    arch_name = checkpoint.get("arch", "vit_base_patch14_reg4_dinov2.lvd142m")
    print(f"Architecture: {arch_name}")

    try:
        # Load with num_classes=0 to get feature extractor without classifier
        dinov2_model = timm.create_model(
            arch_name, 
            pretrained=False,
            num_classes=0,  # removes the classification head
            img_size=518,  
            dynamic_img_size=True 
        )
    except Exception as e:
        print(f"Failed to create {arch_name}, trying default config: {e}")
        try:
            dinov2_model = timm.create_model(
                "vit_base_patch14_reg4_dinov2.lvd142m", 
                pretrained=False,
                num_classes=0,
                img_size=518
            )
        except Exception as e2:
            print(f"Trying without img_size parameter: {e2}")
            dinov2_model = timm.create_model(
                arch_name, 
                pretrained=False,
                num_classes=0
            )

    # Load state dict
    state_dict = checkpoint.get("state_dict_ema", checkpoint.get("state_dict"))
    
    # Filter out classifier head weights if present
    feature_state_dict = {
        k: v for k, v in state_dict.items() 
        if not k.startswith('head.') and not k.startswith('fc.')
    }
    
    # Load with strict=False to handle missing classifier weights
    missing, unexpected = dinov2_model.load_state_dict(feature_state_dict, strict=False)
    
    if missing:
        print(f"Missing keys (expected if removing head): {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)} keys")
    
    dinov2_model.eval()

    # Freeze all parameters
    for param in dinov2_model.parameters():
        param.requires_grad = False
    
    dinov2_model.to(device)
    
    # Get feature dimension
    feature_dim = dinov2_model.num_features
    print(f"DINOv2 loaded successfully")
    print(f"Feature dimension: {feature_dim}")
    print(f"Expected input size: {dinov2_model.patch_embed.img_size}")
    
    return dinov2_model, feature_dim


# ============================================================================
# 2. DUAL-DOMAIN DATASET LOADER
# ============================================================================

class DualDomainDataset(Dataset):
    """Load dataset with two domains: herbarium and photo"""
    
    def __init__(self, root_dir, list_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Read the list file
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path = parts[0]
                    class_name = parts[1]
                    
                    # Determine domain from path
                    if 'herbarium' in img_path:
                        domain = 'herbarium'
                    elif 'photo' in img_path:
                        domain = 'photo'
                    else:
                        continue
                    
                    # Build class mapping
                    if class_name not in self.class_to_idx:
                        idx = len(self.class_to_idx)
                        self.class_to_idx[class_name] = idx
                        self.idx_to_class[idx] = class_name
                    
                    full_path = os.path.join(root_dir, img_path)
                    self.samples.append({
                        'path': full_path,
                        'class_idx': self.class_to_idx[class_name],
                        'class_name': class_name,
                        'domain': domain,
                        'domain_idx': 0 if domain == 'herbarium' else 1
                    })
        
        print(f"\nDataset loaded:")
        print(f"Total samples: {len(self.samples)}")
        print(f"Total classes: {len(self.class_to_idx)}")
        print(f"Herbarium samples: {sum(1 for s in self.samples if s['domain'] == 'herbarium')}")
        print(f"Photo samples: {sum(1 for s in self.samples if s['domain'] == 'photo')}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")

            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': sample['class_idx'],
            'domain': sample['domain_idx'],
            'domain_name': sample['domain']
        }


# ============================================================================
# 3. FEATURE EXTRACTION
# ============================================================================

def extract_features_by_domain(model, dataloader, device):
    """Extract features and organize by domain"""
    
    herb_features, herb_labels = [], []
    photo_features, photo_labels = [], []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch['image'].to(device)
            labels = batch['label'].numpy()
            domains = batch['domain'].numpy()
            
            features = model(images).cpu().numpy()
            
            # Separate by domain
            herb_mask = domains == 0
            photo_mask = domains == 1
            
            if herb_mask.sum() > 0:
                herb_features.append(features[herb_mask])
                herb_labels.append(labels[herb_mask])
            
            if photo_mask.sum() > 0:
                photo_features.append(features[photo_mask])
                photo_labels.append(labels[photo_mask])
    
    herb_features = np.vstack(herb_features) if herb_features else np.array([])
    herb_labels = np.concatenate(herb_labels) if herb_labels else np.array([])
    photo_features = np.vstack(photo_features) if photo_features else np.array([])
    photo_labels = np.concatenate(photo_labels) if photo_labels else np.array([])
    
    print(f"\nFeature extraction complete:")
    print(f"Herbarium: {herb_features.shape}")
    print(f"Photo: {photo_features.shape}")
    
    return (herb_features, herb_labels), (photo_features, photo_labels)


# ============================================================================
# 4. DOMAIN-AWARE FEATURE NORMALIZATION
# ============================================================================

class DomainAwareNormalizer:
    """Normalize features separately for each domain"""
    
    def __init__(self):
        self.scalers = {
            'herbarium': StandardScaler(),
            'photo': StandardScaler()
        }
        self.is_fitted = False
    
    def fit(self, features, domain_labels):
        """Fit normalizers on training data"""
        herbarium_mask = domain_labels == 0
        photo_mask = domain_labels == 1
        
        if herbarium_mask.sum() > 0:
            self.scalers['herbarium'].fit(features[herbarium_mask])
        if photo_mask.sum() > 0:
            self.scalers['photo'].fit(features[photo_mask])
        
        self.is_fitted = True
        print("Domain-aware normalizers fitted")
    
    def transform(self, features, domain_labels):
        """Apply domain-specific normalization"""
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
    
    def fit_transform(self, features, domain_labels):
        self.fit(features, domain_labels)
        return self.transform(features, domain_labels)


# ============================================================================
# 5. SYNTHETIC TARGET-LIKE EMBEDDINGS
# ============================================================================

class SyntheticEmbeddingGenerator:
    """Generate synthetic photo-like embeddings from herbarium samples"""
    
    def __init__(self, method='interpolation', alpha=0.5):
        self.method = method
        self.alpha = alpha
        self.domain_shift = None
        self.class_shifts = {}
    
    def fit(self, herb_features, photo_features, herb_labels=None, photo_labels=None):
        """Learn domain shift from herbarium to photo"""
        
        if self.method == 'interpolation':
            self.domain_shift = photo_features.mean(axis=0) - herb_features.mean(axis=0)
            print(f"Learned global domain shift (interpolation)")
        
        elif self.method == 'class_conditional' and herb_labels is not None and photo_labels is not None:
            paired_classes = set(herb_labels) & set(photo_labels)
            
            for cls in paired_classes:
                herb_mask = herb_labels == cls
                photo_mask = photo_labels == cls
                
                if herb_mask.sum() > 0 and photo_mask.sum() > 0:
                    class_shift = (photo_features[photo_mask].mean(axis=0) - 
                                  herb_features[herb_mask].mean(axis=0))
                    self.class_shifts[cls] = class_shift
            
            self.domain_shift = photo_features.mean(axis=0) - herb_features.mean(axis=0)
            
            print(f"Learned class-conditional shifts for {len(self.class_shifts)} paired classes")
            print(f"Learned global fallback shift for unpaired classes")
        
        elif self.method == 'pca_projection':
            combined = np.vstack([herb_features, photo_features])
            self.pca = PCA(n_components=min(512, herb_features.shape[1]))
            self.pca.fit(combined)
            
            herb_pca = self.pca.transform(herb_features)
            photo_pca = self.pca.transform(photo_features)
            self.domain_shift = photo_pca.mean(axis=0) - herb_pca.mean(axis=0)
            print(f"Learned domain shift (PCA projection)")
    
    def generate(self, herb_features, labels=None, num_synthetic=None):
        """Generate synthetic photo-like embeddings from herbarium"""
        
        if num_synthetic is None:
            num_synthetic = len(herb_features)
        
        if num_synthetic < len(herb_features):
            indices = np.random.choice(len(herb_features), num_synthetic, replace=False)
            herb_features = herb_features[indices]
            if labels is not None:
                labels = labels[indices]
        elif num_synthetic > len(herb_features):
            indices = np.random.choice(len(herb_features), num_synthetic, replace=True)
            herb_features = herb_features[indices]
            if labels is not None:
                labels = labels[indices]
        
        if self.method == 'interpolation':
            noise = np.random.randn(*herb_features.shape) * 0.1
            synthetic = herb_features + self.alpha * self.domain_shift + noise
        
        elif self.method == 'class_conditional':
            synthetic = np.zeros_like(herb_features)
            
            if labels is not None:
                for i, (feat, label) in enumerate(zip(herb_features, labels)):
                    if label in self.class_shifts:
                        shift = self.class_shifts[label]
                    else:
                        shift = self.domain_shift
                    
                    noise = np.random.randn(len(feat)) * 0.1
                    synthetic[i] = feat + self.alpha * shift + noise
            else:
                noise = np.random.randn(*herb_features.shape) * 0.1
                synthetic = herb_features + self.alpha * self.domain_shift + noise
        
        elif self.method == 'pca_projection':
            herb_pca = self.pca.transform(herb_features)
            shifted_pca = herb_pca + self.alpha * self.domain_shift
            noise_pca = np.random.randn(*shifted_pca.shape) * 0.05
            synthetic = self.pca.inverse_transform(shifted_pca + noise_pca)
        
        return synthetic


# ============================================================================
# 6. MIX-STREAM TRAINING WITH CLASS-AWARE DOMAIN MIXING
# ============================================================================

class DomainMixingStrategy:
    """Mix herbarium and photo features for training with unpaired class handling"""
    
    def __init__(self, mix_ratio=0.5, use_synthetic=True):
        self.mix_ratio = mix_ratio
        self.use_synthetic = use_synthetic
        self.synthetic_generator = None
        self.paired_classes = None
        self.herb_only_classes = None
    
    def analyze_class_pairing(self, herb_labels, photo_labels):
        """Identify paired and unpaired classes"""
        herb_classes = set(herb_labels)
        photo_classes = set(photo_labels)
        
        self.paired_classes = herb_classes & photo_classes
        self.herb_only_classes = herb_classes - photo_classes
        
        print(f"\nClass Pairing Analysis:")
        print(f"Total herbarium classes: {len(herb_classes)}")
        print(f"Total photo classes: {len(photo_classes)}")
        print(f"Paired classes (have both domains): {len(self.paired_classes)}")
        print(f"Herbarium-only classes: {len(self.herb_only_classes)}")
        
        return self.paired_classes, self.herb_only_classes
    
    def prepare_mixed_data(self, herb_features, herb_labels, 
                          photo_features, photo_labels,
                          synthetic_ratio=0.3,
                          unpaired_weight=1.0):
        """Create mixed training data with class-aware domain mixing"""
        
        if self.paired_classes is None:
            self.analyze_class_pairing(herb_labels, photo_labels)
        
        mixed_features = []
        mixed_labels = []
        mixed_domains = []
        mixed_sources = []
        
        # Handle PAIRED classes
        paired_herb_mask = np.isin(herb_labels, list(self.paired_classes))
        paired_photo_mask = np.isin(photo_labels, list(self.paired_classes))
        
        paired_herb_features = herb_features[paired_herb_mask]
        paired_herb_labels = herb_labels[paired_herb_mask]
        paired_photo_features = photo_features[paired_photo_mask]
        paired_photo_labels = photo_labels[paired_photo_mask]
        
        if len(paired_herb_features) > 0:
            n_paired_herb = int(len(paired_herb_features) * self.mix_ratio)
            indices = np.random.choice(len(paired_herb_features), n_paired_herb, replace=False)
            mixed_features.append(paired_herb_features[indices])
            mixed_labels.append(paired_herb_labels[indices])
            mixed_domains.append(np.zeros(n_paired_herb))
            mixed_sources.extend(['herb_paired'] * n_paired_herb)
        
        if len(paired_photo_features) > 0:
            n_paired_photo = int(len(paired_photo_features) * (1 - self.mix_ratio))
            indices = np.random.choice(len(paired_photo_features), n_paired_photo, replace=False)
            mixed_features.append(paired_photo_features[indices])
            mixed_labels.append(paired_photo_labels[indices])
            mixed_domains.append(np.ones(n_paired_photo))
            mixed_sources.extend(['photo'] * n_paired_photo)
        
        # Handle UNPAIRED classes
        unpaired_herb_mask = np.isin(herb_labels, list(self.herb_only_classes))
        unpaired_herb_features = herb_features[unpaired_herb_mask]
        unpaired_herb_labels = herb_labels[unpaired_herb_mask]
        
        if len(unpaired_herb_features) > 0:
            n_unpaired = int(len(unpaired_herb_features) * unpaired_weight)
            if n_unpaired > len(unpaired_herb_features):
                indices = np.random.choice(len(unpaired_herb_features), n_unpaired, replace=True)
            else:
                indices = np.random.choice(len(unpaired_herb_features), n_unpaired, replace=False)
            
            mixed_features.append(unpaired_herb_features[indices])
            mixed_labels.append(unpaired_herb_labels[indices])
            mixed_domains.append(np.zeros(n_unpaired))
            mixed_sources.extend(['herb_unpaired'] * n_unpaired)
        
        # Generate SYNTHETIC samples
        if self.use_synthetic and self.synthetic_generator is not None and len(unpaired_herb_features) > 0:
            n_synthetic_unpaired = int(len(unpaired_herb_features) * synthetic_ratio)
            
            synthetic_features = self.synthetic_generator.generate(
                unpaired_herb_features, num_synthetic=n_synthetic_unpaired
            )
            synthetic_indices = np.random.choice(len(unpaired_herb_labels), 
                                                n_synthetic_unpaired, replace=True)
            
            mixed_features.append(synthetic_features)
            mixed_labels.append(unpaired_herb_labels[synthetic_indices])
            mixed_domains.append(np.ones(n_synthetic_unpaired) * 0.5)
            mixed_sources.extend(['synthetic_unpaired'] * n_synthetic_unpaired)
        
        if self.use_synthetic and self.synthetic_generator is not None and len(paired_herb_features) > 0:
            n_synthetic_paired = int(len(paired_herb_features) * synthetic_ratio * 0.5)
            
            synthetic_features = self.synthetic_generator.generate(
                paired_herb_features, num_synthetic=n_synthetic_paired
            )
            synthetic_indices = np.random.choice(len(paired_herb_labels), 
                                                n_synthetic_paired, replace=True)
            
            mixed_features.append(synthetic_features)
            mixed_labels.append(paired_herb_labels[synthetic_indices])
            mixed_domains.append(np.ones(n_synthetic_paired) * 0.5)
            mixed_sources.extend(['synthetic_paired'] * n_synthetic_paired)
        
        # Concatenate and shuffle
        X = np.vstack(mixed_features)
        y = np.concatenate(mixed_labels)
        domains = np.concatenate(mixed_domains)
        
        shuffle_idx = np.random.permutation(len(X))
        
        # Print summary
        source_counts = Counter(mixed_sources)
        
        print(f"\nMixed dataset created:")
        print(f"Paired classes:")
        print(f" - Herbarium: {source_counts.get('herb_paired', 0)} samples")
        print(f" - Photo: {source_counts.get('photo', 0)} samples")
        print(f" - Synthetic: {source_counts.get('synthetic_paired', 0)} samples")
        print(f"Unpaired classes (herbarium only):")
        print(f" - Herbarium: {source_counts.get('herb_unpaired', 0)} samples")
        print(f" - Synthetic: {source_counts.get('synthetic_unpaired', 0)} samples")
        print(f"Total: {len(X)} samples")
        
        return X[shuffle_idx], y[shuffle_idx], domains[shuffle_idx]


# ============================================================================
# 7. COMPLETE TRAINING PIPELINE
# ============================================================================

def train_with_domain_adaptation(dinov2_model, train_loader, device, use_class_conditional=True):
    """Complete training pipeline with domain adaptation"""
    
    # Step 1: Extract Features by Domain
    print("STEP 1: Extract Features by Domain")
    (herb_features, herb_labels), (photo_features, photo_labels) = \
        extract_features_by_domain(dinov2_model, train_loader, device)
    
    # Step 2: Domain-aware normalization
    print("STEP 2: Domain-Aware Normalization")
    normalizer = DomainAwareNormalizer()
    
    all_features = np.vstack([herb_features, photo_features])
    all_domains = np.concatenate([
        np.zeros(len(herb_features)),
        np.ones(len(photo_features))
    ])
    
    normalized_features = normalizer.fit_transform(all_features, all_domains)
    
    herb_features_norm = normalized_features[:len(herb_features)]
    photo_features_norm = normalized_features[len(herb_features):]
    
    # Step 3: Generate synthetic embeddings
    print("STEP 3: Generate Synthetic Target-like Embeddings")
    
    if use_class_conditional:
        synthetic_gen = SyntheticEmbeddingGenerator(
            method='class_conditional', 
            alpha=0.7
        )
        synthetic_gen.fit(
            herb_features_norm, photo_features_norm,
            herb_labels, photo_labels
        )
    else:
        synthetic_gen = SyntheticEmbeddingGenerator(
            method='interpolation', 
            alpha=0.7
        )
        synthetic_gen.fit(herb_features_norm, photo_features_norm)
    
    # Step 4: Class-aware mix-stream training
    print("STEP 4: Class-Aware Mix-Stream Training Data Preparation")

    mixer = DomainMixingStrategy(mix_ratio=0.5, use_synthetic=True)
    mixer.synthetic_generator = synthetic_gen
    
    X_train, y_train, domains_train = mixer.prepare_mixed_data(
        herb_features_norm, herb_labels,
        photo_features_norm, photo_labels,
        synthetic_ratio=0.5,
        unpaired_weight=1.2
    )
    
    print(f"\nFinal training set: {X_train.shape}")
    
    return X_train, y_train, domains_train, normalizer, mixer


# ============================================================================
# 8. SAVE TRAINED MODELS
# ============================================================================

def save_trained_model(classifier, normalizer, train_dataset, save_dir='saved_models'):
    """Save the complete trained model pipeline"""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    print("Saving Trained Models")
    
    # 1. Save classifier
    classifier_path = save_path / 'classifier.pkl'
    with open(classifier_path, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Classifier saved to: {classifier_path}")
    
    # 2. Save normalizer
    normalizer_path = save_path / 'normalizer.pkl'
    with open(normalizer_path, 'wb') as f:
        pickle.dump(normalizer, f)
    print(f"Normalizer saved to: {normalizer_path}")
    
    # 3. Save class mappings
    class_mapping = {
        'class_to_idx': train_dataset.class_to_idx,
        'idx_to_class': train_dataset.idx_to_class
    }
    mapping_path = save_path / 'class_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Class mapping saved to: {mapping_path}")
    
    # 4. Save metadata
    metadata = {
        'num_classes': len(train_dataset.class_to_idx),
        'classifier_type': type(classifier).__name__,
        'feature_dim': classifier.n_features_in_ if hasattr(classifier, 'n_features_in_') else None
    }
    metadata_path = save_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    print("\nAll models saved successfully!")
    print(f"Save directory: {save_path.absolute()}")
    
    return save_path


# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(560), 
        transforms.CenterCrop(518), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # STEP 1: Load DINOv2 Model
    dinov2_model, feature_dim = load_dinov2_from_kagglehub(device)
    
    # STEP 2: Load Training Dataset
    print("Loading Training Dataset")

    
    train_dataset = DualDomainDataset(
        root_dir='dataset',
        list_file='dataset/list/train.txt',
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
   
    # STEP 3: Extract Features and Prepare Mix-Stream Training Data
    X_train, y_train, domains_train, normalizer, mixer = train_with_domain_adaptation(
        dinov2_model, train_loader, device, use_class_conditional=True
    )
    

    # STEP 4: Train Classifier
    print("Training Classifier")

    # Random Forest 
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    clf.fit(X_train, y_train)
    
    # # SVM
    # print("Training SVM Classifier...")
    # clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, verbose=False)
    # clf.fit(X_train, y_train)
    
    print("Classifier training complete!")
    

    # STEP 5: Save Trained Models
    save_dir = save_trained_model(
        classifier=clf,
        normalizer=normalizer,
        train_dataset=train_dataset,
        save_dir='saved_models'
    )
    
    print(f"\nTraining complete! Models saved to: {save_dir}")



if __name__ == "__main__":

    main()
