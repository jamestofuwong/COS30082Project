import matplotlib
matplotlib.use('Agg') 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import csv
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
import warnings
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
warnings.filterwarnings('ignore')

# ==============================================================
# Model Architecture (EXACTLY matching updated training script)
# ==============================================================

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
        
        # Final transformation
        output = self.feature_transform(style_normalized)
        
        return output

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

class EnhancedHybridModel(nn.Module):
    def __init__(self, resnet_backbone, num_classes, dinov2_dim=768, adapter_dim=256):
        super().__init__()
        self.resnet = resnet_backbone
        self.resnet_dim = 2048
        
        # Style Normalization Network for CNN features (EXACTLY as in training)
        self.snn = StyleNormalizationNetwork(
            input_dim=self.resnet_dim,
            hidden_dim=512,
            output_dim=512
        )
        
        # Feature fusion: ViT + CNN + SNN (3328 = 768 + 2048 + 512)
        self.feature_fusion = nn.Sequential(
            nn.Linear(dinov2_dim + self.resnet_dim + 512, adapter_dim),  # Updated to match training
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
        
    def forward(self, resnet_input, dinov2_features, domain_alpha=1.0, return_features=False):
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
        
        if return_features:
            return class_logits, domain_logits, fused_features
        return class_logits, domain_logits

# ==============================================================
# Data Loading (unchanged)
# ==============================================================
def precompute_dinov2_test_features(dataset_root, test_list, device, save_path="dinov2_test_features.pth"):
    """Precompute DINOv2 features for test set"""
    if os.path.exists(save_path):
        print("Loading precomputed DINOv2 test features...")
        features_dict = torch.load(save_path)
        return features_dict
    
    print("Precomputing DINOv2 features for test set...")
    
    import kagglehub
    import timm
    
    dinov2_path = kagglehub.model_download("juliostat/dinov2_patch14_reg4_onlyclassifier_then_all/PyTorch/default")
    ckpt_file = None
    for root, dirs, files in os.walk(dinov2_path):
        for f in files:
            if f.endswith(".pth.tar"):
                ckpt_file = os.path.join(root, f)
                break
        if ckpt_file:
            break

    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    arch_name = checkpoint.get("arch", "vit_base_patch14_reg4_dinov2.lvd142m")
    dinov2_model = timm.create_model(arch_name, pretrained=False)
    state_dict = checkpoint.get("state_dict_ema", checkpoint.get("state_dict"))
    dinov2_model.load_state_dict(state_dict, strict=False)
    dinov2_model.eval().to(device)
    
    dinov2_transform = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features_dict = {}
    test_root = Path(dataset_root)
    
    with open(test_list, 'r') as f:
        test_files = [line.strip() for line in f if line.strip()]
    
    with torch.no_grad():
        for test_file in tqdm(test_files, desc="Extracting test features"):
            img_path = test_root / test_file
            if img_path.exists():
                img = Image.open(img_path).convert('RGB')
                img = dinov2_transform(img).unsqueeze(0).to(device)
                
                features = dinov2_model(img)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                features = features.flatten(1).cpu()
                
                features_dict[test_file] = features
            else:
                print(f"Warning: Test image not found: {img_path}")
    
    torch.save(features_dict, save_path)
    print("DINOv2 test features saved!")
    
    return features_dict

class TestDataset(Dataset):
    def __init__(self, dataset_root, test_list, groundtruth_file, dinov2_features_dict, label_map):
        self.dataset_root = Path(dataset_root)
        self.dinov2_features_dict = dinov2_features_dict
        self.label_map = label_map
        
        with open(test_list, 'r') as f:
            self.test_files = [line.strip() for line in f if line.strip()]
        
        self.labels = {}
        with open(groundtruth_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path = parts[0]
                    label = int(parts[1])
                    self.labels[img_path] = label
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {len(self.test_files)} test samples")
    
    def __len__(self):
        return len(self.test_files)
    
    def __getitem__(self, idx):
        test_file = self.test_files[idx]
        
        img_path = self.dataset_root / test_file
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        
        dinov2_features = self.dinov2_features_dict[test_file]
        if dinov2_features.dim() > 2:
            dinov2_features = dinov2_features.squeeze(0)
        
        original_label = self.labels[test_file]
        mapped_label = self.label_map[original_label]
        
        return img_tensor, dinov2_features, mapped_label, test_file, original_label

def get_label_map(dataset_root, train_list):
    """Get the label mapping from training data"""
    original_labels = []
    with open(train_list, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label = int(parts[1])
                original_labels.append(label)
    
    unique_labels = sorted(set(original_labels))
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    reverse_map = {new_label: old_label for old_label, new_label in label_map.items()}
    num_classes = len(unique_labels)
    
    print(f"Label mapping: {num_classes} classes")
    return label_map, reverse_map, num_classes

def load_class_lists():
    """Load paired and unpaired class lists"""
    paired_classes = set()
    with open('dataset/list/class_with_pairs.txt', 'r') as f:
        for line in f:
            class_id = line.strip()
            if class_id:
                paired_classes.add(int(class_id))
    
    unpaired_classes = set()
    with open('dataset/list/class_without_pairs.txt', 'r') as f:
        for line in f:
            class_id = line.strip()
            if class_id:
                unpaired_classes.add(int(class_id))
    
    return paired_classes, unpaired_classes

# ==============================================================
# Visualization Functions - 6 PLOTS (unchanged)
# ==============================================================

def plot_confusion_matrix(y_true, y_pred, title, top1_acc, top5_acc, save_path, max_classes=50):
    """Plot confusion matrix with accuracy scores displayed"""
    
    # Limit to top N classes for visualization
    from collections import Counter
    class_counts = Counter(y_true)
    top_classes = [cls for cls, _ in class_counts.most_common(max_classes)]
    
    # Filter to top classes
    mask = np.isin(y_true, top_classes)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_classes)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot heatmap
    sns.heatmap(cm_normalized, xticklabels=top_classes, yticklabels=top_classes,cmap='Blues', 
                cbar_kws={'label': 'Proportion'}, ax=ax, fmt='.2f', square=True)
    
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add accuracy text box in top-right corner
    textstr = f'Top-1 Acc: {top1_acc:.4f} ({top1_acc*100:.2f}%)\nTop-5 Acc: {top5_acc:.4f} ({top5_acc*100:.2f}%)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top',
            horizontalalignment='right', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix: {save_path}")


def plot_top5_distribution(y_true, y_proba, class_mapping, title, top1_acc, top5_acc, save_path):
    """Plot Top-5 prediction distribution"""
    
    # Get top-5 predictions for each sample
    top5_indices = np.argsort(y_proba, axis=1)[:, -5:][:, ::-1]
    
    # Check if true label is in top-5
    correct_positions = []
    for i, true_idx in enumerate(y_true):
        if true_idx in top5_indices[i]:
            pos = np.where(top5_indices[i] == true_idx)[0][0] + 1
            correct_positions.append(pos)
        else:
            correct_positions.append(6)  # Not in top-5
    
    # Count occurrences
    position_counts = [correct_positions.count(i) for i in range(1, 7)]
    position_labels = ['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5', 'Not in Top-5']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#95a5a6']
    bars = ax.bar(position_labels, position_counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count in zip(bars, position_counts):
        height = bar.get_height()
        percentage = (count / len(y_true)) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prediction Position', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add accuracy text box
    textstr = f'Top-1 Acc: {top1_acc:.4f} ({top1_acc*100:.2f}%)\nTop-5 Acc: {top5_acc:.4f} ({top5_acc*100:.2f}%)\nTotal Samples: {len(y_true)}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=11, verticalalignment='top', 
            horizontalalignment='right', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Top-5 distribution: {save_path}")

# ==============================================================
# Evaluation Function (updated model loading)
# ==============================================================
def evaluate_subset(model, test_loader, device, subset_name, paired_classes, unpaired_classes, reverse_map, output_dir):
    """Evaluate model on a specific subset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_original_labels = []
    all_filenames = []
    
    with torch.no_grad():
        for images, dinov2_features, labels, filenames, original_labels in tqdm(test_loader, desc=f"Evaluating {subset_name}"):
            images = images.to(device)
            dinov2_features = dinov2_features.to(device)
            
            if dinov2_features.dim() > 2:
                dinov2_features = dinov2_features.squeeze(1)
            
            # Use the updated forward method - don't request features during evaluation
            class_logits, _ = model(images, dinov2_features, domain_alpha=0.0, return_features=False)
            probs = F.softmax(class_logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(class_logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_original_labels.append(original_labels.cpu().numpy())
            all_filenames.extend(filenames)
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_proba = np.concatenate(all_probs)
    y_orig = np.concatenate(all_original_labels)
    
    # Filter for subset
    if subset_name == "Paired Classes":
        mask = np.array([label in paired_classes for label in y_orig])
    elif subset_name == "Unpaired Classes":
        mask = np.array([label in unpaired_classes for label in y_orig])
    else:  # Overall
        mask = np.ones(len(y_true), dtype=bool)
    
    y_pred_sub = y_pred[mask]
    y_true_sub = y_true[mask]
    y_proba_sub = y_proba[mask]
    y_orig_sub = y_orig[mask]
    
    if len(y_true_sub) == 0:
        print(f"No samples for {subset_name}")
        return {'top1': 0, 'top5': 0, 'samples': 0}
    
    # Calculate accuracies
    top1_acc = (y_pred_sub == y_true_sub).sum() / len(y_true_sub)
    top5_acc = top_k_accuracy_score(y_true_sub, y_proba_sub, k=5, labels=np.arange(y_proba_sub.shape[1]))
    
    print(f"\n{subset_name} Results:")
    print(f"  Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"  Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    print(f"  Samples: {len(y_true_sub)}")

    # Convert to percentages for plotting
    top1_acc_pct = top1_acc * 100
    top5_acc_pct = top5_acc * 100

    # Plot 1: Confusion Matrix
    cm_path = os.path.join(output_dir, f'{subset_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plot_confusion_matrix(
        y_true_sub, y_pred_sub, 
        f'{subset_name} - Confusion Matrix (Top {min(50, len(np.unique(y_true)))} Classes)',
        top1_acc, top5_acc,
        cm_path
    )
    
    # Plot 2: Top-5 Distribution
    top5_path = os.path.join(output_dir, f'{subset_name.lower().replace(" ", "_")}_top5_distribution.png')
    plot_top5_distribution(
        y_true_sub, y_proba_sub, reverse_map,
        f'{subset_name} - Top-5 Prediction Distribution',
        top1_acc, top5_acc,
        top5_path
    )
    
    return {
        'top1': top1_acc_pct, 
        'top5': top5_acc_pct,  
        'samples': len(y_true_sub),
        'y_true': y_true_sub,
        'y_pred': y_pred_sub,
        'y_proba': y_proba_sub,
        'y_orig': y_orig_sub
    }

def evaluate_test_set(model_path, dataset_root, test_list, groundtruth_file, train_list, device=None):
    """Main evaluation function with 6 visualizations"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Evaluating model: {model_path}")
    print(f"Using device: {device}")
    
    # Load label mapping and class lists
    label_map, reverse_map, num_classes = get_label_map(dataset_root, train_list)
    paired_classes, unpaired_classes = load_class_lists()
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    print("Model checkpoint loaded")
    
    resnet_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet_backbone = nn.Sequential(*list(resnet_backbone.children())[:-1])
    resnet_backbone = resnet_backbone.to(device)
    
    # Create model with EXACT architecture from training
    model = EnhancedHybridModel(resnet_backbone, num_classes).to(device)
    
    # Load state dict - should now match perfectly
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")
    
    # Precompute DINOv2 features
    dinov2_features_dict = precompute_dinov2_test_features(dataset_root, test_list, device)
    
    # Create test dataset
    test_dataset = TestDataset(dataset_root, test_list, groundtruth_file, 
                               dinov2_features_dict, label_map)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Create output directory
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate all subsets
    overall = evaluate_subset(model, test_loader, device, "Overall", paired_classes, unpaired_classes, reverse_map, output_dir)
    paired = evaluate_subset(model, test_loader, device, "Paired Classes", paired_classes, unpaired_classes, reverse_map, output_dir)
    unpaired = evaluate_subset(model, test_loader, device, "Unpaired Classes", paired_classes, unpaired_classes, reverse_map, output_dir)

    summary_file = os.path.join(output_dir, "test_results_summary.csv")
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Subset', 'Top-1 Accuracy (%)', 'Top-5 Accuracy (%)', 'Samples'])
        writer.writerow(['Overall', f'{overall["top1"]:.2f}', f'{overall["top5"]:.2f}', overall['samples']])
        writer.writerow(['Paired Classes', f'{paired["top1"]:.2f}', f'{paired["top5"]:.2f}', paired['samples']])
        writer.writerow(['Unpaired Classes', f'{unpaired["top1"]:.2f}', f'{unpaired["top5"]:.2f}', unpaired['samples']])
    
    print(f"Summary saved to: {summary_file}")
    
    print("EVALUATION COMPLETE!")

    print(f"Overall Top-1: {overall['top1']:.2f}% | Top-5: {overall['top5']:.2f}%")
    print(f"Paired Top-1: {paired['top1']:.2f}% | Top-5: {paired['top5']:.2f}%")
    print(f"Unpaired Top-1: {unpaired['top1']:.2f}% | Top-5: {unpaired['top5']:.2f}%")
    print(f"\nAll results saved to: {output_dir}/")
    
    return overall['top1'], overall['top5']

def main():
    """Main evaluation function"""
    config = {
        'model_path': 'testing/best_enhanced_model.pth',
        'dataset_root': 'dataset',
        'test_list': 'dataset/list/test.txt',
        'groundtruth_file': 'dataset/list/groundtruth.txt',
        'train_list': 'dataset/list/train.txt'
    }
    
    # Check if files exist
    if not os.path.exists(config['model_path']):
        print(f"Model file not found: {config['model_path']}")
        print("Please train the model first or update the model_path")
        return
    
    for file_key in ['test_list', 'groundtruth_file', 'train_list']:
        if not os.path.exists(config[file_key]):
            print(f"File not found: {config[file_key]}")
            return
    
    print("Starting Test Set Evaluation")
    print("="*60)
    
    # Run evaluation
    top1_acc, top5_acc = evaluate_test_set(**config)
    
    print("\nEvaluation Complete!")
    print(f"Overall Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Overall Top-5 Accuracy: {top5_acc:.2f}%")

if __name__ == "__main__":
    main()