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
from collections import Counter
warnings.filterwarnings('ignore')

# ==============================================================
# Model Architecture (same as training)
# ==============================================================
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.7):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
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

# ==============================================================
# Data Loading
# ==============================================================
class SimpleDataset(Dataset):
    def __init__(self, root_dir, list_file):
        self.root_dir = Path(root_dir)
        self.samples = []
        
        # Load class lists
        self.classes_with_pairs = set()
        self.classes_without_pairs = set()
        with open('dataset/list/class_with_pairs.txt', 'r') as f:
            self.classes_with_pairs = set(int(line.strip()) for line in f)
        with open('dataset/list/class_without_pairs.txt', 'r') as f:
            self.classes_without_pairs = set(int(line.strip()) for line in f)
        
        # Create label mapping
        original_labels = []
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    label = int(parts[1])
                    original_labels.append(label)
        
        unique_labels = sorted(set(original_labels))
        self.label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        self.reverse_map = {new_label: old_label for old_label, new_label in self.label_map.items()}
        self.num_classes = len(unique_labels)
        
        # Load samples
        for line in open(list_file, 'r'):
            parts = line.strip().split()
            if len(parts) >= 2:
                img_path = parts[0]
                original_label = int(parts[1])
                full_path = self.root_dir / img_path
                if full_path.exists():
                    domain = 0 if 'herbarium' in img_path else 1
                    has_pair = original_label in self.classes_with_pairs
                    mapped_label = self.label_map[original_label]
                    self.samples.append((str(full_path), mapped_label, domain, has_pair, original_label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class TestDataset(Dataset):
    def __init__(self, dataset_root, test_list, groundtruth_file, label_map):
        self.dataset_root = Path(dataset_root)
        self.label_map = label_map
        
        # Load test files
        with open(test_list, 'r') as f:
            self.test_files = [line.strip() for line in f if line.strip()]
        
        # Load ground truth labels
        self.labels = {}
        with open(groundtruth_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path = parts[0]
                    label = int(parts[1])
                    self.labels[img_path] = label
        
        # Load class lists for paired/unpaired
        self.classes_with_pairs = set()
        self.classes_without_pairs = set()
        with open('dataset/list/class_with_pairs.txt', 'r') as f:
            self.classes_with_pairs = set(int(line.strip()) for line in f)
        with open('dataset/list/class_without_pairs.txt', 'r') as f:
            self.classes_without_pairs = set(int(line.strip()) for line in f)
        
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
        
        original_label = self.labels[test_file]
        mapped_label = self.label_map[original_label]
        
        # Determine if this sample belongs to paired/unpaired class
        has_pair = original_label in self.classes_with_pairs
        
        return img_tensor, mapped_label, test_file, original_label, has_pair

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
# Visualization Functions - 6 PLOTS
# ==============================================================

def plot_confusion_matrix(y_true, y_pred, title, top1_acc, top5_acc, save_path, max_classes=50):
    """Plot confusion matrix with accuracy scores displayed"""
    
    # Limit to top N classes for visualization
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
    sns.heatmap(cm_normalized, xticklabels=top_classes, yticklabels=top_classes, cmap='Blues', 
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

def plot_accuracy_comparison(results, save_path):
    """Plot accuracy comparison between subsets"""
    subsets = list(results.keys())
    top1_scores = [results[subset]['top1'] for subset in subsets]
    top5_scores = [results[subset]['top5'] for subset in subsets]
    
    x = np.arange(len(subsets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, top1_scores, width, label='Top-1 Accuracy', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, top5_scores, width, label='Top-5 Accuracy', color='lightcoral', edgecolor='black')
    
    ax.set_xlabel('Subsets', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison Across Subsets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subsets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved accuracy comparison: {save_path}")

# ==============================================================
# Evaluation Function
# ==============================================================
def evaluate_subset(model, test_loader, device, subset_name, paired_classes, unpaired_classes, reverse_map, output_dir):
    """Evaluate model on a specific subset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_original_labels = []
    all_filenames = []
    all_has_pairs = []
    
    with torch.no_grad():
        for images, labels, filenames, original_labels, has_pairs in tqdm(test_loader, desc=f"Evaluating {subset_name}"):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_original_labels.append(original_labels.cpu().numpy())
            all_filenames.extend(filenames)
            all_has_pairs.extend(has_pairs)
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_proba = np.concatenate(all_probs)
    y_orig = np.concatenate(all_original_labels)
    has_pairs_array = np.array(all_has_pairs)
    
    # Filter for subset
    if subset_name == "Paired Classes":
        mask = has_pairs_array  # Use the has_pairs information directly
    elif subset_name == "Unpaired Classes":
        mask = ~has_pairs_array  # Use the inverse of has_pairs
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
        f'{subset_name} - Confusion Matrix (Top {min(50, len(np.unique(y_true_sub)))} Classes)',
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

def evaluate_efficientnet_test_set(model_path, dataset_root, test_list, groundtruth_file, train_list, device=None):
    """Main evaluation function for EfficientNet with 6+ visualizations"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Evaluating EfficientNet model: {model_path}")
    print(f"Using device: {device}")
    
    # Load label mapping and class lists
    label_map, reverse_map, num_classes = get_label_map(dataset_root, train_list)
    paired_classes, unpaired_classes = load_class_lists()
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    print("Model checkpoint loaded")
    
    model = EfficientNetClassifier(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("EfficientNet model loaded successfully!")
    
    # Create test dataset
    test_dataset = TestDataset(dataset_root, test_list, groundtruth_file, label_map)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Create output directory
    output_dir = 'efficientnet_evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate all subsets
    overall = evaluate_subset(model, test_loader, device, "Overall", paired_classes, unpaired_classes, reverse_map, output_dir)
    paired = evaluate_subset(model, test_loader, device, "Paired Classes", paired_classes, unpaired_classes, reverse_map, output_dir)
    unpaired = evaluate_subset(model, test_loader, device, "Unpaired Classes", paired_classes, unpaired_classes, reverse_map, output_dir)

    # Plot accuracy comparison
    comparison_results = {
        'Overall': overall,
        'Paired': paired,
        'Unpaired': unpaired
    }
    comparison_path = os.path.join(output_dir, "accuracy_comparison.png")
    plot_accuracy_comparison(comparison_results, comparison_path)

    # Save detailed results
    summary_file = os.path.join(output_dir, "efficientnet_test_results_summary.csv")
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Subset', 'Top-1 Accuracy (%)', 'Top-5 Accuracy (%)', 'Samples'])
        writer.writerow(['Overall', f'{overall["top1"]:.2f}', f'{overall["top5"]:.2f}', overall['samples']])
        writer.writerow(['Paired Classes', f'{paired["top1"]:.2f}', f'{paired["top5"]:.2f}', paired['samples']])
        writer.writerow(['Unpaired Classes', f'{unpaired["top1"]:.2f}', f'{unpaired["top5"]:.2f}', unpaired['samples']])
    
    print(f"Summary saved to: {summary_file}")
    
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"Overall Top-1: {overall['top1']:.2f}% | Top-5: {overall['top5']:.2f}%")
    print(f"Paired Top-1: {paired['top1']:.2f}% | Top-5: {paired['top5']:.2f}%")
    print(f"Unpaired Top-1: {unpaired['top1']:.2f}% | Top-5: {unpaired['top5']:.2f}%")
    print(f"\nAll results saved to: {output_dir}/")
    
    return overall['top1'], overall['top5']

def main():
    """Main evaluation function for EfficientNet"""
    config = {
        'model_path': 'efficientnet_baseline/best_efficientnet_model.pth', 
        'dataset_root': 'dataset',
        'test_list': 'dataset/list/test.txt',
        'groundtruth_file': 'dataset/list/groundtruth.txt',
        'train_list': 'dataset/list/train.txt'
    }
    
    # Check if files exist
    if not os.path.exists(config['model_path']):
        print(f"Model file not found: {config['model_path']}")
        print("Please train the EfficientNet model first or update the model_path")
        return
    
    for file_key in ['test_list', 'groundtruth_file', 'train_list']:
        if not os.path.exists(config[file_key]):
            print(f"File not found: {config[file_key]}")
            return
    
    # Check for class list files
    class_files = ['dataset/list/class_with_pairs.txt', 'dataset/list/class_without_pairs.txt']
    for class_file in class_files:
        if not os.path.exists(class_file):
            print(f"Class list file not found: {class_file}")
            return
    
    print("Starting EfficientNet Test Set Evaluation")
    print("="*60)
    
    # Run evaluation
    top1_acc, top5_acc = evaluate_efficientnet_test_set(**config)
    
    print("\nEfficientNet Evaluation Complete!")
    print(f"Overall Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Overall Top-5 Accuracy: {top5_acc:.2f}%")

if __name__ == "__main__":
    main()