import matplotlib
matplotlib.use('Agg') 

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import csv  # Add this import
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

    try:
        dinov2_model = timm.create_model(
            arch_name, 
            pretrained=False,
            num_classes=0,
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

    state_dict = checkpoint.get("state_dict_ema", checkpoint.get("state_dict"))
    feature_state_dict = {
        k: v for k, v in state_dict.items() 
        if not k.startswith('head.') and not k.startswith('fc.')
    }
    
    dinov2_model.load_state_dict(feature_state_dict, strict=False)
    dinov2_model.eval()

    for param in dinov2_model.parameters():
        param.requires_grad = False
    
    dinov2_model.to(device)
    
    print(f"DINOv2 loaded successfully")
    
    return dinov2_model


# ============================================================================
# 2. DOMAIN-AWARE NORMALIZER
# ============================================================================

class DomainAwareNormalizer:
    """Normalize features separately for each domain"""
    
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
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
# 3. TEST DATASET LOADER
# ============================================================================

class TestDataset(Dataset):
    """Load test dataset with groundtruth"""
    
    def __init__(self, root_dir, groundtruth_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Read groundtruth file
        with open(groundtruth_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path = parts[0] 
                    class_name = parts[1] 
                    
                    full_path = os.path.join(root_dir, img_path)
                    
                    self.samples.append({
                        'path': full_path,
                        'class_name': class_name,
                        'img_name': os.path.basename(img_path)
                    })
        
        print(f"\nTest dataset loaded:")
        print(f"Total samples: {len(self.samples)}")
        
        # Count unique classes
        unique_classes = set(s['class_name'] for s in self.samples)
        print(f"   Unique classes: {len(unique_classes)}")
    
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
            'class_name': sample['class_name'],
            'img_name': sample['img_name'],
            'path': sample['path']
        }


# ============================================================================
# 4. LOAD CLASS LISTS
# ============================================================================

def load_class_lists():
    """Load paired and unpaired class lists"""
    

    print("Loading Class Lists")

    # Load paired classes
    paired_classes = set()
    with open('dataset/list/class_with_pairs.txt', 'r') as f:
        for line in f:
            class_id = line.strip()
            if class_id:
                paired_classes.add(class_id)
    
    print(f"Paired classes: {len(paired_classes)}")
    
    # Load unpaired classes
    unpaired_classes = set()
    with open('dataset/list/class_without_pairs.txt', 'r') as f:
        for line in f:
            class_id = line.strip()
            if class_id:
                unpaired_classes.add(class_id)
    
    print(f"Unpaired classes: {len(unpaired_classes)}")
    
    return paired_classes, unpaired_classes


# ============================================================================
# 5. LOAD SAVED MODELS
# ============================================================================

def load_saved_models(model_dir='saved_models'):
    """Load all saved model components"""
    
    model_path = Path(model_dir)
    

    print("Loading Saved Models")
    
    # 1. Load classifier
    classifier_path = model_path / 'classifier.pkl'
    with open(classifier_path, 'rb') as f:
        classifier = pickle.load(f)
    print(f"Classifier loaded from: {classifier_path}")
    
    # 2. Load normalizer
    normalizer_path = model_path / 'normalizer.pkl'
    with open(normalizer_path, 'rb') as f:
        normalizer = pickle.load(f)
    print(f"Normalizer loaded from: {normalizer_path}")
    
    # 3. Load class mapping
    mapping_path = model_path / 'class_mapping.json'
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    print(f"Class mapping loaded from: {mapping_path}")
    
    # 4. Load metadata
    metadata_path = model_path / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"Metadata loaded: {metadata}")
    
    return classifier, normalizer, class_mapping, metadata


# ============================================================================
# 6. FEATURE EXTRACTION FOR TEST DATA
# ============================================================================

def extract_test_features(model, dataloader, device):
    """Extract features from test data"""
    
    features_list = []
    class_names_list = []
    img_names_list = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting test features"):
            images = batch['image'].to(device)
            
            features = model(images).cpu().numpy()
            features_list.append(features)
            
            class_names_list.extend(batch['class_name'])
            img_names_list.extend(batch['img_name'])
    
    features = np.vstack(features_list)
    
    print(f"\nFeature extraction complete: {features.shape}")
    
    return features, class_names_list, img_names_list


# ============================================================================
# 7. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix_with_accuracy(y_true, y_pred, title, top1_acc, top5_acc, save_path, max_classes=50):
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


def plot_top5_accuracy_distribution(y_true, y_proba, class_mapping, title, top1_acc, top5_acc, save_path):
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


# ============================================================================
# 8. EVALUATION FUNCTION
# ============================================================================

def evaluate_subset(X_test, y_test, y_test_names, classifier, subset_name, 
                   class_mapping, output_dir='evaluation_results'):
    """Evaluate a specific subset (overall/paired/unpaired)"""
    
    print(f"Evaluating: {subset_name}")
    print(f" Samples: {len(y_test)}")
    
    if len(y_test) == 0:
        print(f"No samples for {subset_name}, skipping")
        return {'top1': 0, 'top5': 0, 'samples': 0}
    
    # Get prediction probabilities
    y_pred = classifier.predict(X_test)

    # Calculate Top-1 accuracy
    top1_acc = accuracy_score(y_test, y_pred)

    try:
        y_proba = classifier.predict_proba(X_test)
        print(f" Using predict_proba for Top-5 calculation")
    except AttributeError:
        # SVM without probability=True, use decision_function instead
        print(f" Using decision_function for Top-5 calculation (SVM without probability)")
        y_proba = classifier.decision_function(X_test)
    
    # Calculate Top-5 accuracy
    top5_acc = top_k_accuracy_score(y_test, y_proba, k=5, labels=np.arange(y_proba.shape[1]))
    
    # Convert to percentages for summary
    top1_acc_pct = top1_acc * 100
    top5_acc_pct = top5_acc * 100
    
    print(f"\nResults for {subset_name}:")
    print(f" Top-1 Accuracy: {top1_acc:.4f} ({top1_acc_pct:.2f}%)")
    print(f" Top-5 Accuracy: {top5_acc:.4f} ({top5_acc_pct:.2f}%)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix (Top-1)
    cm_path = os.path.join(output_dir, f'{subset_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plot_confusion_matrix_with_accuracy(
        y_test, y_pred, 
        f'{subset_name} - Confusion Matrix (Top {min(50, len(np.unique(y_test)))} Classes)',
        top1_acc, top5_acc,
        cm_path
    )
    
    # Plot Top-5 distribution
    top5_path = os.path.join(output_dir, f'{subset_name.lower().replace(" ", "_")}_top5_distribution.png')
    plot_top5_accuracy_distribution(
        y_test, y_proba, class_mapping,
        f'{subset_name} - Top-5 Prediction Distribution',
        top1_acc, top5_acc,
        top5_path
    )
    
    return {
        'top1': top1_acc_pct, 
        'top5': top5_acc_pct,  
        'samples': len(y_test)
    }


# ============================================================================
# 9. CREATE SUMMARY CSV
# ============================================================================

def create_summary_csv(results_dict, output_dir='evaluation_results'):
    """Create a summary CSV file with all results"""
    summary_file = os.path.join(output_dir, "test_results_summary.csv")
    
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Subset', 'Top-1 Accuracy (%)', 'Top-5 Accuracy (%)', 'Samples'])
        
        for subset_name, results in results_dict.items():
            if results['samples'] > 0:  # Only include if there are samples
                writer.writerow([
                    subset_name,
                    f"{results['top1']:.2f}",
                    f"{results['top5']:.2f}",
                    results['samples']
                ])
    
    print(f"Summary saved to: {summary_file}")
    return summary_file


# ============================================================================
# 10. MAIN EVALUATION
# ============================================================================

def main():
    """Main evaluation function"""
    
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
    
    # Load class lists
    paired_classes, unpaired_classes = load_class_lists()
    
    # Load DINOv2 model
    dinov2_model = load_dinov2_from_kagglehub(device)
    
    # Load saved models
    classifier, normalizer, class_mapping, metadata = load_saved_models('saved_models_no_synthetic')
    
    # Load test dataset
    print("Loading Test Dataset")
    
    test_dataset = TestDataset(
        root_dir='dataset',
        groundtruth_file='dataset/list/groundtruth.txt',
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Extract features
    test_features, test_class_names, test_img_names = extract_test_features(
        dinov2_model, test_loader, device
    )
    
    # Prepare data
    class_to_idx = {str(k): v for k, v in class_mapping['class_to_idx'].items()}
    
    # Convert test class names to indices
    y_test = []
    valid_indices = []
    
    for i, class_name in enumerate(test_class_names):
        class_name_str = str(class_name)
        if class_name_str in class_to_idx:
            y_test.append(class_to_idx[class_name_str])
            valid_indices.append(i)
    
    # Filter features for valid samples
    X_test = test_features[valid_indices]
    y_test = np.array(y_test)
    test_class_names_valid = [test_class_names[i] for i in valid_indices]
    
    # Normalize features (all test images are photo domain)
    domains = np.ones(len(X_test))
    X_test_norm = normalizer.transform(X_test, domains)
    
    print(f"\nTotal valid test samples: {len(y_test)}")
    
    # Separate into paired and unpaired
    paired_mask = np.array([str(cn) in paired_classes for cn in test_class_names_valid])
    unpaired_mask = np.array([str(cn) in unpaired_classes for cn in test_class_names_valid])
    
    print(f" Paired samples: {paired_mask.sum()}")
    print(f" Unpaired samples: {unpaired_mask.sum()}")
    
    # Create output directory
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate all subsets and collect results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    results_dict = {}
    
    # Evaluate overall
    overall_results = evaluate_subset(
        X_test_norm, y_test, test_class_names_valid,
        classifier, "Overall", class_mapping, output_dir
    )
    results_dict["Overall"] = overall_results
    
    # Evaluate paired classes
    if paired_mask.sum() > 0:
        paired_results = evaluate_subset(
            X_test_norm[paired_mask], y_test[paired_mask],
            [test_class_names_valid[i] for i in range(len(test_class_names_valid)) if paired_mask[i]],
            classifier, "Paired Classes", class_mapping, output_dir
        )
        results_dict["Paired Classes"] = paired_results
    
    # Evaluate unpaired classes
    if unpaired_mask.sum() > 0:
        unpaired_results = evaluate_subset(
            X_test_norm[unpaired_mask], y_test[unpaired_mask],
            [test_class_names_valid[i] for i in range(len(test_class_names_valid)) if unpaired_mask[i]],
            classifier, "Unpaired Classes", class_mapping, output_dir
        )
        results_dict["Unpaired Classes"] = unpaired_results
    
    # Create summary CSV
    summary_file = create_summary_csv(results_dict, output_dir)
    
    print("FINAL SUMMARY")
  
    
    for subset_name, results in results_dict.items():
        if results['samples'] > 0:
            print(f"{subset_name:20} | Top-1: {results['top1']:6.2f}% | Top-5: {results['top5']:6.2f}% | Samples: {results['samples']}")
    
    print("\nEvaluation Complete!")
    print(f"Results saved to: {output_dir}/")
    print(f"Summary file: {summary_file}")


if __name__ == "__main__":
    main()