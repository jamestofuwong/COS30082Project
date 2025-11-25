import os
import random
import csv
from tqdm import tqdm
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# Dataset and Augmentation
# ==============================================================
class SimpleDataset(Dataset):
    def __init__(self, root_dir, list_file):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.classes_with_pairs = set()
        self.classes_without_pairs = set()
        with open('dataset/list/class_with_pairs.txt', 'r') as f:
            self.classes_with_pairs = set(int(line.strip()) for line in f)
        with open('dataset/list/class_without_pairs.txt', 'r') as f:
            self.classes_without_pairs = set(int(line.strip()) for line in f)
        original_labels = []
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    label = int(parts[1])
                    original_labels.append(label)
        unique_labels = sorted(set(original_labels))
        self.label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        self.class_counts = Counter(original_labels)
        photo_counts = {}
        photo_path = Path(root_dir) / 'train' / 'photo'
        if photo_path.exists():
            for class_folder in photo_path.iterdir():
                if class_folder.is_dir():
                    class_id = int(class_folder.name)
                    num_images = len(list(class_folder.glob('*.jpg'))) + len(list(class_folder.glob('*.png')))
                    photo_counts[class_id] = num_images
        self.minority_classes = [cls for cls in self.class_counts.keys() if cls in photo_counts and photo_counts[cls] <= 10]
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
                    is_minority = original_label in self.minority_classes
                    self.samples.append((str(full_path), mapped_label, domain, has_pair, is_minority))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

class EnhancedImageDataset(Dataset):
    def __init__(self, dataset, standard_transform, enhanced_transform):
        self.dataset = dataset
        self.standard_transform = standard_transform
        self.enhanced_transform = enhanced_transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img_path, label, domain, has_pair, is_minority = self.dataset[idx]
        img = Image.open(img_path).convert('RGB')
        if is_minority and random.random() < 0.7:
            img = self.enhanced_transform(img)
        else:
            img = self.standard_transform(img)
        return img, label, domain, has_pair

def get_enhanced_transform(augment_type='standard'):
    if augment_type == 'enhanced':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Less aggressive crop
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),  # Reduced jitter
            transforms.RandomRotation(10),  # Reduced rotation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Less aggressive
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),  # Reduced
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ==============================================================
# EfficientNet Baseline Model
# ==============================================================
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.7):  # Much higher dropout
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
        features = self.dropout2(features)  # Double dropout in training
        logits = self.classifier(features)
        return logits

# ==============================================================
# Focal Loss and Pair/Unpair Loss
# ==============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class PairedFocalLoss(nn.Module):
    def __init__(self, gamma=2, paired_weight=1.7, unpaired_weight=0.6, label_smoothing=0.1):  # Stronger paired/unpaired weighting
        super().__init__()
        self.focal_loss = FocalLoss(alpha=1, gamma=gamma, reduction='none')
        self.paired_weight = paired_weight
        self.unpaired_weight = unpaired_weight
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, targets, has_pairs):
        # Apply label smoothing
        if self.label_smoothing > 0:
            num_classes = logits.size(-1)
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = F.log_softmax(logits, dim=1)
            smoothed_loss = -(true_dist * log_probs).sum(dim=1)
            focal_losses = smoothed_loss
        else:
            focal_losses = self.focal_loss(logits, targets)
        
        paired_mask = torch.tensor(has_pairs, dtype=torch.float32, device=logits.device)
        paired_loss = (focal_losses * paired_mask * self.paired_weight).sum()
        unpaired_loss = (focal_losses * (1 - paired_mask) * self.unpaired_weight).sum()
        total_samples = paired_mask.sum() * self.paired_weight + (1 - paired_mask).sum() * self.unpaired_weight
        return (paired_loss + unpaired_loss) / (total_samples + 1e-8)

# ==============================================================
# Training Function
# ==============================================================
def train_efficientnet_baseline(
    dataset_root,
    train_list,
    num_classes,
    image_size=224,
    epochs=30,
    batch_size=32,
    lr=1e-4,
    device=None,
    val_split=0.2,
    seed=42,
    save_dir="efficientnet_baseline"
):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(save_dir, exist_ok=True)
    metrics_file = os.path.join(save_dir, "training_metrics.csv")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 
                        'paired_acc', 'unpaired_acc', 'learning_rate'])  # Removed minority_acc
    simple_ds = SimpleDataset(dataset_root, train_list)
    labels = [s[1] for s in simple_ds.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    train_transform = get_enhanced_transform('standard')
    enhanced_transform = get_enhanced_transform('enhanced')
    image_ds = EnhancedImageDataset(simple_ds, train_transform, enhanced_transform)
    train_ds = Subset(image_ds, train_idx)
    val_ds = Subset(image_ds, val_idx)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    model = EfficientNetClassifier(num_classes, dropout=0.7).to(device)  # Much higher dropout
    criterion = PairedFocalLoss(gamma=2, paired_weight=1.7, unpaired_weight=0.6, label_smoothing=0.1)  # Stronger paired/unpaired weighting
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)  # Much higher weight decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_acc = 0.0
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_epoch = 0

    paired_mapped_labels = set()
    for orig_label in simple_ds.classes_with_pairs:
        if orig_label in simple_ds.label_map:
            paired_mapped_labels.add(simple_ds.label_map[orig_label])
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct = 0.0, 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, (imgs, labels, domains, has_pairs) in enumerate(pbar):
            imgs = imgs.to(device)
            labels = labels.to(device)
            has_pairs = has_pairs.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels, has_pairs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            if batch_idx % 5 == 0:
                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*running_correct/((batch_idx+1)*batch_size):.2f}',
                    'lr': f'{current_lr:.2e}'
                })
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * running_correct / len(train_ds)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        paired_correct, paired_total = 0, 0
        unpaired_correct, unpaired_total = 0, 0
        with torch.no_grad():
            for imgs, labels, domains, has_pairs in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                has_pairs = has_pairs.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels, has_pairs)
                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                for i in range(len(labels)):
                    is_paired = has_pairs[i].item()
                    
                    if is_paired:  # Paired = both domains, reliable labels
                        paired_total += 1
                        paired_correct += (preds[i] == labels[i]).item()
                    else:  # Unpaired = one domain, less reliable
                        unpaired_total += 1
                        unpaired_correct += (preds[i] == labels[i]).item()
        
        val_loss = val_loss / val_total if val_total > 0 else 0
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
        paired_acc = 100.0 * paired_correct / paired_total if paired_total > 0 else 0
        unpaired_acc = 100.0 * unpaired_correct / unpaired_total if unpaired_total > 0 else 0
        
        scheduler.step()

        # Early stopping based on validation accuracy improvement (patience=4)
        if val_acc > best_val_acc + 0.1:
            best_val_acc = val_acc
            best_epoch = epoch
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'paired_acc': paired_acc,
                'unpaired_acc': unpaired_acc
            }, os.path.join(save_dir, "best_efficientnet_model.pth"))
            print(f"New best model! Val Acc: {val_acc:.2f}%")
        else:
            early_stopping_counter += 1

        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc,
                           paired_acc, unpaired_acc, current_lr])

        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        print(f"  \033[92mPaired Classes: {paired_acc:.2f}%\033[0m | \033[91mUnpaired Classes: {unpaired_acc:.2f}%\033[0m")
        print(f"  Learning Rate: {current_lr:.2e}")

        # Stop if no improvement in val_acc for 4 epochs
        if early_stopping_counter >= 4:
            print(f"\nEarly stopping triggered! No improvement in validation accuracy for 4 epochs.")
            break

    print(f"\n Generating training plots...")
    plot_training_metrics(metrics_file, save_dir)
    print(f"\n FINAL MODEL ANALYSIS:")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}% (at epoch {best_epoch})")
    print(f"   \033[92mPaired Classes Performance: {paired_acc:.2f}%\033[0m")
    print(f"   \033[91mUnpaired Classes Performance: {unpaired_acc:.2f}%\033[0m")
    return model, best_val_acc

def plot_training_metrics(metrics_file, save_dir):
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return
    epochs = []
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    paired_accs, unpaired_accs = [], []  # Removed minority_accs
    learning_rates = []
    with open(metrics_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row['epoch']))
                train_losses.append(float(row['train_loss']))
                val_losses.append(float(row['val_loss']))
                train_accs.append(float(row['train_acc']))
                val_accs.append(float(row['val_acc']))
                paired_accs.append(float(row.get('paired_acc', 0.0)))
                unpaired_accs.append(float(row.get('unpaired_acc', 0.0)))
                learning_rates.append(float(row.get('learning_rate', 0.0)))
            except (ValueError, KeyError):
                continue
    if not epochs:
        print("No valid rows found in metrics file; skipping plots.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # Changed to 2x2 grid
    
    ax = axes[0][0]
    ax.plot(epochs, train_losses, label='Train Loss')
    ax.plot(epochs, val_losses, label='Val Loss')
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    
    ax = axes[0][1]
    ax.plot(epochs, train_accs, label='Train Acc')
    ax.plot(epochs, val_accs, label='Val Acc')
    ax.set_title('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    
    ax = axes[1][0]
    ax.plot(epochs, paired_accs, label='Paired Acc', color='green', linewidth=2)  # Green
    ax.plot(epochs, unpaired_accs, label='Unpaired Acc', color='red', linewidth=2)  # Red
    ax.set_title('Specialized Accuracies')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    
    ax = axes[1][1]
    ax.plot(epochs, learning_rates, label='Learning Rate')
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    out_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved training curves to: {out_path}")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def analyze_folder_distribution(root_dir):
    print(f"\nAnalyzing Folder Distribution in: {root_dir}")
    photo_path = Path(root_dir) / 'train' / 'photo'
    herbarium_path = Path(root_dir) / 'train' / 'herbarium'
    photo_counts = {}
    herbarium_counts = {}
    if photo_path.exists():
        for class_folder in photo_path.iterdir():
            if class_folder.is_dir():
                class_id = int(class_folder.name)
                num_images = len(list(class_folder.glob('*.jpg'))) + len(list(class_folder.glob('*.png')))
                photo_counts[class_id] = num_images
    if herbarium_path.exists():
        for class_folder in herbarium_path.iterdir():
            if class_folder.is_dir():
                class_id = int(class_folder.name)
                num_images = len(list(class_folder.glob('*.jpg'))) + len(list(class_folder.glob('*.png')))
                herbarium_counts[class_id] = num_images
    print(f"Photo classes: {len(photo_counts)}")
    print(f"Herbarium classes: {len(herbarium_counts)}")
    few_photo_classes = {cls: count for cls, count in photo_counts.items() if count <= 10}
    few_herbarium_classes = {cls: count for cls, count in herbarium_counts.items() if count <= 10}
    print(f"Photo classes with ≤10 samples: {len(few_photo_classes)}")
    print(f"Herbarium classes with ≤10 samples: {len(few_herbarium_classes)}")
    return photo_counts, herbarium_counts

if __name__ == "__main__":
    photo_counts, herbarium_counts = analyze_folder_distribution('dataset')
    simple_ds = SimpleDataset('dataset', 'dataset/list/train.txt')
    actual_num_classes = simple_ds.num_classes
    config = {
        'dataset_root': 'dataset',
        'train_list': 'dataset/list/train.txt',
        'num_classes': actual_num_classes,
        'epochs': 40,  # Changed to 40
        'batch_size': 32,
        'lr': 1e-4,  # Slightly increased
        'save_dir': 'efficientnet_baseline'
    }
    print("Starting EfficientNet Baseline Training...")
    model, best_acc = train_efficientnet_baseline(**config)