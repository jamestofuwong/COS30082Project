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
# Global Configuration
# ==============================================================
CONFIG = {
    'dataset_root': 'dataset',
    'train_list': 'dataset/list/train.txt',
    'num_classes': None,  # Will be set automatically
    'epochs': 30,
    'batch_size': 32,
    'lr': 1e-3,
    'image_size': 224,
    'val_split': 0.2,
    'seed': 42,
    'save_dir': 'models4'
}

# ==============================================================
# Enhanced Loss Functions
# ==============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
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

class AdaptiveDomainLoss(nn.Module):
    def __init__(self, alpha=0.3, gamma=2, paired_weight=1.5):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=1, gamma=gamma)
        self.domain_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.paired_weight = paired_weight
        
    def forward(self, class_logits, domain_logits, targets, domains, has_pairs):
        cls_loss = self.focal_loss(class_logits, targets)
        domain_loss = self.domain_loss(domain_logits, domains)
        
        paired_mask = torch.tensor(has_pairs, dtype=torch.float32).to(class_logits.device)
        paired_cls_loss = (cls_loss * paired_mask * self.paired_weight).mean()
        unpaired_cls_loss = (cls_loss * (1 - paired_mask)).mean()
        total_cls_loss = paired_cls_loss + unpaired_cls_loss
        
        return total_cls_loss + self.alpha * domain_loss, cls_loss.mean(), domain_loss, paired_cls_loss, unpaired_cls_loss

# ==============================================================
# Data Augmentation
# ==============================================================
def get_enhanced_transform(augment_type='standard'):
    if augment_type == 'enhanced':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, shear=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ==============================================================
# Feature Precomputation
# ==============================================================
def precompute_dinov2_features(dataset, device, save_path="dinov2_features.pth"):
    if os.path.exists(save_path):
        print("Loading precomputed DINOv2 features...")
        features_dict = torch.load(save_path)
        return features_dict['features'], features_dict['labels'], features_dict['domains'], features_dict['has_pairs']
    
    print("Precomputing DINOv2 features...")
    
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
    
    all_features, all_labels, all_domains, all_has_pairs = [], [], [], []
    
    dinov2_transform = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Precomputing features"):
            img_path, mapped_label, domain, has_pair, _ = dataset[i]
            img = Image.open(img_path).convert('RGB')
            img = dinov2_transform(img).unsqueeze(0).to(device)
            
            features = dinov2_model(img)
            if isinstance(features, (tuple, list)):
                features = features[0]
            features = features.flatten(1).cpu()
            
            all_features.append(features)
            all_labels.append(mapped_label)
            all_domains.append(domain)
            all_has_pairs.append(has_pair)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    all_domains = torch.tensor(all_domains, dtype=torch.long)
    all_has_pairs = torch.tensor(all_has_pairs, dtype=torch.bool)
    
    print(f"Precomputed features shape: {all_features.shape}")
    
    features_dict = {
        'features': all_features,
        'labels': all_labels,
        'domains': all_domains,
        'has_pairs': all_has_pairs
    }
    torch.save(features_dict, save_path)
    print("DINOv2 features saved!")
    
    return all_features, all_labels, all_domains, all_has_pairs

# ==============================================================
# Dataset Classes
# ==============================================================
class PrecomputedDataset(Dataset):
    def __init__(self, features, labels, domains, has_pairs):
        self.features = features
        self.labels = labels
        self.domains = domains
        self.has_pairs = has_pairs
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.domains[idx], self.has_pairs[idx]

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

        print(f"Photo Domain Analysis:")
        print(f"Photo classes total: {len(photo_counts)}")
        print(f"Photo classes with ≤5 samples: {len([c for c in photo_counts.values() if c <= 5])}")
        print(f"Photo classes with ≤10 samples: {len([c for c in photo_counts.values() if c <= 10])}")

        self.minority_classes = [cls for cls in self.class_counts.keys() if cls in photo_counts and photo_counts[cls] <= 10]
        self.majority_classes = [cls for cls in self.class_counts.keys() if cls in photo_counts and photo_counts[cls] > 30]

        print(f"Minority classes (≤10 photo samples): {len(self.minority_classes)}")
        print(f"Majority classes (>30 photo samples): {len(self.majority_classes)}")

        if self.minority_classes:
            minority_examples = [(cls, photo_counts[cls]) for cls in list(self.minority_classes)[:5]]
            print(f"Sample minority classes: {minority_examples}")
        
        with open(list_file, 'r') as f:
            for line in f:
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
        
        print(f"Loaded {len(self.samples)} samples with {self.num_classes} classes")
    
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

# ==============================================================
# Model Architecture
# ==============================================================
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
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(dinov2_dim + self.resnet_dim, adapter_dim),
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
        resnet_features = self.resnet(resnet_input)
        resnet_features = resnet_features.squeeze(3).squeeze(2)
        
        combined_features = torch.cat([dinov2_features, resnet_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        class_logits = self.classifier(fused_features)
        rev_features = grad_reverse(fused_features, domain_alpha)
        domain_logits = self.domain_classifier(rev_features)
        
        return class_logits, domain_logits

# ==============================================================
# Training Utilities
# ==============================================================
def plot_training_metrics(metrics_file, save_dir):
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return

    epochs, train_losses, val_losses = [], [], []
    train_accs, val_accs = [], []
    cls_losses, domain_losses = [], []
    paired_accs, unpaired_accs, minority_accs = [], [], []
    learning_rates = []
    unpaired_losses, paired_losses = [], []

    with open(metrics_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row['epoch']))
                train_losses.append(float(row['train_loss']))
                val_losses.append(float(row['val_loss']))
                train_accs.append(float(row['train_acc']))
                val_accs.append(float(row['val_acc']))
                cls_losses.append(float(row.get('cls_loss', 0.0)))
                domain_losses.append(float(row.get('domain_loss', 0.0)))
                paired_accs.append(float(row.get('paired_acc', 0.0)))
                unpaired_accs.append(float(row.get('unpaired_acc', 0.0)))
                minority_accs.append(float(row.get('minority_acc', 0.0)))
                learning_rates.append(float(row.get('learning_rate', 0.0)))
                unpaired_losses.append(float(row.get('unpaired_loss', 0.0)))
                paired_losses.append(float(row.get('paired_loss', 0.0)))
            except (ValueError, KeyError):
                continue

    if not epochs:
        print("No valid rows found in metrics file; skipping plots.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Panel 1: Train/Val Loss
    ax = axes[0][0]
    ax.plot(epochs, train_losses, label='Train Loss')
    ax.plot(epochs, val_losses, label='Val Loss')
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()

    # Panel 2: Train/Val Accuracy
    ax = axes[0][1]
    ax.plot(epochs, train_accs, label='Train Acc')
    ax.plot(epochs, val_accs, label='Val Acc')
    ax.set_title('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()

    # Panel 3: Component Losses
    ax = axes[0][2]
    ax.plot(epochs, cls_losses, label='Classification Loss')
    ax.plot(epochs, domain_losses, label='Domain Loss')
    ax.set_title('Component Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()

    # Panel 4: Specialized Accuracies
    ax = axes[1][0]
    ax.plot(epochs, paired_accs, label='Paired Acc')
    ax.plot(epochs, unpaired_accs, label='Unpaired Acc')
    ax.plot(epochs, minority_accs, label='Minority Acc')
    ax.set_title('Specialized Accuracies')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()

    # Panel 5: Loss Breakdown
    ax = axes[1][1]
    ax.plot(epochs, unpaired_losses, label='Unpaired Loss', color='red')
    ax.plot(epochs, paired_losses, label='Paired Loss', color='blue')
    ax.set_title('Paired vs Unpaired Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()

    axes[1][2].axis('off')

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
    
    photo_counts, herbarium_counts = {}, {}
    
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

# ==============================================================
# Main Training Function
# ==============================================================
def train_enhanced_hybrid_model():
    seed_everything(CONFIG['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    metrics_file = os.path.join(CONFIG['save_dir'], "training_metrics.csv")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 
                        'cls_loss', 'domain_loss', 'paired_acc', 'unpaired_acc', 'minority_acc', 
                        'learning_rate', 'unpaired_loss', 'paired_loss'])
    
    simple_ds = SimpleDataset(CONFIG['dataset_root'], CONFIG['train_list'])
    dinov2_features, labels, domains, has_pairs = precompute_dinov2_features(simple_ds, device)
    
    precomputed_ds = PrecomputedDataset(dinov2_features, labels, domains, has_pairs)
    
    labels_all = labels.numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=CONFIG['val_split'], random_state=CONFIG['seed'])
    train_idx, val_idx = next(sss.split(np.zeros(len(labels_all)), labels_all))
    
    train_ds = Subset(precomputed_ds, train_idx)
    val_ds = Subset(precomputed_ds, val_idx)
    
    train_transform = get_enhanced_transform('standard')
    enhanced_transform = get_enhanced_transform('enhanced')
    
    image_ds = EnhancedImageDataset(simple_ds, train_transform, enhanced_transform)
    train_img_ds = Subset(image_ds, train_idx)
    val_img_ds = Subset(EnhancedImageDataset(simple_ds, 
                                           get_enhanced_transform('standard'), 
                                           get_enhanced_transform('standard')), val_idx)
    
    train_loader = DataLoader(
        list(zip(train_ds, train_img_ds)), 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        list(zip(val_ds, val_img_ds)), 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    
    resnet_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet_backbone = nn.Sequential(*list(resnet_backbone.children())[:-1])
    resnet_backbone = resnet_backbone.to(device)
    
    model = EnhancedHybridModel(resnet_backbone, CONFIG['num_classes']).to(device)
    criterion = AdaptiveDomainLoss(alpha=0.3, gamma=2, paired_weight=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    best_val_acc, early_stopping_counter = 0.0, 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        running_loss, running_correct = 0.0, 0
        running_cls_loss, running_domain_loss = 0.0, 0.0
        running_paired_cls_loss, running_unpaired_cls_loss = 0.0, 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        for batch_idx, ((features, labels, domains, has_pairs), (imgs, img_labels, _, _)) in enumerate(pbar):
            imgs = imgs.to(device)
            features = features.to(device)
            labels = labels.to(device)
            domains = domains.to(device)
            
            class_logits, domain_logits = model(imgs, features, domain_alpha=1.0)
            total_loss, cls_loss, domain_loss, batch_paired_loss, batch_unpaired_loss = criterion(
                class_logits, domain_logits, labels, domains, has_pairs
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_domain_loss += domain_loss.item()
            running_paired_cls_loss += batch_paired_loss.item()
            running_unpaired_cls_loss += batch_unpaired_loss.item()
            
            preds = class_logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            
            if batch_idx % 5 == 0:
                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*running_correct/((batch_idx+1)*CONFIG["batch_size"]):.2f}%',
                    'lr': f'{current_lr:.2e}'
                })
        
        train_loss = running_loss / len(train_loader)
        train_cls_loss = running_cls_loss / len(train_loader)
        train_domain_loss = running_domain_loss / len(train_loader)
        avg_paired_loss = running_paired_cls_loss / len(train_loader)
        avg_unpaired_loss = running_unpaired_cls_loss / len(train_loader)
        train_acc = 100.0 * running_correct / len(train_ds)
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        paired_correct, paired_total = 0, 0
        unpaired_correct, unpaired_total = 0, 0
        minority_correct, minority_total = 0, 0

        minority_mapped_labels = set()
        for orig_label in simple_ds.minority_classes:
            if orig_label in simple_ds.label_map:
                minority_mapped_labels.add(simple_ds.label_map[orig_label])

        paired_mapped_labels = set()
        for orig_label in simple_ds.classes_with_pairs:
            if orig_label in simple_ds.label_map:
                paired_mapped_labels.add(simple_ds.label_map[orig_label])

        with torch.no_grad():
            for (features, labels, domains, has_pairs), (imgs, img_labels, _, _) in val_loader:
                imgs = imgs.to(device)
                features = features.to(device)
                labels = labels.to(device)
                
                class_logits, _ = model(imgs, features, domain_alpha=0.0)
                loss = F.cross_entropy(class_logits, labels)
                val_loss += loss.item() * labels.size(0)
                
                preds = class_logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                for i in range(len(labels)):
                    label_val = labels[i].item()
                    
                    if label_val in minority_mapped_labels:
                        minority_total += 1
                        minority_correct += (preds[i] == labels[i]).item()
                    
                    if label_val in paired_mapped_labels:
                        paired_total += 1
                        paired_correct += (preds[i] == labels[i]).item()
                    else:
                        unpaired_total += 1
                        unpaired_correct += (preds[i] == labels[i]).item()

        val_loss = val_loss / val_total if val_total > 0 else 0
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
        paired_acc = 100.0 * paired_correct / paired_total if paired_total > 0 else 0
        unpaired_acc = 100.0 * unpaired_correct / unpaired_total if unpaired_total > 0 else 0
        minority_acc = 100.0 * minority_correct / minority_total if minority_total > 0 else 0
        
        scheduler.step()
        
        if val_acc > best_val_acc + 0.1:
            best_val_acc = val_acc
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'paired_acc': paired_acc,
                'unpaired_acc': unpaired_acc,
                'minority_acc': minority_acc
            }, os.path.join(CONFIG['save_dir'], "best_model.pth"))
            print(f"New best model! Val Acc: {val_acc:.2f}%")
        else:
            early_stopping_counter += 1
        
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc, 
                           train_cls_loss, train_domain_loss, paired_acc, unpaired_acc, minority_acc, 
                           current_lr, avg_unpaired_loss, avg_paired_loss])
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        print(f"  Paired Classes: {paired_acc:.2f}% | Unpaired Classes: {unpaired_acc:.2f}%")
        print(f"  Minority Classes: {minority_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        if early_stopping_counter >= 10:
            print(f"\nEarly stopping triggered!")
            break
    
    print(f"\nGenerating training plots...")
    plot_training_metrics(metrics_file, CONFIG['save_dir'])
    
    print(f"\nFINAL MODEL ANALYSIS:")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Paired Classes Performance: {paired_acc:.2f}%")
    print(f"   Unpaired Classes Performance: {unpaired_acc:.2f}%") 
    print(f"   Minority Classes Performance: {minority_acc:.2f}%")

    return model, best_val_acc

# ==============================================================
# Main Execution
# ==============================================================
if __name__ == "__main__":
    photo_counts, herbarium_counts = analyze_folder_distribution('dataset')
    simple_ds = SimpleDataset('dataset', 'dataset/list/train.txt')
    CONFIG['num_classes'] = simple_ds.num_classes

    print("Starting Training...")
    model, best_acc = train_enhanced_hybrid_model()

