# ============================================================
#  Brain Tumor Classification — Custom CNN (Colab-Ready)
#  Dataset: Brain Tumor MRI Dataset (Kaggle)
#  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
# ============================================================
#
# HOW TO USE:
#  1. Upload archive.zip to your Google Drive (root / MyDrive).
#  2. Open this in Google Colab.
#  3. Go to Runtime → Change runtime type → GPU (T4 is free).
#  4. Run ALL cells top to bottom.  That's it!
# ============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1 — Mount Google Drive
# ─────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ─────────────────────────────────────────────────────────────
# CELL 2 — Extract archive.zip from Google Drive
# ─────────────────────────────────────────────────────────────
import os
import zipfile

ZIP_PATH    = '/content/drive/MyDrive/archive.zip'   # ← path to your zip in Drive
EXTRACT_DIR = '/content/brain_tumor_dataset'          # where to extract in Colab RAM

if not os.path.exists(EXTRACT_DIR):
    print(f"Extracting {ZIP_PATH} → {EXTRACT_DIR} ...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    print("Extraction complete!")
else:
    print(f"Already extracted at {EXTRACT_DIR}, skipping.")

# ── Auto-detect the correct Training / Testing folder paths ──
# Kaggle archives sometimes nest inside an extra subfolder.
def find_subfolder(root, name):
    """Recursively search for a folder called `name` under `root`."""
    for dirpath, dirnames, _ in os.walk(root):
        if name in dirnames:
            return os.path.join(dirpath, name)
    return None

TRAIN_PATH = find_subfolder(EXTRACT_DIR, 'Training')
TEST_PATH  = find_subfolder(EXTRACT_DIR, 'Testing')

assert TRAIN_PATH is not None, "Could not find 'Training' folder in archive!"
assert TEST_PATH  is not None, "Could not find 'Testing'  folder in archive!"

print(f"Training folder : {TRAIN_PATH}")
print(f"Testing  folder : {TEST_PATH}")
print(f"Classes found   : {sorted(os.listdir(TRAIN_PATH))}")

# ─────────────────────────────────────────────────────────────
# CELL 3 — Imports & Configuration
# ─────────────────────────────────────────────────────────────
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)
th.backends.cudnn.deterministic = True

# ── Hyperparameters ──────────────────────────────────────────
IMG_SIZE       = 128
BATCH_TRAIN    = 64
BATCH_VALID    = 64
BATCH_TEST     = 64
SPLIT          = 0.9      # 90 % train / 10 % validation (from Training folder)
EPOCHS         = 40
LR             = 1e-4
DROPOUT        = 0.4
OUTPUT_CLASSES = 4
CLASS_NAMES    = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ── Device ───────────────────────────────────────────────────
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {th.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────
# CELL 4 — Custom CNN Model Definition
#          (AlexNet-inspired with BatchNorm + Mish activation)
# ─────────────────────────────────────────────────────────────
class CustomCNN(nn.Module):
    """Custom CNN for brain-tumour classification.

    Architecture (AlexNet-inspired):
      • 4 convolutional blocks with BatchNorm + Mish activation
      • MaxPool after blocks 1, 2, and 4
      • 3 fully-connected layers with Dropout

    Parameters
    ----------
    output_classes : int   — number of output classes (default 4)
    dropout        : float — dropout rate in dense layers (default 0.4)
    activation     : nn.Module — activation for conv layers (default Mish)
    """

    def __init__(
        self,
        output_classes: int = 4,
        dropout: float = 0.4,
        activation: nn.Module = None,
        img_size: int = 128,
    ):
        super(CustomCNN, self).__init__()
        if activation is None:
            activation = nn.Mish()

        # ── Convolutional backbone ────────────────────────────
        # Spatial dims for 128×128 input:
        # Block 1: 128 → Conv(3) → 126 → MaxPool(2) → 63
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            type(activation)(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 2: 63 → Conv(3) → 61 → MaxPool(2) → 30
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            type(activation)(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 3: 30 → Conv(3) → 28
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            type(activation)(),
        )
        # Block 4: 28 → Conv(3) → 26 → MaxPool(2) → 13
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            type(activation)(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Dynamically compute flat size so this never breaks ─
        flat_size = self._get_flat_size(img_size)
        print(f"  [CustomCNN] img_size={img_size}  →  flat_size={flat_size}")

        # ── Classifier head ───────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 1024),
            type(activation)(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 512),
            type(activation)(),
            nn.Dropout(p=dropout),
            nn.Linear(512, output_classes),
        )

    def _get_flat_size(self, img_size: int) -> int:
        """Run a dummy forward pass through conv blocks to get the flat size."""
        with th.no_grad():
            dummy = th.zeros(1, 3, img_size, img_size)
            dummy = self.conv1(dummy)
            dummy = self.conv2(dummy)
            dummy = self.conv3(dummy)
            dummy = self.conv4(dummy)
            return int(dummy.numel())   # batch=1, so numel = C*H*W

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x


# ── Sanity check ─────────────────────────────────────────────
_test_model = CustomCNN(output_classes=OUTPUT_CLASSES, dropout=DROPOUT, img_size=IMG_SIZE)
_dummy      = th.zeros(1, 3, IMG_SIZE, IMG_SIZE)
_out        = _test_model(_dummy)
assert _out.shape == (1, OUTPUT_CLASSES), f"Unexpected output shape: {_out.shape}"
total_params  = sum(p.numel() for p in _test_model.parameters())
trainable_par = sum(p.numel() for p in _test_model.parameters() if p.requires_grad)
print(f"Model output shape  : {_out.shape}  ✓")
print(f"Total parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_par:,}")
del _dummy, _test_model, _out

# ─────────────────────────────────────────────────────────────
# CELL 5 — Data Loading & Augmentation
# ─────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Full training set → split into train + val ───────────────
full_train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
test_dataset       = datasets.ImageFolder(TEST_PATH,  transform=eval_transform)

n_train = int(len(full_train_dataset) * SPLIT)
n_val   = len(full_train_dataset) - n_train
train_dataset, val_dataset_raw = random_split(
    full_train_dataset, [n_train, n_val],
    generator=th.Generator().manual_seed(SEED)
)

# Apply eval_transform (no augmentation) to validation split
class SubsetWithTransform(th.utils.data.Dataset):
    """Wraps a Subset and applies a different transform."""
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.subset.dataset.imgs[self.subset.indices[idx]]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label

val_dataset = SubsetWithTransform(val_dataset_raw, eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_TRAIN, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_VALID, shuffle=False,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_TEST,  shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Training samples  : {n_train}")
print(f"Validation samples: {n_val}")
print(f"Test samples      : {len(test_dataset)}")
print(f"Classes           : {full_train_dataset.classes}")

# ─────────────────────────────────────────────────────────────
# CELL 6 — Visualise Sample Images per Class
# ─────────────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def denorm(tensor):
    """Reverse ImageNet normalisation for display."""
    img = tensor.permute(1, 2, 0).numpy()
    img = img * STD + MEAN
    return np.clip(img, 0, 1)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle('Sample Brain MRI Images per Class', fontsize=16, fontweight='bold')

from PIL import Image as PILImage
for col, cls in enumerate(CLASS_NAMES):
    cls_path = os.path.join(TRAIN_PATH, cls)
    fnames   = [f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for row in range(2):
        fname = random.choice(fnames)
        img   = PILImage.open(os.path.join(cls_path, fname)).convert('RGB')
        img_t = eval_transform(img)
        axes[row][col].imshow(denorm(img_t))
        axes[row][col].set_title(cls.capitalize() if row == 0 else '', fontsize=11)
        axes[row][col].axis('off')

plt.tight_layout()
plt.savefig('sample_images.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved → sample_images.png")

# ─────────────────────────────────────────────────────────────
# CELL 7 — Class Distribution Bar Chart
# ─────────────────────────────────────────────────────────────
def count_classes(root):
    counts = {}
    for cls in sorted(os.listdir(root)):
        p = os.path.join(root, cls)
        if os.path.isdir(p):
            counts[cls] = len(os.listdir(p))
    return counts

train_counts = count_classes(TRAIN_PATH)
test_counts  = count_classes(TEST_PATH)

colours = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, counts, title in zip(axes,
                               [train_counts, test_counts],
                               ['Training Set', 'Testing Set']):
    bars = ax.bar(counts.keys(), counts.values(), color=colours)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Class')
    ax.set_ylabel('Images')
    for bar, v in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10, str(v), ha='center', fontsize=10)

plt.suptitle('Class Distribution', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved → class_distribution.png")

# ─────────────────────────────────────────────────────────────
# CELL 8 — Model, Loss, Optimizer
# ─────────────────────────────────────────────────────────────
model     = CustomCNN(output_classes=OUTPUT_CLASSES, dropout=DROPOUT, img_size=IMG_SIZE).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
print("Model ready.")

# ─────────────────────────────────────────────────────────────
# CELL 9 — Training Loop
# ─────────────────────────────────────────────────────────────
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss':   [], 'val_acc':   [],
}

best_val_acc  = 0.0
best_model_wt = None

for epoch in range(1, EPOCHS + 1):

    # ── Train ────────────────────────────────────────────────
    model.train()
    running_loss = correct = total = 0

    for images, labels in tqdm(train_loader,
                                desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]",
                                leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds      = th.max(outputs, 1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    train_loss = running_loss / total
    train_acc  = correct / total

    # ── Validate ─────────────────────────────────────────────
    model.eval()
    val_loss_sum = val_correct = val_total = 0

    with th.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs       = model(images)
            loss          = criterion(outputs, labels)
            val_loss_sum += loss.item() * images.size(0)
            _, preds      = th.max(outputs, 1)
            val_correct  += (preds == labels).sum().item()
            val_total    += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_acc  = val_correct  / val_total

    scheduler.step(val_loss)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc  = val_acc
        best_model_wt = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch:02d}/{EPOCHS}  "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.2f}%  |  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc*100:.2f}%")

print(f"\nBest Validation Accuracy: {best_val_acc*100:.2f}%")

# ─────────────────────────────────────────────────────────────
# CELL 10 — Training Curves
# ─────────────────────────────────────────────────────────────
ep = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(ep, history['train_loss'], 'b-o', markersize=3, label='Train Loss')
axes[0].plot(ep, history['val_loss'],   'r-o', markersize=3, label='Val Loss')
axes[0].set_title('Loss Curves', fontsize=14)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(ep, [a*100 for a in history['train_acc']], 'b-o', markersize=3, label='Train Acc')
axes[1].plot(ep, [a*100 for a in history['val_acc']],   'r-o', markersize=3, label='Val Acc')
axes[1].set_title('Accuracy Curves', fontsize=14)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

fig.suptitle('Custom CNN — Training History', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('training_curves.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved → training_curves.png")

# ─────────────────────────────────────────────────────────────
# CELL 11 — Test Set Evaluation
# ─────────────────────────────────────────────────────────────
model.load_state_dict(best_model_wt)
model.eval()

all_preds, all_labels = [], []

with th.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
        images  = images.to(device)
        outputs = model(images)
        _, preds = th.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = accuracy_score(all_labels, all_preds)
test_f1  = f1_score(all_labels, all_preds, average='macro')

print("=" * 55)
print("              TEST SET RESULTS")
print("=" * 55)
print(f"  Accuracy (Overall)  : {test_acc*100:.2f}%")
print(f"  F1-Score (Macro)    : {test_f1*100:.2f}%")
print("=" * 55)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds,
                             target_names=CLASS_NAMES, digits=4))

# ─────────────────────────────────────────────────────────────
# CELL 12 — Confusion Matrix
# ─────────────────────────────────────────────────────────────
cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_title('Confusion Matrix (Counts)')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
axes[1].set_title('Confusion Matrix (Normalised)')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')

fig.suptitle(f'Custom CNN — Test Accuracy: {test_acc*100:.2f}%',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved → confusion_matrix.png")

# ─────────────────────────────────────────────────────────────
# CELL 13 — Per-class Accuracy Bar Chart
# ─────────────────────────────────────────────────────────────
per_class_acc = cm_norm.diagonal()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(CLASS_NAMES, per_class_acc * 100, color=colours)
ax.set_ylim(0, 115)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Per-Class Accuracy — Custom CNN', fontsize=14, fontweight='bold')
for bar, acc in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved → per_class_accuracy.png")

# ─────────────────────────────────────────────────────────────
# CELL 14 — Sample Predictions Grid
# ─────────────────────────────────────────────────────────────
model.eval()
images_shown, labels_shown, preds_shown = [], [], []

with th.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        out  = model(imgs)
        _, pr = th.max(out, 1)
        images_shown.append(imgs.cpu())
        labels_shown.append(lbls)
        preds_shown.append(pr.cpu())
        if sum(len(b) for b in images_shown) >= 16:
            break

images_shown = th.cat(images_shown)[:16]
labels_shown = th.cat(labels_shown)[:16]
preds_shown  = th.cat(preds_shown)[:16]

fig, axes = plt.subplots(4, 4, figsize=(14, 14))
for i, ax in enumerate(axes.flat):
    img      = denorm(images_shown[i])
    true_cls = CLASS_NAMES[labels_shown[i]]
    pred_cls = CLASS_NAMES[preds_shown[i]]
    colour   = 'green' if true_cls == pred_cls else 'red'
    ax.imshow(img)
    ax.set_title(f'True : {true_cls}\nPred : {pred_cls}',
                 color=colour, fontsize=9)
    ax.axis('off')

fig.suptitle('Sample Predictions  (green = correct, red = wrong)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved → sample_predictions.png")

# ─────────────────────────────────────────────────────────────
# CELL 15 — Save Best Model back to Google Drive
# ─────────────────────────────────────────────────────────────
save_path = '/content/drive/MyDrive/custom_cnn_brain_tumor.pth'
th.save({
    'model_state_dict': best_model_wt,
    'val_acc'         : best_val_acc,
    'test_acc'        : test_acc,
    'class_names'     : CLASS_NAMES,
}, save_path)
print(f"Model saved to Google Drive → {save_path}")

print("\n✅ All done!  Summary:")
print(f"   Best Val  Accuracy : {best_val_acc*100:.2f}%")
print(f"   Test      Accuracy : {test_acc*100:.2f}%")
print(f"   Test F1  (macro)   : {test_f1*100:.2f}%")
