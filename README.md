# Brain Tumor Classification — Custom CNN 🧠

A **Google Colab-ready** pipeline for brain tumour classification using a custom AlexNet-inspired CNN with PyTorch.

> **Dataset**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)

---

## 📌 Overview

This project trains a custom Convolutional Neural Network (CNN) to classify brain MRI scans into **4 categories**:

| Class | Description |
|-------|-------------|
| `glioma` | Glioma tumour |
| `meningioma` | Meningioma tumour |
| `notumor` | No tumour present |
| `pituitary` | Pituitary tumour |

---

## 🏗️ Model Architecture

**AlexNet-inspired Custom CNN** with:
- 4 Convolutional blocks with **BatchNorm** + **Mish activation**
- MaxPool after blocks 1, 2, and 4
- 3 Fully-connected layers with **Dropout (0.4)**
- Dynamically computed flat size (works for any input resolution)

```
Input (3×128×128)
    ↓
Conv Block 1: Conv2d(3→64)  + BN + Mish + MaxPool  → 64×63×63
    ↓
Conv Block 2: Conv2d(64→128) + BN + Mish + MaxPool  → 128×30×30
    ↓
Conv Block 3: Conv2d(128→256) + BN + Mish           → 256×28×28
    ↓
Conv Block 4: Conv2d(256→256) + BN + Mish + MaxPool → 256×13×13
    ↓
Classifier: Flatten → Linear(43264→1024) → Dropout → Linear(1024→512) → Dropout → Linear(512→4)
    ↓
Output (4 classes)
```

---

## 🚀 Quick Start (Google Colab)

### Step 1 — Get the Dataset
1. Download `archive.zip` from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
2. Upload `archive.zip` to your **Google Drive root** (`MyDrive/archive.zip`)

### Step 2 — Open in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dbAggMVaYNdEEkX2fiFIPXHuYW5WFzh6?usp=sharing)

### Step 3 — Set GPU Runtime
Go to **Runtime → Change runtime type → GPU** (T4 is free!)

### Step 4 — Run All Cells
Execute all cells top-to-bottom. Training takes ~15–20 minutes on a T4 GPU.

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 128 × 128 |
| Batch size | 64 |
| Epochs | 40 |
| Learning rate | 1e-4 |
| Dropout | 0.4 |
| Train/Val split | 90% / 10% |
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |

---

## 📊 Training Pipeline

| Cell | Description |
|------|-------------|
| Cell 1 | Mount Google Drive |
| Cell 2 | Extract dataset archive |
| Cell 3 | Imports & configuration |
| Cell 4 | Custom CNN model definition |
| Cell 5 | Data loading & augmentation |
| Cell 6 | Visualise sample images per class |
| Cell 7 | Class distribution bar chart |
| Cell 8 | Model, loss, optimizer setup |
| Cell 9 | Training loop |
| Cell 10 | Training curves (loss & accuracy) |
| Cell 11 | Test set evaluation |
| Cell 12 | Confusion matrix (counts & normalised) |
| Cell 13 | Per-class accuracy bar chart |
| Cell 14 | Sample predictions grid |
| Cell 15 | Save best model to Google Drive |

---

## 📁 Output Files

After running all cells, the following files are generated:

```
sample_images.png         ← Sample MRI images per class
class_distribution.png    ← Bar chart of class counts
training_curves.png       ← Loss & accuracy over epochs
confusion_matrix.png      ← Count & normalised confusion matrices
per_class_accuracy.png    ← Per-class accuracy bar chart
sample_predictions.png    ← 4×4 grid of test predictions
custom_cnn_brain_tumor.pth ← Saved model weights (in Google Drive)
```

---

## 🧪 Data Augmentation

**Training set** augmentations:
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness & contrast ±0.2)
- ImageNet normalization

**Validation / Test set** — only resize + normalize (no augmentation).

---

## 📦 Requirements

```
torch
torchvision
numpy
matplotlib
seaborn
tqdm
scikit-learn
Pillow
```

All pre-installed in Google Colab. ✅

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
