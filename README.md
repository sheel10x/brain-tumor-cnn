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

## 📊 Results

> Trained on **Tesla T4 GPU** for **40 epochs** | Dataset: 5,040 train / 560 val / 1,600 test images

### 🏆 Final Metrics

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **93.75%** |
| **Test F1-Score (Macro)** | **93.58%** |
| **Best Val Accuracy** | **98.21%** (Epoch 36) |
| **Final Train Accuracy** | **99.48%** |

---

### 📋 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Glioma | 0.9815 | 0.7975 | 0.8800 | 400 |
| Meningioma | 0.8986 | 0.9525 | 0.9248 | 400 |
| No Tumor | 0.9195 | 1.0000 | 0.9581 | 400 |
| Pituitary | 0.9615 | 1.0000 | 0.9804 | 400 |
| **Macro Avg** | **0.9403** | **0.9375** | **0.9358** | **1600** |

---

### 📈 Training History (selected epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 01 | 0.7927 | 67.84% | 0.5838 | 74.46% |
| 05 | 0.3329 | 86.88% | 0.3030 | 87.68% |
| 10 | 0.1836 | 93.13% | 0.2325 | 92.14% |
| 15 | 0.1225 | 95.58% | 0.2119 | 91.96% |
| 20 | 0.0740 | 97.26% | 0.1409 | 94.64% |
| 25 | 0.0609 | 97.88% | 0.1149 | 95.18% |
| 30 | 0.0281 | 99.07% | 0.0593 | 98.04% |
| 35 | 0.0190 | 99.40% | 0.0713 | 97.68% |
| **36** | **0.0160** | **99.48%** | **0.0590** | **98.21% ⭐** |
| 40 | 0.0151 | 99.48% | 0.0750 | 97.50% |

> ⭐ Best model checkpoint saved at **Epoch 36** (Val Acc: 98.21%)

---

### 🖼️ Outputs

All plots and full training logs are available in the Jupyter notebook:
👉 **[`brain_tumor_custom_cnn.ipynb`](brain_tumor_custom_cnn.ipynb)**

| Plot | Description |
|------|-------------|
| Training Curves | Loss & accuracy over 40 epochs |
| Confusion Matrix | Count & normalised heatmaps |
| Per-Class Accuracy | Bar chart per class |
| Sample Predictions | 4×4 grid (green = correct, red = wrong) |
| Class Distribution | Train vs test set class balance |

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
