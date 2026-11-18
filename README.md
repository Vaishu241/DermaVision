# DermaVision: Skin Cancer Detection Using EfficientNetV2 with Hybrid Attention and Ensemble Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)

**B.Tech Capstone Project**  
Base Research Paper: *"Skin Cancer Detection and Classification Using Deep Learning Techniques"* (2026 IEEE AIDE).

---

## 🔬 Project Overview

**DermaVision** is an end-to-end computer vision and explainable AI framework designed to classify dermatoscopic skin lesions into seven distinct diagnostic categories using the **HAM10000** dataset.

The system incorporates a multi-model ensemble combining three complementary architectures:
1. **EfficientNetV2-B3 with Hybrid Attention (CBAM)**: Incorporates Channel and Spatial Attention to highlight lesion boundaries and pigment networks.
2. **ConvNeXt**: Modern convolutional backbone capturing localized inductive biases and depthwise representations.
3. **Vision Transformer (ViT)**: Self-attention transformer capturing long-range contextual relationships and global morphology.
4. **Weighted Ensemble**: Merges predictive probability distributions using weighted probability averaging:
   $$P_{\text{ensemble}}(y \mid x) = w_{\text{eff}} P_{\text{eff}} + w_{\text{conv}} P_{\text{conv}} + w_{\text{vit}} P_{\text{vit}}$$
5. **Explainable AI (Grad-CAM)**: Visualizes localized heatmaps showing anatomical regions contributing to diagnosis.
6. **Streamlit Web Application**: An interactive clinical interface for dermatoscopic image upload and diagnostic prediction.

---

## 📁 Project Directory Structure

```text
DermaVision/
├── app.py                      # Streamlit interactive web application
├── config.py                   # Central configuration (paths, hyperparameters, class labels)
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation and setup guide
│
├── data/                       # Dataset storage
│   ├── raw/                    # Raw HAM10000 images & metadata CSV
│   ├── processed/              # Stratified split metadata (train/val/test)
│   └── README.md               # Dataset download and preparation instructions
│
├── models/                     # Saved model weights & checkpoints
│   ├── checkpoints/            # Intermediate training checkpoints
│   └── README.md               # Model storage notes
│
├── notebooks/                  # Interactive experimentation & Colab training
│   └── README.md               # Notebook guidelines for EDA & GPU training
│
├── src/                        # Core source code modules
│   ├── __init__.py
│   ├── preprocessing.py        # Image loading, patient-level splits & augmentation
│   ├── train_convnext.py       # ConvNeXt model builder & training script
│   ├── train_efficientnet.py   # EfficientNetV2-B3 + Hybrid Attention (CBAM) training
│   ├── train_vit.py            # Vision Transformer (ViT) patch encoder & training
│   ├── ensemble.py             # Weighted probability averaging ensemble logic
│   ├── evaluate.py             # Metrics (F1, AUC-ROC, Accuracy) & confusion matrices
│   └── explainability.py       # Grad-CAM heatmap generation & visualization
│
├── tests/                      # Unit and sanity tests
│   ├── __init__.py
│   └── test_config.py          # Configuration and parameter tests
│
└── docs/                       # Technical reports and documentation
    └── architecture.md         # Detailed pipeline architecture & mathematical formulations
```

---

## 🏷️ Diagnostic Classes (HAM10000)

| Code | Diagnosis | Condition Nature |
| :--- | :--- | :--- |
| **akiec** | Actinic Keratoses and Intraepithelial Carcinoma | Pre-cancerous / Carcinoma |
| **bcc** | Basal Cell Carcinoma | Malignant |
| **bkl** | Benign Keratosis-like Lesions | Benign |
| **df** | Dermatofibroma | Benign |
| **mel** | Melanoma | Malignant |
| **nv** | Melanocytic Nevi (Moles) | Benign |
| **vasc** | Vascular Lesions | Benign |

---

## 🚀 Getting Started

### 1. Environment Setup (Windows Local)

#### Clone / Navigate to Project Directory:
```powershell
cd C:\Users\muvva\.gemini\antigravity\scratch\DermaVision
```

#### Create and Activate Virtual Environment:
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1
```

#### Install Dependencies:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

#### Run Sanity and Preprocessing Tests:
```powershell
python run_tests.py
# Or if pytest is installed:
pytest tests/
```

---

### 2. Dataset Ingestion & Preprocessing

1. Download the **HAM10000** dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) or the [ISIC Archive](https://www.isic-archive.com/).
2. Extract the files into `data/raw/`:
   - `HAM10000_metadata.csv`
   - `HAM10000_images_part_1/`
   - `HAM10000_images_part_2/`
3. Execute the preprocessing pipeline:
```powershell
python -m src.preprocessing
```
This validates image readability, prevents patient leakage using lesion-aware group stratification, and saves `train.csv`, `val.csv`, and `test.csv` to `data/processed/`.

---

### 3. Model Training (Local or Google Colab)

For fast GPU training, Google Colab (T4 / A100) or a local NVIDIA GPU is recommended.

#### A. Train EfficientNetV2-B3 with Hybrid Attention:
```powershell
python src/train_efficientnet.py --epochs 30 --batch-size 32 --lr 0.0001
```

#### B. Train ConvNeXt:
```powershell
python src/train_convnext.py --epochs 30 --batch-size 32 --lr 0.0001
```

#### C. Train Vision Transformer (ViT):
```powershell
python src/train_vit.py --epochs 30 --batch-size 32 --lr 0.0001
```

Trained weights will be saved in `models/`.

---

### 4. Running the Streamlit Web Application

Launch the frontend dashboard:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ⚕️ Academic & Ethical Disclaimer
This project is developed for B.Tech capstone academic evaluation. It is not an FDA/CE-approved medical device and should not replace clinical diagnoses by certified medical professionals.
