# DermaVision: Skin Cancer Detection Using EfficientNetV2 with Hybrid Attention and Ensemble Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)

**B.Tech Capstone Project**  
Base Research Paper: *"Skin Cancer Detection and Classification Using Deep Learning Techniques"* (2026 IEEE AIDE).

---

## 🔬 Project Overview

DermaVision is an end-to-end computer vision and Explainable Artificial Intelligence (XAI) framework developed to classify dermatoscopic skin lesion images into seven diagnostic categories using the HAM10000 dataset.

The proposed system integrates three deep learning architectures: EfficientNetV2-B3 with Hybrid Attention (CBAM), ConvNeXt, and Vision Transformer (ViT) to learn complementary features from skin lesion images. EfficientNetV2-B3 with CBAM utilizes channel and spatial attention mechanisms to focus on relevant image features, while ConvNeXt captures detailed visual patterns and Vision Transformer identifies global contextual relationships through self-attention mechanisms.

To improve prediction performance, a Weighted Ensemble Learning approach combines the probability outputs of all three models to generate the final classification. The system also incorporates Grad-CAM (Gradient-weighted Class Activation Mapping) for explainability by generating visual heatmaps that highlight image regions influencing the model's predictions.

Additionally, DermaVision provides an interactive web-based interface developed using Streamlit, allowing users to upload dermatoscopic images and view the predicted classification along with explainability visualizations.

The project aims to demonstrate the application of deep learning, ensemble learning, and explainable AI in automated skin lesion classification for academic and research purposes.

---


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
cd C:\Users\<your-username>\DermaVision
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
