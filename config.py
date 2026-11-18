"""
DermaVision: Skin Cancer Detection Using EfficientNetV2 with Hybrid Attention and Ensemble Learning
Configuration Module
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base Directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Check if dataset is in root folder (e.g. Downloads/DermaVision) or in data/raw
if (BASE_DIR / "HAM10000_metadata.csv").is_file():
    RAW_DATA_DIR = BASE_DIR
    METADATA_PATH = BASE_DIR / "HAM10000_metadata.csv"
    RAW_PART1_DIR = BASE_DIR / "HAM10000_images_part_1"
    RAW_PART2_DIR = BASE_DIR / "HAM10000_images_part_2"
else:
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    METADATA_PATH = RAW_DATA_DIR / "HAM10000_metadata.csv"
    RAW_PART1_DIR = RAW_DATA_DIR / "HAM10000_images_part_1"
    RAW_PART2_DIR = RAW_DATA_DIR / "HAM10000_images_part_2"

DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_CSV_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_CSV_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_CSV_PATH = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
SMOKE_TEST_DIR = MODELS_DIR / "smoke_test"

REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"

METRICS_JSON_PATH = REPORTS_DIR / "metrics_summary.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
ROC_CURVES_PATH = REPORTS_DIR / "roc_curves.png"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------------------
# Dataset Specifications (HAM10000)
# ---------------------------------------------------------------------------
# 7 skin lesion diagnostic categories (exact alphabetical order)
CLASS_NAMES = [
    "akiec",  # Actinic keratoses and intraepithelial carcinoma
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis-like lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevi
    "vasc"    # Vascular lesions
]

CLASS_LABELS = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

NUM_CLASSES = len(CLASS_NAMES)
LABEL_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {idx: name for idx, name in enumerate(CLASS_NAMES)}

MALIGNANCY_STATUS = {
    "akiec": "Pre-cancerous",
    "bcc": "Malignant",
    "bkl": "Benign",
    "df": "Benign",
    "mel": "Malignant",
    "nv": "Benign",
    "vasc": "Benign"
}

IS_MALIGNANT = {
    "akiec": True,  # Pre-cancerous intraepithelial carcinoma
    "bcc": True,   # Basal cell carcinoma
    "bkl": False,  # Benign keratosis-like lesions
    "df": False,   # Dermatofibroma
    "mel": True,   # Melanoma
    "nv": False,   # Melanocytic nevi
    "vasc": False  # Vascular lesions
}

# ---------------------------------------------------------------------------
# Image & Training Hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = (256, 256)
INPUT_SHAPE = (256, 256, 3)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Ensemble Weights (Weighted Probability Averaging)
# Reference defaults:
# - PRODUCTION_ENSEMBLE_WEIGHTS: 1.0 on validated EfficientNetV2-B3 production model
# - RESEARCH_ENSEMBLE_WEIGHTS: 0.40, 0.35, 0.25 (aligned with IEEE AIDE 2026 methodology)
# ---------------------------------------------------------------------------
PRODUCTION_ENSEMBLE_WEIGHTS = {
    "efficientnet": 1.0,  # EfficientNetV2-B3 + Hybrid Attention (Production Validated)
    "convnext": 0.0,      # ConvNeXt (Reserved for retrained checkpoint)
    "vit": 0.0            # Vision Transformer (Reserved for retrained checkpoint)
}

RESEARCH_ENSEMBLE_WEIGHTS = {
    "efficientnet": 0.40,  # EfficientNetV2-B3 + Hybrid Attention
    "convnext": 0.35,      # ConvNeXt
    "vit": 0.25            # Vision Transformer
}

# Active default weights protect against degraded ViT / smoke-test ConvNeXt
ENSEMBLE_WEIGHTS = PRODUCTION_ENSEMBLE_WEIGHTS.copy()

# ---------------------------------------------------------------------------
# Model Weight Checkpoint Filenames (Production & Smoke Test)
# ---------------------------------------------------------------------------
MODEL_PATHS = {
    "efficientnet": MODELS_DIR / "efficientnetv2_hybrid_attention.keras",
    "convnext": MODELS_DIR / "convnext_best.keras",
    "vit": MODELS_DIR / "vit_best.keras"
}

SMOKE_TEST_MODEL_PATHS = {
    "efficientnet": SMOKE_TEST_DIR / "smoke_efficientnetv2_hybrid.keras",
    "convnext": SMOKE_TEST_DIR / "smoke_convnext.keras",
    "vit": SMOKE_TEST_DIR / "smoke_vit.keras"
}
