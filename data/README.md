# HAM10000 Dataset Guide for DermaVision

## Dataset Overview
The **HAM10000** ("Human Against Machine with 10,000 training images") dataset consists of 10,015 dermatoscopic images of pigmented lesions.

### Diagnostic Categories (7 Classes):
1. **akiec**: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (Precancerous/Carcinoma)
2. **bcc**: Basal cell carcinoma (Malignant)
3. **bkl**: Benign keratosis-like lesions (solar lentigines / seborrheic keratoses / lichen-planus like keratoses) (Benign)
4. **df**: Dermatofibroma (Benign)
5. **mel**: Melanoma (Malignant)
6. **nv**: Melanocytic nevi (Benign)
7. **vasc**: Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas, hemorrhage) (Benign)

---

## Directory Setup

Place the downloaded HAM10000 files in this directory structure:

```
data/
├── raw/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   │   ├── ISIC_0024306.jpg
│   │   └── ...
│   └── HAM10000_images_part_2/
│       ├── ISIC_0029306.jpg
│       └── ...
└── processed/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

> **Note:** The images from part 1 and part 2 can also be merged into a single `all_images/` directory. `src/preprocessing.py` supports scanning both subfolders or a merged folder.
