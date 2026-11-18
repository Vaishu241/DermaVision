"""
preprocessing.py - HAM10000 Data Ingestion, Group-Aware Splitting & Preprocessing.

Base Research Paper: "Skin Cancer Detection and Classification Using Deep Learning Techniques" (2026 IEEE AIDE).

Features:
- Image path matching across split raw folders (HAM10000_images_part_1 & part_2).
- Metadata validation, integrity verification, and dataset summary reporting.
- Group-aware (lesion_id) stratified splitting to eliminate patient data leakage.
- Resolution standardization to 256x256 RGB float32 [0.0, 1.0].
- Training-only data augmentation pipeline (flips, rotations, zooms, contrast).
- Robust label mapping ('dx' <-> 'target') and stratified smoke-test subsetting.
- Balanced inverse class weight calculation for all 7 classes.
- High-performance tf.data pipeline creation with prefetching.
- Command-line interface runnable via: python -m src.preprocessing
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any, Union
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe/lazy imports for optional heavy libraries
try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    from sklearn.utils.class_weight import compute_class_weight
except ImportError:
    StratifiedGroupKFold = None
    train_test_split = None
    compute_class_weight = None

import config


# ---------------------------------------------------------------------------
# 1. Dataset Availability & Setup Instructions
# ---------------------------------------------------------------------------

def check_raw_dataset_exists() -> Dict[str, Any]:
    """
    Check if HAM10000 raw files exist in data/raw/ or project root.

    Returns:
        dict: Status report containing booleans for metadata and image directories.
    """
    metadata_exists = config.METADATA_PATH.is_file()
    part1_exists = config.RAW_PART1_DIR.is_dir()
    part2_exists = config.RAW_PART2_DIR.is_dir()
    all_images_dir = config.RAW_DATA_DIR / "all_images"
    all_images_exists = all_images_dir.is_dir()

    images_available = (part1_exists or part2_exists or all_images_exists)
    is_ready = metadata_exists and images_available

    return {
        "metadata_exists": metadata_exists,
        "part1_exists": part1_exists,
        "part2_exists": part2_exists,
        "all_images_exists": all_images_exists,
        "is_ready": is_ready
    }


def print_dataset_instructions():
    """Display instructions for obtaining and organizing the HAM10000 dataset."""
    print("=" * 80)
    print("HAM10000 DATASET NOT DETECTED IN data/raw/")
    print("=" * 80)
    print("Please follow these steps to place the dataset:")
    print("1. Download the HAM10000 dataset from Kaggle or the ISIC Archive:")
    print("   https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
    print("\n2. Extract the archives into the following directory layout:")
    print(f"   {config.RAW_DATA_DIR.resolve()}\\")
    print(f"   |-- HAM10000_metadata.csv")
    print(f"   |-- HAM10000_images_part_1/   (ISIC_0024306.jpg, ...)")
    print(f"   \\-- HAM10000_images_part_2/   (ISIC_0029306.jpg, ...)")
    print("\n   (Alternatively, all images can be merged into a single folder 'data/raw/all_images/')")
    print("\n3. Once extracted, rerun the preprocessing pipeline:")
    print("   python -m src.preprocessing")
    print("=" * 80)


# ---------------------------------------------------------------------------
# 2. Path Matching & Integrity Checks
# ---------------------------------------------------------------------------

def find_image_file(image_id: str, search_dirs: List[Path]) -> Optional[Path]:
    """
    Locate an image file corresponding to an image_id across multiple folders.
    Supports .jpg, .jpeg, and .png extensions.
    """
    for folder in search_dirs:
        if not folder.is_dir():
            continue
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            candidate = folder / f"{image_id}{ext}"
            if candidate.is_file():
                return candidate
    return None


def match_image_paths(df: pd.DataFrame, search_dirs: Optional[List[Path]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Match each metadata image_id with its physical image file path.

    Args:
        df: Metadata DataFrame containing 'image_id'.
        search_dirs: Directories to search for image files.

    Returns:
        Tuple: (updated_df_with_image_path, summary_dict)
    """
    if search_dirs is None:
        search_dirs = [
            config.RAW_PART1_DIR,
            config.RAW_PART2_DIR,
            config.RAW_DATA_DIR / "all_images",
            config.RAW_DATA_DIR
        ]

    # Check for duplicate image_ids in metadata
    duplicate_mask = df.duplicated(subset=['image_id'], keep=False)
    duplicate_count = duplicate_mask.sum()
    duplicate_ids = df.loc[duplicate_mask, 'image_id'].unique().tolist() if duplicate_count > 0 else []

    # Build fast index mapping filename stem -> resolved path
    lookup = {}
    for folder in search_dirs:
        if folder.is_dir():
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    lookup[f.stem] = str(f.resolve())

    paths = []
    missing_ids = []

    for img_id in df['image_id']:
        found_path = lookup.get(str(img_id))
        if found_path is not None:
            paths.append(found_path)
        else:
            paths.append(None)
            missing_ids.append(img_id)

    matched_df = df.copy()
    matched_df['image_path'] = paths

    # Ensure target mapping is aligned
    if 'dx' in matched_df.columns:
        matched_df['target'] = matched_df['dx'].map(config.LABEL_TO_IDX)

    summary = {
        "total_records": len(df),
        "matched_count": len(df) - len(missing_ids),
        "missing_count": len(missing_ids),
        "missing_ids": missing_ids,
        "duplicate_id_count": duplicate_count,
        "duplicate_ids": duplicate_ids
    }

    return matched_df, summary


# ---------------------------------------------------------------------------
# 3. Validation & Dataset Summary
# ---------------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame, sample_check_limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Validate image readability, channels, dimensions, and diagnostic labels.

    Args:
        df: DataFrame with 'image_path' and 'dx' columns.
        sample_check_limit: If specified, validates image readability on a sample of images.

    Returns:
        Dict: Validation report with class counts, dimensions, and readability stats.
    """
    # 1. Label validation
    invalid_labels = df[~df['dx'].isin(config.CLASS_NAMES)]
    invalid_label_count = len(invalid_labels)
    class_counts = df['dx'].value_counts().to_dict()

    # 2. Image readability and dimensions check
    valid_paths_df = df[df['image_path'].notna()]
    check_df = valid_paths_df if sample_check_limit is None else valid_paths_df.head(sample_check_limit)

    unreadable_files = []
    detected_dimensions = set()
    detected_modes = set()

    for idx, row in check_df.iterrows():
        img_path = row['image_path']
        try:
            with Image.open(img_path) as img:
                detected_dimensions.add(img.size)  # (width, height)
                detected_modes.add(img.mode)
        except Exception as e:
            unreadable_files.append((img_path, str(e)))

    report = {
        "total_images": len(df),
        "images_with_valid_paths": len(valid_paths_df),
        "class_counts": class_counts,
        "invalid_label_count": invalid_label_count,
        "unreadable_count": len(unreadable_files),
        "unreadable_files": unreadable_files,
        "detected_dimensions": [f"{w}x{h}" for w, h in detected_dimensions],
        "detected_color_modes": list(detected_modes),
    }

    return report


def print_dataset_summary(report: Dict[str, Any]):
    """Print a clean, structured summary table of dataset validation results."""
    print("\n" + "=" * 70)
    print("                    HAM10000 DATASET SUMMARY")
    print("=" * 70)
    print(f"Total Metadata Records:   {report['total_images']}")
    print(f"Images with Valid Paths:  {report['images_with_valid_paths']}")
    print(f"Unreadable Image Files:   {report['unreadable_count']}")
    print(f"Invalid Diagnostic Labels: {report['invalid_label_count']}")
    print(f"Detected Dimensions:      {', '.join(report['detected_dimensions']) if report['detected_dimensions'] else 'N/A'}")
    print(f"Detected Color Modes:     {', '.join(report['detected_color_modes']) if report['detected_color_modes'] else 'N/A'}")

    print("\nClass Distribution:")
    print("-" * 70)
    print(f"{'Class Code':<10} | {'Diagnostic Description':<42} | {'Count':<8} | {'Ratio':<8}")
    print("-" * 70)
    total = report['total_images']
    for cls in config.CLASS_NAMES:
        cnt = report['class_counts'].get(cls, 0)
        pct = (cnt / total * 100) if total > 0 else 0
        desc = config.CLASS_LABELS.get(cls, cls)
        print(f"{cls:<10} | {desc:<42} | {cnt:<8} | {pct:>6.2f}%")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# 4. Group-Aware Stratified Splitting (Preventing Patient/Lesion Leakage)
# ---------------------------------------------------------------------------

def split_data_lesion_aware(df: pd.DataFrame,
                             test_ratio: float = config.TEST_RATIO,
                             val_ratio: float = config.VAL_RATIO,
                             random_state: int = config.RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform group-aware train/val/test split based on 'lesion_id'.

    IMPORTANT RESEARCH & CLINICAL CONTEXT:
    In HAM10000, 10,015 images represent 7,470 unique lesions (lesion_id).
    Many lesions have 2, 3, or more images taken at different times or magnifications.
    Splitting images randomly causes severe data leakage (identical lesions in train & test),
    artificially inflating evaluation accuracy by up to 10-15%.

    This function groups unique lesions, performs stratified splitting on the unique lesions,
    and then maps all images of each lesion into the corresponding split.
    """
    if 'lesion_id' not in df.columns:
        raise ValueError("DataFrame must contain 'lesion_id' column for group-aware splitting.")
    if 'dx' not in df.columns:
        raise ValueError("DataFrame must contain 'dx' column for class stratification.")

    # Aggregate at the lesion level (each lesion_id maps to its diagnosis)
    lesion_df = df.groupby('lesion_id')['dx'].agg(lambda x: x.iloc[0]).reset_index()

    train_val_ratio = 1.0 - test_ratio

    # First split: Split unique lesions into (train+val) and (test)
    train_val_lesions, test_lesions = train_test_split(
        lesion_df,
        test_size=test_ratio,
        stratify=lesion_df['dx'],
        random_state=random_state
    )

    # Second split: Split (train+val) lesions into (train) and (val)
    val_relative_ratio = val_ratio / train_val_ratio
    train_lesions, val_lesions = train_test_split(
        train_val_lesions,
        test_size=val_relative_ratio,
        stratify=train_val_lesions['dx'],
        random_state=random_state
    )

    # Map back all images corresponding to the assigned lesions
    train_set = set(train_lesions['lesion_id'])
    val_set = set(val_lesions['lesion_id'])
    test_set = set(test_lesions['lesion_id'])

    # Strict leakage validation
    assert train_set.isdisjoint(val_set), "CRITICAL: Overlap between Train and Validation lesions!"
    assert train_set.isdisjoint(test_set), "CRITICAL: Overlap between Train and Test lesions!"
    assert val_set.isdisjoint(test_set), "CRITICAL: Overlap between Validation and Test lesions!"

    train_df = df[df['lesion_id'].isin(train_set)].copy().reset_index(drop=True)
    val_df = df[df['lesion_id'].isin(val_set)].copy().reset_index(drop=True)
    test_df = df[df['lesion_id'].isin(test_set)].copy().reset_index(drop=True)

    # Add numeric target index strictly matching config.LABEL_TO_IDX
    train_df['target'] = train_df['dx'].map(config.LABEL_TO_IDX)
    val_df['target'] = val_df['dx'].map(config.LABEL_TO_IDX)
    test_df['target'] = test_df['dx'].map(config.LABEL_TO_IDX)

    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                test_df: pd.DataFrame,
                output_dir: Path = config.PROCESSED_DATA_DIR):
    """Save split metadata to data/processed/ without duplicating images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"Successfully saved split metadata to {output_dir.resolve()}:")
    print(f"  - train.csv: {len(train_df)} images ({len(train_df['lesion_id'].unique())} unique lesions)")
    print(f"  - val.csv:   {len(val_df)} images ({len(val_df['lesion_id'].unique())} unique lesions)")
    print(f"  - test.csv:  {len(test_df)} images ({len(test_df['lesion_id'].unique())} unique lesions)")


# ---------------------------------------------------------------------------
# 5. Image Preprocessing & Normalization
# ---------------------------------------------------------------------------

def load_and_preprocess_image(image_path: Union[str, Path],
                              target_size: Tuple[int, int] = config.IMAGE_SIZE,
                              scale_mode: str = "raw_255") -> np.ndarray:
    """
    Load an image, convert to RGB, resize to target_size (256x256), and scale according to scale_mode.

    Args:
        image_path: Path string to image file.
        target_size: Desired dimensions (width, height), default (256, 256).
        scale_mode: 'raw_255' for [0, 255] float32 (for models with internal rescaling/normalization),
                    or 'normalized_01' for [0, 1] float32. Defaults to 'raw_255'.

    Returns:
        np.ndarray: Preprocessed float32 array of shape (height, width, 3).
    """
    with Image.open(str(image_path)) as img:
        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize(target_size, Image.Resampling.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32)
        if scale_mode == "normalized_01":
            img_array = img_array / 255.0
        return img_array


def preprocess_image_for_model(image_input: Union[str, Path, np.ndarray, Any],
                               model_name: str = "efficientnet",
                               target_size: Tuple[int, int] = config.IMAGE_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Model-specific preprocessing contract adhering to architectural requirements.

    Contract:
    - EfficientNetV2: Raw [0.0, 255.0] float32 (internal Keras Rescaling(1/255) + Normalization).
    - ConvNeXt: Raw [0.0, 255.0] float32 (internal Keras Normalization expecting [0, 255]).
    - Vision Transformer (ViT): Normalized [0.0, 1.0] float32 (no internal rescaling layer).
    - Ensemble: Returns raw [0.0, 255.0] float32 batch tensor; model-specific conversions applied during dispatch.

    Args:
        image_input: Filepath (str/Path), numpy array, or PIL Image.
        model_name: Architecture key ('efficientnet', 'convnext', 'vit', 'ensemble').
        target_size: Desired dimensions (height, width).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - batch_tensor: (1, H, W, 3) float32 tensor with model-appropriate scaling.
            - original_uint8: (H, W, 3) uint8 image in [0, 255] for UI visualization / Grad-CAM.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        arr = image_input.copy()
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            img = Image.fromarray(arr).convert("RGB")
        elif arr.ndim == 3:
            img = Image.fromarray(arr)
        else:
            raise ValueError(f"Unsupported image array shape: {image_input.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Standardize dimensions (width, height)
    img_resized = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    original_uint8 = np.array(img_resized, dtype=np.uint8)

    # Determine scaling contract
    name_clean = model_name.lower()
    if name_clean == "vit":
        # ViT requires [0, 1] float32
        norm_array = original_uint8.astype(np.float32) / 255.0
    else:
        # EfficientNet, ConvNeXt, Ensemble require [0, 255] float32
        norm_array = original_uint8.astype(np.float32)

    batch_tensor = np.expand_dims(norm_array, axis=0)
    return batch_tensor, original_uint8


# ---------------------------------------------------------------------------
# 6. Data Augmentation (Training Only)
# ---------------------------------------------------------------------------

def get_data_augmentation_pipeline(image_size: Tuple[int, int] = config.IMAGE_SIZE):
    """
    Build Keras data augmentation pipeline applied ONLY to training images.

    Includes:
    - Random horizontal and vertical flips (dermoscopic lesions are orientation-invariant)
    - Small rotation (factor 0.1 = ~36 degrees)
    - Small zoom (factor 0.1 = +/- 10%)
    - Small contrast adjustment (factor 0.1)

    Returns:
        tf.keras.Sequential: Augmentation pipeline.
    """
    if tf is None:
        raise ImportError("TensorFlow must be installed to create Keras augmentation layers.")

    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", name="aug_random_flip"),
        tf.keras.layers.RandomRotation(0.1, fill_mode="reflect", name="aug_random_rotation"),
        tf.keras.layers.RandomZoom(0.1, fill_mode="reflect", name="aug_random_zoom"),
        tf.keras.layers.RandomContrast(0.1, name="aug_random_contrast"),
    ], name="training_augmentation_pipeline")


# ---------------------------------------------------------------------------
# 7. Class Weight Calculation & Smoke Test Subsetting
# ---------------------------------------------------------------------------

def compute_class_weights(df: pd.DataFrame,
                          target_col: str = "target",
                          num_classes: int = config.NUM_CLASSES) -> Dict[int, float]:
    """
    Compute balanced class weights using inverse class frequency.
    Ensures all 7 classes have a weight defined even if some classes are missing in small subsets.
    """
    df = df.copy()
    if target_col not in df.columns:
        if 'dx' in df.columns:
            df['target'] = df['dx'].map(config.LABEL_TO_IDX)
            target_col = 'target'
        else:
            raise KeyError(f"Column '{target_col}' not found. Available: {list(df.columns)}")

    labels = df[target_col].dropna().astype(int).to_numpy()
    if len(labels) == 0:
        return {c: 1.0 for c in range(num_classes)}

    if compute_class_weight is not None:
        present_classes = np.sort(np.unique(labels))
        weights = compute_class_weight(class_weight="balanced", classes=present_classes, y=labels)
        weight_dict = {int(c): float(w) for c, w in zip(present_classes, weights)}
    else:
        # Fallback manual inverse frequency calculation
        total_samples = len(labels)
        class_counts = pd.Series(labels).value_counts().to_dict()
        weight_dict = {
            int(c): float(total_samples / (len(class_counts) * count))
            for c, count in class_counts.items()
        }

    # Ensure every single class in [0, num_classes-1] has a defined weight
    for idx in range(num_classes):
        if idx not in weight_dict:
            weight_dict[idx] = 1.0

    return weight_dict


def create_smoke_test_subset(df: pd.DataFrame,
                             samples_per_class: int = 5,
                             random_state: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Create a representative stratified subset containing samples from all available classes.
    Filters for valid existing physical image files.
    """
    df = df.copy()
    if 'target' not in df.columns and 'dx' in df.columns:
        df['target'] = df['dx'].map(config.LABEL_TO_IDX)

    # Filter to valid physical files if image_path exists
    if 'image_path' in df.columns:
        df = df[df['image_path'].apply(lambda p: os.path.isfile(str(p)))].reset_index(drop=True)

    samples = []
    for cls_name, cls_idx in config.LABEL_TO_IDX.items():
        cls_df = df[(df['dx'] == cls_name) | (df['target'] == cls_idx)]
        if len(cls_df) > 0:
            n_take = min(samples_per_class, len(cls_df))
            samples.append(cls_df.sample(n=n_take, random_state=random_state))

    if samples:
        smoke_df = pd.concat(samples, ignore_index=True)
        return smoke_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df.head(samples_per_class * config.NUM_CLASSES)


# ---------------------------------------------------------------------------
# 8. High-Performance tf.data.Dataset Pipeline
# ---------------------------------------------------------------------------

def _parse_image_tf(image_path: Any, label: Any, image_size: Tuple[int, int], scale_to_unit: bool = False):
    """Read and decode image using TensorFlow native ops. Preserves raw [0, 255] float32 by default."""
    img_bytes = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.cast(img, tf.float32)
    if scale_to_unit:
        img = img / 255.0
    img = tf.image.resize(img, image_size)
    return img, label


def create_tf_dataset(df: pd.DataFrame,
                      batch_size: int = config.BATCH_SIZE,
                      is_training: bool = True,
                      image_size: Tuple[int, int] = config.IMAGE_SIZE,
                      scale_to_unit: bool = False) -> Any:
    """
    Construct high-performance tf.data pipeline with batching, augmentation, and prefetching.
    Preserves raw [0, 255] float32 values so models with internal Rescaling/Normalization work correctly.

    Args:
        df: DataFrame with 'image_path' and 'target' (or 'dx') columns.
        batch_size: Batch size (default 32).
        is_training: If True, applies shuffling and data augmentation.
        image_size: Target image dimensions (default 256x256).
        scale_to_unit: Whether to divide pixel values by 255 (default False, preserves [0, 255]).

    Returns:
        tf.data.Dataset: Ready-to-train dataset pipeline.
    """
    if tf is None:
        raise ImportError("TensorFlow is required to build tf.data.Dataset pipelines.")

    df = df.copy()
    if 'target' not in df.columns:
        if 'dx' in df.columns:
            df['target'] = df['dx'].map(config.LABEL_TO_IDX)
        else:
            raise KeyError("DataFrame must contain 'target' or 'dx' column for labels.")
    elif 'dx' in df.columns and df['target'].isna().any():
        df['target'] = df['dx'].map(config.LABEL_TO_IDX)

    image_paths = df['image_path'].astype(str).values
    labels = df['target'].astype(int).values
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=config.NUM_CLASSES)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, one_hot_labels))

    # Read and decode in parallel
    dataset = dataset.map(
        lambda path, lbl: _parse_image_tf(path, lbl, image_size, scale_to_unit),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(df), 1000), seed=config.RANDOM_SEED)
        dataset = dataset.batch(batch_size)
        aug_model = get_data_augmentation_pipeline(image_size)
        dataset = dataset.map(
            lambda x, y: (aug_model(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------------
# 9. Main Pipeline Execution
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline() -> bool:
    """Execute full HAM10000 preprocessing pipeline."""
    print("\n--- [DermaVision] Step 2: HAM10000 Preprocessing Pipeline ---")
    status = check_raw_dataset_exists()

    if not status["is_ready"]:
        print_dataset_instructions()
        return False

    print(f"Loading metadata from {config.METADATA_PATH}...")
    df = pd.read_csv(config.METADATA_PATH)
    print(f"Total raw metadata records loaded: {len(df)}")

    print("Matching image files across raw partitions...")
    matched_df, match_summary = match_image_paths(df)

    if match_summary["missing_count"] > 0:
        print(f"WARNING: {match_summary['missing_count']} images listed in metadata could not be found!")
    if match_summary["duplicate_id_count"] > 0:
        print(f"WARNING: {match_summary['duplicate_id_count']} duplicate image IDs found in metadata!")

    print("Validating dataset integrity...")
    val_report = validate_dataset(matched_df, sample_check_limit=100)
    print_dataset_summary(val_report)

    # Discard records with missing image paths
    clean_df = matched_df[matched_df['image_path'].notna()].copy()

    # Split dataset with lesion-aware group stratification
    print("Performing lesion-aware stratified train/val/test splitting...")
    train_df, val_df, test_df = split_data_lesion_aware(
        clean_df,
        test_ratio=config.TEST_RATIO,
        val_ratio=config.VAL_RATIO,
        random_state=config.RANDOM_SEED
    )

    save_splits(train_df, val_df, test_df)

    class_weights = compute_class_weights(train_df)
    print("\nCalculated Inverse Class Weights for Training:")
    for cls_idx, weight in sorted(class_weights.items()):
        cls_name = config.IDX_TO_LABEL[cls_idx]
        print(f"  Class {cls_idx} ({cls_name:<6}): {weight:.4f}")

    print("\n--- Preprocessing Pipeline Completed Successfully ---")
    return True


if __name__ == "__main__":
    success = run_preprocessing_pipeline()
    sys.exit(0 if success else 1)
