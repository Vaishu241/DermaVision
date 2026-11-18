"""
tests/test_preprocessing.py - Unit and integration tests for HAM10000 preprocessing pipeline.
Supports both pytest (pytest tests/) and direct Python execution (python tests/test_preprocessing.py).
"""

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.preprocessing import (
    match_image_paths,
    validate_dataset,
    split_data_lesion_aware,
    load_and_preprocess_image,
    preprocess_image_for_model,
    compute_class_weights
)


def create_mock_env(base_dir: Path):
    """Helper to populate mock dataset directory and metadata."""
    part1_dir = base_dir / "part_1"
    part2_dir = base_dir / "part_2"
    part1_dir.mkdir(parents=True, exist_ok=True)
    part2_dir.mkdir(parents=True, exist_ok=True)

    # Generate 4 sample images
    Image.new("RGB", (600, 450), color="red").save(part1_dir / "ISIC_0001.jpg")
    Image.new("RGB", (600, 450), color="blue").save(part1_dir / "ISIC_0002.jpg")
    Image.new("RGB", (600, 450), color="green").save(part2_dir / "ISIC_0003.jpg")
    Image.new("RGB", (400, 300), color="yellow").save(part2_dir / "ISIC_0004.jpg")

    # Metadata with multi-image lesions, a missing image (ISIC_0005), and a duplicate ID
    metadata = pd.DataFrame([
        {"lesion_id": "HAM_001", "image_id": "ISIC_0001", "dx": "mel"},
        {"lesion_id": "HAM_001", "image_id": "ISIC_0002", "dx": "mel"},  # Same lesion, 2 images
        {"lesion_id": "HAM_002", "image_id": "ISIC_0003", "dx": "nv"},
        {"lesion_id": "HAM_003", "image_id": "ISIC_0004", "dx": "bcc"},
        {"lesion_id": "HAM_004", "image_id": "ISIC_0005", "dx": "bkl"},  # Missing physical file
        {"lesion_id": "HAM_005", "image_id": "ISIC_0001", "dx": "mel"},  # Duplicate image_id
    ])

    return {
        "part1_dir": part1_dir,
        "part2_dir": part2_dir,
        "metadata": metadata
    }


def test_image_path_matching(mock_env=None):
    """Test matching image IDs across multiple folders and detecting missing files and duplicate IDs."""
    temp_dir = None
    if mock_env is None:
        temp_dir = tempfile.TemporaryDirectory()
        mock_env = create_mock_env(Path(temp_dir.name))

    try:
        search_dirs = [mock_env["part1_dir"], mock_env["part2_dir"]]
        matched_df, summary = match_image_paths(mock_env["metadata"], search_dirs=search_dirs)

        assert summary["total_records"] == 6
        assert summary["missing_count"] == 1
        assert "ISIC_0005" in summary["missing_ids"]
        assert summary["duplicate_id_count"] == 2
        assert "ISIC_0001" in summary["duplicate_ids"]

        matched_only = matched_df[matched_df['image_path'].notna()]
        for p in matched_only['image_path']:
            assert Path(p).is_file()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def test_label_and_image_validation(mock_env=None):
    """Test dataset validation detects image dimensions and invalid diagnostic labels."""
    temp_dir = None
    if mock_env is None:
        temp_dir = tempfile.TemporaryDirectory()
        mock_env = create_mock_env(Path(temp_dir.name))

    try:
        search_dirs = [mock_env["part1_dir"], mock_env["part2_dir"]]
        matched_df, _ = match_image_paths(mock_env["metadata"], search_dirs=search_dirs)

        # Introduce an invalid label for testing
        matched_df.loc[0, 'dx'] = 'invalid_cancer_type'

        report = validate_dataset(matched_df)
        assert report["invalid_label_count"] == 1
        assert report["unreadable_count"] == 0
        assert "600x450" in report["detected_dimensions"]
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def test_group_aware_split_no_lesion_leakage():
    """
    CRITICAL TEST: Ensure images belonging to the same lesion_id NEVER leak across train, val, or test splits.
    """
    # Create multi-image lesions distributed across all 7 classes
    records = []
    classes = config.CLASS_NAMES

    for i in range(70):  # 70 unique lesions (10 per class)
        dx = classes[i % len(classes)]
        lesion_id = f"LESION_{i:04d}"
        # Some lesions have 1 image, others have 2 or 3 images
        num_images = 1 if i % 3 == 0 else (2 if i % 3 == 1 else 3)
        for img_idx in range(num_images):
            records.append({
                "lesion_id": lesion_id,
                "image_id": f"IMG_{lesion_id}_{img_idx}",
                "dx": dx
            })

    df = pd.DataFrame(records)
    total_images = len(df)
    unique_lesions = df['lesion_id'].nunique()

    train_df, val_df, test_df = split_data_lesion_aware(
        df, test_ratio=0.20, val_ratio=0.20, random_state=42
    )

    # Verify image sum matches total
    assert len(train_df) + len(val_df) + len(test_df) == total_images

    # Extract lesion sets
    train_lesions = set(train_df['lesion_id'])
    val_lesions = set(val_df['lesion_id'])
    test_lesions = set(test_df['lesion_id'])

    # Total unique lesions must equal sum of split lesions (disjoint sets)
    assert len(train_lesions) + len(val_lesions) + len(test_lesions) == unique_lesions

    # ZERO LEAKAGE ASSERTIONS
    assert train_lesions.isdisjoint(val_lesions), "LEAKAGE: Overlap between Train and Val lesions!"
    assert train_lesions.isdisjoint(test_lesions), "LEAKAGE: Overlap between Train and Test lesions!"
    assert val_lesions.isdisjoint(test_lesions), "LEAKAGE: Overlap between Val and Test lesions!"

    # Target indices check
    for split_df in [train_df, val_df, test_df]:
        assert 'target' in split_df.columns
        assert set(split_df['target'].unique()).issubset(set(range(config.NUM_CLASSES)))


def test_image_preprocessing_resize_and_normalization():
    """Test image loading converts any format to (256, 256, 3) float32 with appropriate scaling."""
    with tempfile.TemporaryDirectory() as td:
        test_img_path = Path(td) / "test_sample.png"
        img = Image.new("RGBA", (512, 384), color=(200, 100, 50, 255))
        img.save(test_img_path)

        # 1. Default raw_255 mode
        processed_raw = load_and_preprocess_image(str(test_img_path), target_size=(256, 256), scale_mode="raw_255")
        assert processed_raw.shape == (256, 256, 3)
        assert processed_raw.dtype == np.float32
        assert np.isfinite(processed_raw).all()
        assert processed_raw.max() > 1.0

        # 2. normalized_01 mode
        processed_01 = load_and_preprocess_image(str(test_img_path), target_size=(256, 256), scale_mode="normalized_01")
        assert processed_01.shape == (256, 256, 3)
        assert processed_01.dtype == np.float32
        assert np.isfinite(processed_01).all()
        assert processed_01.min() >= 0.0 and processed_01.max() <= 1.0


def test_model_specific_preprocessing_contracts():
    """
    CRITICAL TEST (Stage 2 Contract):
    Verify synthetic images for all architectures satisfy:
    - Output shape (1, 256, 256, 3)
    - Dtype float32
    - Finite values (no NaN, no Inf)
    - Channel order (RGB verified: R=255, G=100, B=50)
    - Expected value ranges:
      * EfficientNet: raw [0.0, 255.0] (for internal Rescaling/Normalization)
      * ConvNeXt: raw [0.0, 255.0] (for internal Normalization)
      * ViT: normalized [0.0, 1.0] (no internal rescaling)
    """
    # Create synthetic RGB test array with distinct channels
    # Channel 0 (R) = 220, Channel 1 (G) = 110, Channel 2 (B) = 40
    synthetic_img = np.zeros((300, 400, 3), dtype=np.uint8)
    synthetic_img[:, :, 0] = 220
    synthetic_img[:, :, 1] = 110
    synthetic_img[:, :, 2] = 40

    for model_name, expected_max, expected_scale_desc in [
        ("efficientnet", 220.0, "raw_255"),
        ("convnext", 220.0, "raw_255"),
        ("ensemble", 220.0, "raw_255"),
        ("vit", 220.0 / 255.0, "normalized_01")
    ]:
        batch_tensor, orig_uint8 = preprocess_image_for_model(
            synthetic_img, model_name=model_name, target_size=(256, 256)
        )

        # 1. Output shape check
        assert batch_tensor.shape == (1, 256, 256, 3), f"{model_name} shape mismatch: {batch_tensor.shape}"
        assert orig_uint8.shape == (256, 256, 3), f"{model_name} visualization image shape mismatch"

        # 2. Dtype check
        assert batch_tensor.dtype == np.float32, f"{model_name} tensor must be float32"
        assert orig_uint8.dtype == np.uint8, f"{model_name} original must be uint8"

        # 3. Finite values check
        assert np.isfinite(batch_tensor).all(), f"{model_name} tensor contains NaN or Inf!"

        # 4. Channel order verification (RGB ordering preserved)
        if expected_scale_desc == "raw_255":
            assert np.allclose(batch_tensor[0, 0, 0, 0], 220.0, atol=1e-3), "Red channel mismatch"
            assert np.allclose(batch_tensor[0, 0, 0, 1], 110.0, atol=1e-3), "Green channel mismatch"
            assert np.allclose(batch_tensor[0, 0, 0, 2], 40.0, atol=1e-3), "Blue channel mismatch"
        else:
            assert np.allclose(batch_tensor[0, 0, 0, 0], 220.0 / 255.0, atol=1e-3), "Red channel normalized mismatch"
            assert np.allclose(batch_tensor[0, 0, 0, 1], 110.0 / 255.0, atol=1e-3), "Green channel normalized mismatch"
            assert np.allclose(batch_tensor[0, 0, 0, 2], 40.0 / 255.0, atol=1e-3), "Blue channel normalized mismatch"

        # 5. Value ranges check
        if expected_scale_desc == "raw_255":
            assert batch_tensor.max() > 1.0, f"{model_name} should not be normalized to [0, 1]"
            assert batch_tensor.max() <= 255.0, f"{model_name} exceeds 255"
        else:
            assert batch_tensor.max() <= 1.0, f"{model_name} exceeds 1.0"
            assert batch_tensor.min() >= 0.0, f"{model_name} below 0.0"


def test_class_weights_calculation():
    """Test inverse frequency class weight computation."""
    data = {"target": [config.LABEL_TO_IDX["nv"]] * 80 +
                      [config.LABEL_TO_IDX["mel"]] * 10 +
                      [config.LABEL_TO_IDX["bcc"]] * 10}
    df = pd.DataFrame(data)

    weights = compute_class_weights(df, target_col="target")
    nv_idx = config.LABEL_TO_IDX["nv"]
    mel_idx = config.LABEL_TO_IDX["mel"]

    assert weights[nv_idx] < weights[mel_idx]


if __name__ == "__main__":
    print("Running Preprocessing Test Suite...")
    test_image_path_matching()
    print("  [PASS] test_image_path_matching")
    test_label_and_image_validation()
    print("  [PASS] test_label_and_image_validation")
    test_group_aware_split_no_lesion_leakage()
    print("  [PASS] test_group_aware_split_no_lesion_leakage (Zero patient leakage verified!)")
    test_image_preprocessing_resize_and_normalization()
    print("  [PASS] test_image_preprocessing_resize_and_normalization")
    test_model_specific_preprocessing_contracts()
    print("  [PASS] test_model_specific_preprocessing_contracts (Shape, dtype, RGB order, scaling verified!)")
    test_class_weights_calculation()
    print("  [PASS] test_class_weights_calculation")
    print("\nAll Preprocessing Tests Passed Successfully!")
