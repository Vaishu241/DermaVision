"""
test_labels_and_predictions.py - Comprehensive Unit Tests for Stage 3.

Verifies:
1. Deterministic output neuron index-to-class mappings (0..6) matching HAM10000 alphabetical order.
2. Exact one-hot predictions for all 7 classes, confirming correct diagnostic codes and malignancy flags.
3. Top-k rank ordering, percentage formatting, and arbitrary probability distributions.
4. Shannon entropy, confidence margin, and out-of-distribution uncertainty quantification.
5. Specific Melanoma (mel, idx=4) vs Benign Keratosis (bkl, idx=2) differentiation regression test.
6. Object-oriented WeightedEnsemble helper methods.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.ensemble import (
    map_index_to_class,
    get_top_k_predictions,
    compute_prediction_uncertainty,
    WeightedEnsemble,
)


def test_deterministic_index_to_class_mapping():
    """Verify that every neuron index 0..6 deterministically maps to the correct HAM10000 class."""
    expected_order = [
        (0, "akiec", "Actinic Keratoses / Intraepithelial Carcinoma", "Pre-cancerous", True),
        (1, "bcc", "Basal Cell Carcinoma", "Malignant", True),
        (2, "bkl", "Benign Keratosis-like Lesions", "Benign", False),
        (3, "df", "Dermatofibroma", "Benign", False),
        (4, "mel", "Melanoma", "Malignant", True),
        (5, "nv", "Melanocytic Nevi", "Benign", False),
        (6, "vasc", "Vascular Lesions", "Benign", False),
    ]

    for idx, expected_code, expected_label, expected_malig, expected_is_malig in expected_order:
        meta = map_index_to_class(idx)
        assert meta["index"] == idx, f"Index mismatch for {idx}: got {meta['index']}"
        assert meta["code"] == expected_code, f"Code mismatch for index {idx}: expected {expected_code}, got {meta['code']}"
        assert meta["label"] == expected_label, f"Label mismatch for index {idx}: expected {expected_label}, got {meta['label']}"
        assert meta["malignancy"] == expected_malig, f"Malignancy mismatch for index {idx}: expected {expected_malig}, got {meta['malignancy']}"
        assert meta["is_malignant"] == expected_is_malig, f"is_malignant flag mismatch for {idx}"

        # Round-trip verification against config
        assert config.LABEL_TO_IDX[meta["code"]] == idx, f"Round-trip failed for code {meta['code']}"
        assert config.IDX_TO_LABEL[idx] == meta["code"], f"Config IDX_TO_LABEL failed for idx {idx}"

    # Test out of bounds
    for invalid_idx in [-1, 7, 10, -100]:
        try:
            map_index_to_class(invalid_idx)
            assert False, f"Expected IndexError for out-of-bounds index {invalid_idx}"
        except IndexError:
            pass

    # Test non-integer types
    for invalid_val in ["mel", None, [1], {"idx": 0}]:
        try:
            map_index_to_class(invalid_val)
            assert False, f"Expected TypeError for invalid type {type(invalid_val)}"
        except TypeError:
            pass


def test_one_hot_predictions_for_all_7_classes():
    """Verify one-hot probability vectors for all 7 classes produce accurate Top-1 predictions."""
    for class_idx in range(7):
        one_hot = np.zeros(7, dtype=np.float32)
        one_hot[class_idx] = 1.0

        top3 = get_top_k_predictions(one_hot, k=3)
        assert len(top3) == 3, f"Expected 3 predictions, got {len(top3)}"

        # Top 1 must match the class
        top1 = top3[0]
        expected_meta = map_index_to_class(class_idx)
        assert top1["rank"] == 1
        assert top1["index"] == class_idx
        assert top1["code"] == expected_meta["code"]
        assert top1["label"] == expected_meta["label"]
        assert abs(top1["probability"] - 1.0) < 1e-5
        assert top1["percentage"] == "100.00%"
        assert top1["is_malignant"] == expected_meta["is_malignant"]
        assert top1["malignancy"] == expected_meta["malignancy"]

        # Remaining ranks must have 0 probability
        assert abs(top3[1]["probability"] - 0.0) < 1e-5
        assert abs(top3[2]["probability"] - 0.0) < 1e-5


def test_top_k_rank_ordering_and_probabilities():
    """Verify rank ordering, slicing, and handling of various input shapes."""
    # Probability distribution:
    # 0 (akiec): 0.05, 1 (bcc): 0.10, 2 (bkl): 0.05, 3 (df): 0.00,
    # 4 (mel): 0.60, 5 (nv): 0.15, 6 (vasc): 0.05
    dist = [0.05, 0.10, 0.05, 0.00, 0.60, 0.15, 0.05]

    # Test 1D array
    top3 = get_top_k_predictions(np.array(dist), k=3)
    assert top3[0]["code"] == "mel" and abs(top3[0]["probability"] - 0.60) < 1e-4
    assert top3[1]["code"] == "nv" and abs(top3[1]["probability"] - 0.15) < 1e-4
    assert top3[2]["code"] == "bcc" and abs(top3[2]["probability"] - 0.10) < 1e-4
    assert top3[0]["rank"] == 1 and top3[1]["rank"] == 2 and top3[2]["rank"] == 3

    # Test 2D array of shape (1, 7)
    top3_2d = get_top_k_predictions(np.array([dist]), k=3)
    assert [p["code"] for p in top3_2d] == ["mel", "nv", "bcc"]

    # Test list input
    top3_list = get_top_k_predictions(dist, k=3)
    assert [p["code"] for p in top3_list] == ["mel", "nv", "bcc"]

    # Test k bounds
    top1 = get_top_k_predictions(dist, k=1)
    assert len(top1) == 1
    top7 = get_top_k_predictions(dist, k=7)
    assert len(top7) == 7
    top_all = get_top_k_predictions(dist, k=10)
    assert len(top_all) == 7  # Clamped to 7 classes

    # Test invalid shapes / values
    try:
        get_top_k_predictions([0.1, 0.2, 0.3])
        assert False, "Expected ValueError for vector of wrong length"
    except ValueError:
        pass

    try:
        get_top_k_predictions([np.nan, 0.2, 0.3, 0.1, 0.1, 0.2, 0.1])
        assert False, "Expected ValueError for NaN input"
    except ValueError:
        pass


def test_prediction_uncertainty_and_entropy():
    """Verify Shannon entropy calculation, confidence margin, and uncertainty categories."""
    # Case 1: High certainty (peaked on Melanoma)
    high_cert = [0.01, 0.01, 0.01, 0.01, 0.94, 0.01, 0.01]
    diag_high = compute_prediction_uncertainty(high_cert)
    assert diag_high["confidence"] >= 0.90
    assert diag_high["normalized_entropy"] < 0.25
    assert diag_high["margin"] > 0.90
    assert diag_high["is_uncertain"] is False
    assert diag_high["uncertainty_level"] == "Low"

    # Case 2: Maximum uncertainty / Uniform distribution (out-of-distribution noise)
    uniform = [1.0 / 7.0] * 7
    diag_uniform = compute_prediction_uncertainty(uniform)
    assert abs(diag_uniform["normalized_entropy"] - 1.0) < 1e-3
    assert abs(diag_uniform["margin"] - 0.0) < 1e-3
    assert diag_uniform["is_uncertain"] is True
    assert diag_uniform["uncertainty_level"] == "High"
    assert "Specialist" in diag_uniform["clinical_recommendation"] or "biopsy" in diag_uniform["clinical_recommendation"]

    # Case 3: Borderline ambiguity (Top-1 and Top-2 within 5%)
    borderline = [0.02, 0.02, 0.44, 0.02, 0.42, 0.05, 0.03]
    diag_borderline = compute_prediction_uncertainty(borderline)
    assert diag_borderline["margin"] < 0.15
    assert diag_borderline["uncertainty_level"] in ["Moderate", "High"]


def test_melanoma_vs_bkl_differentiation():
    """Regression test: Ensure Melanoma predictions are not misclassified as BKL."""
    # Scenario: Model produces high confidence for Melanoma (index 4)
    mel_probs = np.array([0.01, 0.02, 0.05, 0.01, 0.86, 0.03, 0.02])
    top_mel = get_top_k_predictions(mel_probs, k=3)

    assert top_mel[0]["code"] == "mel", f"Expected top prediction 'mel', got {top_mel[0]['code']}"
    assert top_mel[0]["is_malignant"] is True
    assert top_mel[0]["malignancy"] == "Malignant"
    assert top_mel[0]["index"] == 4

    # BKL (index 2) must be ranked lower with lower probability
    bkl_entry = [p for p in top_mel if p["code"] == "bkl"]
    if bkl_entry:
        assert bkl_entry[0]["rank"] > top_mel[0]["rank"]
        assert bkl_entry[0]["probability"] < top_mel[0]["probability"]

    # Inverse Scenario: High confidence for BKL (index 2)
    bkl_probs = np.array([0.02, 0.01, 0.88, 0.01, 0.04, 0.02, 0.02])
    top_bkl = get_top_k_predictions(bkl_probs, k=3)
    assert top_bkl[0]["code"] == "bkl"
    assert top_bkl[0]["is_malignant"] is False
    assert top_bkl[0]["malignancy"] == "Benign"


def test_weighted_ensemble_methods():
    """Verify WeightedEnsemble wrapper methods match standalone functions."""
    ensemble = WeightedEnsemble()
    dist = [0.05, 0.10, 0.05, 0.00, 0.60, 0.15, 0.05]

    top_k_method = ensemble.get_top_k(dist, k=3)
    top_k_func = get_top_k_predictions(dist, k=3)
    assert top_k_method == top_k_func

    unc_method = ensemble.compute_uncertainty(dist)
    unc_func = compute_prediction_uncertainty(dist)
    assert unc_method == unc_func