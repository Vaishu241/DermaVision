"""
test_ensemble_inference.py - Unit tests for Stage 5: Ensemble Inference & Weight Optimization.

Verifies:
1. End-to-end predict_image() pipeline on synthetic and real dermoscopic images.
2. Differential top-k ranking and all 7 class probabilities.
3. Individual backbone prediction dispatch (e.g. model_selection='efficientnet').
4. Constrained validation weight optimization using scipy.optimize.minimize (SLSQP/Nelder-Mead).
5. Protection against degraded models via optimal weight assignment.
6. Batch dataset prediction with predict_dataset().
7. Real Melanoma image verification (ISIC_0026120.jpg).
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.ensemble import (
    WeightedEnsemble,
    predict_image,
    get_default_ensemble,
)


def test_predict_image_synthetic_array():
    """Verify predict_image runs smoothly on a synthetic 256x256x3 image array."""
    synth_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    res = predict_image(synth_img, model_selection="ensemble")

    required_keys = [
        "top_k",
        "all_probabilities",
        "probabilities_vector",
        "predicted_class",
        "predicted_label",
        "confidence",
        "uncertainty",
        "model_used",
        "active_weights",
        "preprocessed_image",
    ]
    for key in required_keys:
        assert key in res, f"Missing key '{key}' in prediction response."

    # Validate probability properties
    probs = res["probabilities_vector"]
    assert len(probs) == 7, f"Expected 7 class probabilities, got {len(probs)}"
    assert abs(sum(probs) - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {sum(probs)}"
    for p in probs:
        assert 0.0 <= p <= 1.0

    # Validate top-3 ranking
    top3 = res["top_k"]
    assert len(top3) == 3
    assert top3[0]["probability"] >= top3[1]["probability"] >= top3[2]["probability"]
    assert res["predicted_class"] == top3[0]["code"]
    assert res["confidence"] == top3[0]["probability"]

    # Validate uncertainty
    assert "entropy" in res["uncertainty"]
    assert "normalized_entropy" in res["uncertainty"]
    assert "is_uncertain" in res["uncertainty"]


def test_individual_model_inference():
    """Verify individual model prediction routing."""
    ensemble = WeightedEnsemble()
    synth_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    res = ensemble.predict_image(synth_img, model_selection="efficientnet")
    assert res["model_used"] == "efficientnet"
    assert res["active_weights"] == {"efficientnet": 1.0}
    assert len(res["probabilities_vector"]) == 7


def test_validation_weight_optimization_log_loss():
    """Verify weight optimization correctly assigns high weights to accurate models and penalizes degraded models."""
    ensemble = WeightedEnsemble()
    N = 100

    np.random.seed(42)
    # 100 random ground truth classes (0..6)
    val_gt = np.random.randint(0, 7, size=N)

    # Model 1 (EfficientNet): High accuracy (~85% on true class)
    eff_preds = np.random.dirichlet(np.ones(7) * 0.1, size=N)
    for i in range(N):
        eff_preds[i, val_gt[i]] += 4.0
    eff_preds = eff_preds / np.sum(eff_preds, axis=-1, keepdims=True)

    # Model 2 (ConvNeXt): Medium accuracy
    conv_preds = np.random.dirichlet(np.ones(7) * 0.5, size=N)
    for i in range(N):
        conv_preds[i, val_gt[i]] += 1.0
    conv_preds = conv_preds / np.sum(conv_preds, axis=-1, keepdims=True)

    # Model 3 (ViT): Degenerate / Uninformative noise
    vit_preds = np.ones((N, 7)) / 7.0

    val_preds = {
        "efficientnet": eff_preds,
        "convnext": conv_preds,
        "vit": vit_preds,
    }

    opt_weights = ensemble.optimize_weights_on_validation(val_preds, val_gt, metric="log_loss")

    # Weights must sum strictly to 1.0
    assert abs(sum(opt_weights.values()) - 1.0) < 1e-4
    for w in opt_weights.values():
        assert w >= 0.0

    # The high accuracy model (efficientnet) must have the highest weight
    assert opt_weights["efficientnet"] > opt_weights["convnext"]
    assert opt_weights["efficientnet"] > opt_weights["vit"]
    assert opt_weights["efficientnet"] > 0.60, f"Expected efficientnet > 0.60, got {opt_weights['efficientnet']}"


def test_validation_weight_optimization_macro_f1():
    """Verify weight optimization with macro F1 metric."""
    ensemble = WeightedEnsemble()
    N = 80
    np.random.seed(123)
    val_gt = np.random.randint(0, 7, size=N)

    eff_preds = np.zeros((N, 7))
    for i in range(N):
        eff_preds[i, val_gt[i]] = 0.90
        eff_preds[i, (val_gt[i] + 1) % 7] = 0.10

    vit_preds = np.ones((N, 7)) / 7.0

    val_preds = {
        "efficientnet": eff_preds,
        "vit": vit_preds,
    }

    opt_weights = ensemble.optimize_weights_on_validation(val_preds, val_gt, metric="f1_macro")
    assert abs(sum(opt_weights.values()) - 1.0) < 1e-4
    assert opt_weights["efficientnet"] > 0.80


def test_predict_dataset_batching():
    """Verify batch inference over multiple samples."""
    ensemble = WeightedEnsemble()
    synth_batch = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(4)]

    matrix = ensemble.predict_dataset(synth_batch, model_selection="ensemble")
    assert matrix.shape == (4, 7), f"Expected shape (4, 7), got {matrix.shape}"
    for row in matrix:
        assert abs(np.sum(row) - 1.0) < 1e-4


def test_real_melanoma_image_prediction_if_available():
    """Verify prediction on real test melanoma sample if HAM10000 dataset is present."""
    test_paths = [
        Path("C:/Users/muvva/Downloads/DermaVision/HAM10000_images_part_1/ISIC_0026120.jpg"),
        Path("C:/Users/muvva/Downloads/DermaVision/HAM10000_images_part_2/ISIC_0026120.jpg"),
    ]
    mel_img_path = None
    for p in test_paths:
        if p.is_file():
            mel_img_path = p
            break

    if mel_img_path is not None:
        res = predict_image(mel_img_path, model_selection="ensemble")
        # Ensure Melanoma is top prediction (addressing the core bug)
        assert res["predicted_class"] == "mel", f"Expected Melanoma (mel), got {res['predicted_class']}"
        assert res["top_k"][0]["is_malignant"] is True
        assert res["confidence"] > 0.35, f"Expected high confidence for melanoma, got {res['confidence']}"

        # BKL must not be rank 1
        assert res["top_k"][0]["code"] != "bkl", "Regression: Melanoma was predicted as BKL!"