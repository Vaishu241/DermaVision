"""
test_evaluate.py - Sanity and unit tests for evaluation metrics and plots.
"""
import sys
import tempfile
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.evaluate import (
    compute_classification_metrics,
    compute_melanoma_metrics,
    plot_confusion_matrix,
    plot_roc_curves
)
from src.explainability import (
    overlay_gradcam,
    generate_gradcam_visualization
)


def test_compute_classification_metrics_perfect():
    # 7 samples, exactly one per class
    y_true = np.array([0, 1, 2, 3, 4, 5, 6])
    y_pred_probs = np.eye(7)
    
    metrics = compute_classification_metrics(y_true, y_pred_probs)
    
    assert abs(metrics["accuracy"] - 1.0) < 1e-5
    assert abs(metrics["macro_f1"] - 1.0) < 1e-5
    assert abs(metrics["weighted_f1"] - 1.0) < 1e-5
    assert metrics["auc_roc"] is not None and abs(metrics["auc_roc"] - 1.0) < 1e-5
    
    # Melanoma (class 4) metrics
    mel = metrics["melanoma_metrics"]
    assert abs(mel["recall_sensitivity"] - 1.0) < 1e-5
    assert abs(mel["specificity"] - 1.0) < 1e-5
    assert abs(mel["f1_score"] - 1.0) < 1e-5


def test_melanoma_specific_metrics():
    # True: 2 melanoma (idx 4), 8 non-melanoma
    y_true = np.array([4, 4, 0, 1, 2, 3, 5, 6, 0, 1])
    # Predicted: 1 true mel detected, 1 missed (predicted 0)
    y_pred_probs = np.zeros((10, 7))
    y_pred_probs[0, 4] = 1.0  # correct mel
    y_pred_probs[1, 0] = 1.0  # missed mel (false neg)
    for i in range(2, 10):
        y_pred_probs[i, y_true[i]] = 1.0  # all others correct
    
    metrics = compute_classification_metrics(y_true, y_pred_probs)
    mel = metrics["melanoma_metrics"]
    
    # Sensitivity = 1/2 = 0.5
    assert abs(mel["recall_sensitivity"] - 0.5) < 1e-5
    # Specificity = 8/8 = 1.0
    assert abs(mel["specificity"] - 1.0) < 1e-5


def test_plot_generation():
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
    y_pred = y_true.copy()
    y_pred_probs = np.eye(7)[y_true]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cm_path = Path(tmpdir) / "cm.png"
        roc_path = Path(tmpdir) / "roc.png"
        
        plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
        assert cm_path.is_file() and cm_path.stat().st_size > 0
        
        plot_roc_curves(y_true, y_pred_probs, save_path=roc_path)
        assert roc_path.is_file() and roc_path.stat().st_size > 0


def test_compute_classification_metrics():
    test_compute_classification_metrics_perfect()


def test_compute_melanoma_metrics():
    y_true = np.array([4, 4, 0, 1, 2, 3, 5, 6, 1, 2])
    y_preds = np.array([4, 4, 4, 1, 2, 3, 5, 6, 1, 2])
    y_pred_probs = np.zeros((10, 7))
    for i, p in enumerate(y_preds):
        y_pred_probs[i, p] = 1.0

    res = compute_melanoma_metrics(y_true, y_pred_probs)
    assert res["tp"] == 2
    assert res["fp"] == 1
    assert res["fn"] == 0
    assert res["tn"] == 7
    assert abs(res["melanoma_sensitivity"] - 1.0) < 1e-4
    assert abs(res["melanoma_precision"] - (2.0 / 3.0)) < 1e-3
    assert abs(res["melanoma_specificity"] - (7.0 / 8.0)) < 1e-3


def test_gradcam_overlay_synthesis():
    orig = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    heatmap = np.random.uniform(0.0, 1.0, (8, 8)).astype(np.float32)

    overlaid = overlay_gradcam(orig, heatmap, alpha=0.4)
    assert overlaid.shape == (256, 256, 3)
    assert overlaid.dtype == np.uint8


def test_real_gradcam_with_model():
    synth_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    overlaid, heatmap = generate_gradcam_visualization(synth_img)
    assert overlaid.shape == (256, 256, 3)
    assert heatmap.ndim == 2

