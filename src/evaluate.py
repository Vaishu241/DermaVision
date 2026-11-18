"""
evaluate.py - Comprehensive evaluation module for DermaVision.

Calculates:
- Multiclass Accuracy, Precision, Recall, Macro F1, Weighted F1
- Per-class sensitivity/specificity
- Multiclass One-vs-Rest AUC-ROC
- Generates Confusion Matrix and ROC Curve figures
"""

import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
        auc,
        f1_score,
        accuracy_score,
        precision_score,
        recall_score
    )
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend safe for all environments
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    pass

import config
from src.preprocessing import preprocess_image_for_model


def compute_classification_metrics(y_true: np.ndarray,
                                   y_pred_probs: np.ndarray) -> Dict[str, Any]:
    """
    Compute full set of classification metrics for HAM10000 7-class task.

    Args:
        y_true: Ground truth integer class indices of shape (N,) or one-hot (N, 7).
        y_pred_probs: Predicted probability distributions of shape (N, 7).

    Returns:
        Dict[str, Any]: Dictionary containing scalar metrics and classification report.
    """
    if y_true.ndim > 1:
        y_true_indices = np.argmax(y_true, axis=-1)
    else:
        y_true_indices = np.asarray(y_true, dtype=int)

    y_pred_indices = np.argmax(y_pred_probs, axis=-1)

    acc = float(accuracy_score(y_true_indices, y_pred_indices))
    macro_f1 = float(f1_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0))

    try:
        macro_auc = float(roc_auc_score(y_true_indices, y_pred_probs, multi_class="ovr", average="macro"))
    except Exception:
        macro_auc = None

    report = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    mel_dict = compute_melanoma_metrics(y_true_indices, y_pred_probs)
    mel_metrics_compat = {
        "recall_sensitivity": mel_dict["melanoma_sensitivity"],
        "sensitivity": mel_dict["melanoma_sensitivity"],
        "specificity": mel_dict["melanoma_specificity"],
        "precision": mel_dict["melanoma_precision"],
        "f1_score": mel_dict["melanoma_f1"]
    }

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "auc_roc": round(macro_auc, 4) if macro_auc is not None else None,
        "classification_report": report,
        "melanoma_metrics": mel_metrics_compat,
        **mel_dict
    }


def compute_melanoma_metrics(y_true: np.ndarray,
                             y_pred_probs: np.ndarray) -> Dict[str, float]:
    """
    Compute dedicated clinical sensitivity, specificity, and precision for Melanoma (class 'mel', index 4).

    Args:
        y_true: Array of true class indices (N,) or one-hot (N, 7).
        y_pred_probs: Array of predicted probabilities (N, 7).

    Returns:
        Dict[str, float]: Specificity, sensitivity (recall), precision, F1, balanced accuracy.
    """
    if y_true.ndim > 1:
        y_true_indices = np.argmax(y_true, axis=-1)
    else:
        y_true_indices = np.asarray(y_true, dtype=int)

    y_pred_indices = np.argmax(y_pred_probs, axis=-1)
    mel_idx = config.LABEL_TO_IDX.get("mel", 4)

    y_true_binary = (y_true_indices == mel_idx).astype(int)
    y_pred_binary = (y_pred_indices == mel_idx).astype(int)

    tp = int(np.sum((y_pred_binary == 1) & (y_true_binary == 1)))
    fp = int(np.sum((y_pred_binary == 1) & (y_true_binary == 0)))
    fn = int(np.sum((y_pred_binary == 0) & (y_true_binary == 1)))
    tn = int(np.sum((y_pred_binary == 0) & (y_true_binary == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0

    return {
        "melanoma_sensitivity": round(float(sensitivity), 4),
        "melanoma_specificity": round(float(specificity), 4),
        "melanoma_precision": round(float(precision), 4),
        "melanoma_f1": round(float(f1), 4),
        "melanoma_balanced_accuracy": round(float(balanced_acc), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          save_path: Optional[Path] = None,
                          normalize: bool = True):
    """
    Generate and save formatted confusion matrix heatmap for all 7 lesion classes.

    Args:
        y_true: Array of true class indices.
        y_pred: Array of predicted class indices.
        save_path: Path to save figure (e.g. reports/confusion_matrix.png).
        normalize: Whether to normalize matrix by true class support.
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=-1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    if normalize:
        sums = cm.sum(axis=1)[:, np.newaxis]
        sums[sums == 0] = 1.0
        cm_norm = cm.astype("float") / sums
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(config.NUM_CLASSES),
        yticks=np.arange(config.NUM_CLASSES),
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        title="HAM10000 7-Class Confusion Matrix",
        ylabel="Ground Truth Lesion Class",
        xlabel="Predicted Lesion Class"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm_norm.max() / 2.0 if cm_norm.max() > 0 else 0.5
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            val = format(cm_norm[i, j], fmt)
            ax.text(
                j, i, val,
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=10
            )

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_roc_curves(y_true: np.ndarray,
                    y_pred_probs: np.ndarray,
                    save_path: Optional[Path] = None):
    """
    Plot and save multiclass One-vs-Rest ROC curves for all 7 classes.

    Args:
        y_true: Array of true class indices or one-hot.
        y_pred_probs: Array of predicted probabilities (N, 7).
        save_path: Path to save figure.
    """
    if y_true.ndim == 1:
        N = len(y_true)
        y_one_hot = np.zeros((N, config.NUM_CLASSES))
        for i, c in enumerate(y_true):
            y_one_hot[i, int(c)] = 1.0
    else:
        y_one_hot = y_true

    plt.figure(figsize=(9, 7))

    for idx, name in enumerate(config.CLASS_NAMES):
        try:
            fpr, tpr, _ = roc_curve(y_one_hot[:, idx], y_pred_probs[:, idx])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{name.upper()} (AUC = {roc_auc:.3f})")
        except Exception:
            continue

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("Multi-Class One-vs-Rest ROC Curves (HAM10000)", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def evaluate_test_set(test_data: Optional[Union[pd.DataFrame, Path, str]] = None,
                      ensemble_model: Optional[Any] = None,
                      save_reports: bool = True,
                      num_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    End-to-end evaluation pipeline for the test dataset using the trained ensemble.

    Args:
        test_data: Test DataFrame or CSV path. Defaults to config.TEST_CSV_PATH.
        ensemble_model: WeightedEnsemble instance. If None, uses get_default_ensemble().
        save_reports: Whether to save metrics summary JSON and figures to reports/.
        num_samples: Optional limit on test samples (useful for fast verification).

    Returns:
        Dict[str, Any]: Comprehensive metrics dictionary.
    """
    from src.ensemble import get_default_ensemble

    if ensemble_model is None:
        ensemble_model = get_default_ensemble()

    # Load test dataset
    if test_data is None:
        test_path = config.TEST_CSV_PATH
        if not test_path.is_file():
            # Check fallback
            alt = Path("C:/Users/muvva/Downloads/DermaVision/data/processed/test.csv")
            if alt.is_file():
                test_path = alt
            else:
                raise FileNotFoundError(f"Test split CSV not found at {config.TEST_CSV_PATH}")
        df = pd.read_csv(test_path)
    elif isinstance(test_data, (str, Path)):
        df = pd.read_csv(test_data)
    else:
        df = test_data.copy()

    if num_samples is not None and len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=config.RANDOM_SEED)

    valid_records = []
    path_col = "image_path" if "image_path" in df.columns else ("path" if "path" in df.columns else None)
    if path_col is None:
        raise KeyError(f"Expected 'image_path' or 'path' column in test DataFrame, found: {df.columns.tolist()}")

    for _, row in df.iterrows():
        raw_val = row[path_col]
        p = Path(raw_val)
        if not p.is_file():
            # Check candidate directories
            candidates = [
                PROJECT_ROOT / "HAM10000_images_part_1" / p.name,
                PROJECT_ROOT / "HAM10000_images_part_2" / p.name,
                PROJECT_ROOT / "dataset" / "ham10000_images_part_1" / p.name,
                PROJECT_ROOT / "dataset" / "ham10000_images_part_2" / p.name,
                Path("C:/Users/muvva/Downloads/DermaVision") / p.name,
                Path("C:/Users/muvva/Downloads/DermaVision/HAM10000_images_part_1") / p.name,
                Path("C:/Users/muvva/Downloads/DermaVision/HAM10000_images_part_2") / p.name,
                PROJECT_ROOT / "data" / "raw" / "HAM10000_images_part_1" / p.name,
                PROJECT_ROOT / "data" / "raw" / "HAM10000_images_part_2" / p.name,
            ]
            for cand in candidates:
                if cand.is_file():
                    p = cand
                    break

        if p.is_file():
            target_val = row["target"] if "target" in row else config.LABEL_TO_IDX.get(row.get("dx", ""), 0)
            valid_records.append((p, int(target_val)))

    if len(valid_records) == 0:
        raise RuntimeError("No valid test images found on disk to evaluate.")

    image_paths = [r[0] for r in valid_records]
    y_true = np.array([r[1] for r in valid_records], dtype=int)

    # Run predictions
    y_pred_probs = ensemble_model.predict_dataset(image_paths, model_selection="ensemble")
    y_pred = np.argmax(y_pred_probs, axis=-1)

    # Calculate metrics
    class_metrics = compute_classification_metrics(y_true, y_pred_probs)
    mel_metrics = compute_melanoma_metrics(y_true, y_pred_probs)

    summary = {
        "num_evaluated_samples": len(valid_records),
        **class_metrics,
        **mel_metrics,
        "ensemble_weights_used": ensemble_model.weights,
    }

    if save_reports:
        reports_dir = getattr(config, "REPORTS_DIR", Path(__file__).resolve().parent.parent / "reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save metrics JSON
        json_path = reports_dir / "metrics_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 2. Confusion matrix
        cm_path = reports_dir / "confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, save_path=cm_path, normalize=True)

        # 3. ROC curves
        roc_path = reports_dir / "roc_curves.png"
        plot_roc_curves(y_true, y_pred_probs, save_path=roc_path)

        print(f"Evaluation complete. Reports saved to {reports_dir.resolve()}:")
        print(f"  - Metrics JSON: {json_path.name}")
        print(f"  - Confusion Matrix: {cm_path.name}")
        print(f"  - ROC Curves: {roc_path.name}")

    return summary
