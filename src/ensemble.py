"""
ensemble.py - Weighted probability averaging ensemble for DermaVision.

Combines predictions from:
1. EfficientNetV2-B3 with Hybrid Attention
2. ConvNeXt
3. Vision Transformer (ViT)

Formulation:
    P_ensemble(y | x) = \sum_{m \in {eff, conv, vit}} w_m \cdot P_m(y | x)
    subject to \sum w_m = 1, w_m \ge 0
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers
except ImportError:
    tf = None
    layers = None

import config
from src.preprocessing import preprocess_image_for_model


# ---------------------------------------------------------------------------
# Stage 3: Class Label Mapping & Diagnostic Utility Functions
# ---------------------------------------------------------------------------

def map_index_to_class(class_idx: Union[int, np.integer]) -> Dict[str, Any]:
    """
    Deterministically map output neuron index (0..6) to diagnostic class metadata.

    Alphabetical HAM10000 Class Indexing Contract:
        0: akiec - Actinic Keratoses / Intraepithelial Carcinoma (Pre-cancerous)
        1: bcc   - Basal Cell Carcinoma (Malignant)
        2: bkl   - Benign Keratosis-like Lesions (Benign)
        3: df    - Dermatofibroma (Benign)
        4: mel   - Melanoma (Malignant)
        5: nv    - Melanocytic Nevi (Benign)
        6: vasc  - Vascular Lesions (Benign)

    Args:
        class_idx: Integer class index (0 to 6).

    Returns:
        Dict[str, Any] with:
            - 'index': int
            - 'code': str (e.g., 'mel')
            - 'label': str (e.g., 'Melanoma')
            - 'malignancy': str ('Malignant', 'Pre-cancerous', 'Benign')
            - 'is_malignant': bool
    """
    try:
        idx = int(class_idx)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Class index must be an integer, got {type(class_idx)}: {class_idx}") from e

    if idx < 0 or idx >= config.NUM_CLASSES:
        raise IndexError(
            f"Class index {idx} out of bounds for {config.NUM_CLASSES} classes (expected 0..{config.NUM_CLASSES - 1})."
        )

    code = config.IDX_TO_LABEL[idx]
    return {
        "index": idx,
        "code": code,
        "label": config.CLASS_LABELS[code],
        "malignancy": config.MALIGNANCY_STATUS[code],
        "is_malignant": config.IS_MALIGNANT[code],
    }


def get_top_k_predictions(probabilities: Union[np.ndarray, List[float]], k: int = 3) -> List[Dict[str, Any]]:
    """
    Extract top-k ranked predictions with differential diagnosis details.

    Args:
        probabilities: 1D array of shape (7,) or 2D array of shape (1, 7)
                       representing softmax/ensemble probabilities for the 7 HAM10000 classes.
        k: Number of top classes to return (default 3, max 7).

    Returns:
        List of dicts ordered from highest to lowest probability, containing:
            - 'rank': 1..k
            - 'index': class index (0..6)
            - 'code': class code ('mel', 'nv', etc.)
            - 'label': human-readable diagnostic label
            - 'probability': float probability in [0.0, 1.0]
            - 'percentage': formatted percentage string (e.g., "75.14%")
            - 'malignancy': malignancy status
            - 'is_malignant': boolean flag
    """
    probs = np.asarray(probabilities, dtype=np.float64).squeeze()
    if probs.ndim != 1 or len(probs) != config.NUM_CLASSES:
        raise ValueError(f"Expected probability vector of length {config.NUM_CLASSES}, got shape {probs.shape}")

    if not np.all(np.isfinite(probs)):
        raise ValueError("Probabilities contain NaN or Inf values.")

    # Numerical safeguard: clamp negative epsilon and normalize if needed
    probs = np.clip(probs, 0.0, 1.0)
    total = np.sum(probs)
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones_like(probs) / len(probs)

    k = max(1, min(k, config.NUM_CLASSES))
    sorted_indices = np.argsort(probs)[::-1]

    top_k = []
    for rank, idx in enumerate(sorted_indices[:k], start=1):
        meta = map_index_to_class(idx)
        prob_val = float(probs[idx])
        top_k.append({
            "rank": rank,
            "index": meta["index"],
            "code": meta["code"],
            "label": meta["label"],
            "probability": prob_val,
            "percentage": f"{prob_val * 100:.2f}%",
            "malignancy": meta["malignancy"],
            "is_malignant": meta["is_malignant"],
        })

    return top_k


def compute_prediction_uncertainty(probabilities: Union[np.ndarray, List[float]]) -> Dict[str, Any]:
    """
    Compute Shannon entropy, margin of confidence, and uncertainty diagnostics.

    Entropy formula:
        H(p) = - \\sum_{i=1}^{C} p_i \\log_2(p_i)
        H_norm(p) = H(p) / \\log_2(C)  \\in [0, 1]

    Margin:
        Margin = p_top1 - p_top2

    Thresholds:
        - Low uncertainty: Top-1 >= 0.70 AND H_norm <= 0.45 AND Margin >= 0.35
        - Moderate uncertainty: Top-1 >= 0.45 AND H_norm <= 0.75 AND Margin >= 0.15
        - High uncertainty: Top-1 < 0.45 OR H_norm > 0.75 OR Margin < 0.15 (triggers out-of-distribution / review alert)

    Args:
        probabilities: 1D probability array of shape (7,) or (1, 7).

    Returns:
        Dict[str, Any] containing:
            - 'entropy': float (raw Shannon entropy in bits)
            - 'max_entropy': float (theoretical maximum for C classes)
            - 'normalized_entropy': float in [0.0, 1.0]
            - 'confidence': float (top-1 probability)
            - 'margin': float (p_top1 - p_top2)
            - 'is_uncertain': bool (True if high uncertainty / borderline)
            - 'uncertainty_level': str ("Low", "Moderate", "High")
            - 'clinical_recommendation': str
    """
    probs = np.asarray(probabilities, dtype=np.float64).squeeze()
    if probs.ndim != 1 or len(probs) != config.NUM_CLASSES:
        raise ValueError(f"Expected probability vector of length {config.NUM_CLASSES}, got shape {probs.shape}")

    if not np.all(np.isfinite(probs)):
        raise ValueError("Probabilities contain NaN or Inf values.")

    # Safe normalization with epsilon floor
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / np.sum(probs)

    # Shannon entropy in bits (base 2)
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = float(np.log2(len(probs)))
    normalized_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))

    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = float(top1 - top2)

    if top1 >= 0.70 and normalized_entropy <= 0.45 and margin >= 0.35:
        level = "Low"
        is_uncertain = False
        recommendation = "High-confidence prediction consistent with specific diagnostic category."
    elif top1 >= 0.45 and normalized_entropy <= 0.75 and margin >= 0.15:
        level = "Moderate"
        is_uncertain = False
        recommendation = "Moderate confidence; differential diagnosis recommended due to intermediate margin."
    else:
        level = "High"
        is_uncertain = True
        recommendation = "High uncertainty / Potential out-of-distribution lesion. Specialist dermoscopic evaluation and biopsy strongly recommended."

    return {
        "entropy": round(entropy, 4),
        "max_entropy": round(max_entropy, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "confidence": round(top1, 4),
        "margin": round(margin, 4),
        "is_uncertain": is_uncertain,
        "uncertainty_level": level,
        "clinical_recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Stage 4: Custom Attention & Transformer Layers for Deserialization
# ---------------------------------------------------------------------------

if tf is not None and layers is not None:
    @tf.keras.utils.register_keras_serializable(package="DermaVision")
    class ChannelAttention(layers.Layer):
        """
        Channel Attention Module (CBAM).
        Accepts ratio or reduction_ratio for full backward compatibility.
        """
        def __init__(self, ratio: int = 8, reduction_ratio: Optional[int] = None, **kwargs):
            super().__init__(**kwargs)
            self.ratio = reduction_ratio if reduction_ratio is not None else ratio

        def build(self, input_shape):
            channel = input_shape[-1]
            self.shared_dense_one = layers.Dense(
                channel // self.ratio,
                activation="relu",
                kernel_initializer="he_normal",
                use_bias=True,
            )
            self.shared_dense_two = layers.Dense(
                channel,
                kernel_initializer="he_normal",
                use_bias=True,
            )
            self.avg_pool = layers.GlobalAveragePooling2D()
            self.max_pool = layers.GlobalMaxPooling2D()
            super().build(input_shape)

        def call(self, inputs):
            channel = inputs.shape[-1]
            avg = self.shared_dense_two(self.shared_dense_one(layers.Reshape((1, 1, channel))(self.avg_pool(inputs))))
            mx = self.shared_dense_two(self.shared_dense_one(layers.Reshape((1, 1, channel))(self.max_pool(inputs))))
            feat = tf.nn.sigmoid(avg + mx)
            return inputs * feat

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"ratio": self.ratio, "reduction_ratio": self.ratio})
            return cfg

    @tf.keras.utils.register_keras_serializable(package="DermaVision")
    class SpatialAttention(layers.Layer):
        """
        Spatial Attention Module (CBAM).
        """
        def __init__(self, kernel_size: int = 7, **kwargs):
            super().__init__(**kwargs)
            self.kernel_size = kernel_size

        def build(self, input_shape):
            self.conv = layers.Conv2D(
                1,
                (self.kernel_size, self.kernel_size),
                strides=1,
                padding="same",
                activation="sigmoid",
                kernel_initializer="he_normal",
                use_bias=False,
            )
            super().build(input_shape)

        def call(self, inputs):
            avg = tf.reduce_mean(inputs, axis=-1, keepdims=True)
            mx = tf.reduce_max(inputs, axis=-1, keepdims=True)
            concat = tf.concat([avg, mx], axis=-1)
            feat = self.conv(concat)
            return inputs * feat

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"kernel_size": self.kernel_size})
            return cfg

    @tf.keras.utils.register_keras_serializable(package="DermaVision")
    class HybridAttention(layers.Layer):
        """
        Hybrid CBAM module combining Channel and Spatial Attention sequentially.
        """
        def __init__(self, ratio: int = 8, reduction_ratio: Optional[int] = None, kernel_size: int = 7, **kwargs):
            super().__init__(**kwargs)
            self.ratio = reduction_ratio if reduction_ratio is not None else ratio
            self.kernel_size = kernel_size

        def build(self, input_shape):
            self.channel_att = ChannelAttention(ratio=self.ratio)
            self.spatial_att = SpatialAttention(kernel_size=self.kernel_size)
            super().build(input_shape)

        def call(self, inputs):
            x = self.channel_att(inputs)
            return self.spatial_att(x)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"ratio": self.ratio, "reduction_ratio": self.ratio, "kernel_size": self.kernel_size})
            return cfg

    @tf.keras.utils.register_keras_serializable(package="DermaVision")
    class SpatialPool(layers.Layer):
        """
        Custom pooling layer used in EfficientNetV2-B3 Hybrid Attention checkpoint.
        Supports mode='avg' and mode='max' over channels.
        """
        def __init__(self, mode: str = "avg", **kwargs):
            super().__init__(**kwargs)
            self.mode = mode

        def call(self, inputs):
            if self.mode == "max":
                return tf.reduce_max(inputs, axis=-1, keepdims=True)
            return tf.reduce_mean(inputs, axis=-1, keepdims=True)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"mode": self.mode})
            return cfg

    @tf.keras.utils.register_keras_serializable(package="DermaVision")
    class Patches(layers.Layer):
        """
        Custom patch extraction layer for Vision Transformer (ViT).
        """
        def __init__(self, patch_size: int = 16, **kwargs):
            super().__init__(**kwargs)
            self.patch_size = patch_size

        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            return tf.reshape(patches, [batch_size, -1, patch_dims])

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"patch_size": self.patch_size})
            return cfg

    @tf.keras.utils.register_keras_serializable(package="DermaVision")
    class PatchEncoder(layers.Layer):
        """
        Projection and learnable positional embedding layer for ViT.
        """
        def __init__(self, num_patches: int = 256, projection_dim: int = 64, **kwargs):
            super().__init__(**kwargs)
            self.num_patches = num_patches
            self.projection_dim = projection_dim
            self.projection = layers.Dense(units=projection_dim)
            self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

        def build(self, input_shape):
            self.projection.build(input_shape)
            self.position_embedding.build((self.num_patches,))
            super().build(input_shape)

        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            return self.projection(patch) + self.position_embedding(positions)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
            return cfg
else:
    ChannelAttention = None
    SpatialAttention = None
    HybridAttention = None
    SpatialPool = None
    Patches = None
    PatchEncoder = None


def get_custom_objects() -> Dict[str, Any]:
    """Return dictionary of all custom Keras layers required for checkpoint deserialization."""
    if tf is None:
        return {}
    return {
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
        "HybridAttention": HybridAttention,
        "SpatialPool": SpatialPool,
        "Patches": Patches,
        "PatchEncoder": PatchEncoder,
    }


def find_model_checkpoint(model_name: str, custom_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Search for trained model checkpoint with multi-path fallback.

    Priority Order:
    1. custom_path (if provided and exists)
    2. config.MODEL_PATHS[model_name]
    3. config.MODELS_DIR / f"{model_name}_best.keras"
    4. Model-specific filenames in config.MODELS_DIR
    5. Fallback directory C:/Users/muvva/Downloads/DermaVision/models
    6. Smoke test paths
    """
    if custom_path is not None:
        p = Path(custom_path)
        if p.is_file():
            return p

    candidates: List[Path] = []
    models_dir = getattr(config, "MODELS_DIR", Path(__file__).resolve().parent.parent / "models")
    fallback_dir = Path("C:/Users/muvva/Downloads/DermaVision/models")

    # Direct config path
    if hasattr(config, "MODEL_PATHS") and model_name in config.MODEL_PATHS:
        candidates.append(Path(config.MODEL_PATHS[model_name]))

    # Standard model names
    candidates.append(models_dir / f"{model_name}_best.keras")
    candidates.append(fallback_dir / f"{model_name}_best.keras")

    # Name-specific variations
    name_clean = model_name.lower().replace("-", "_").replace(" ", "_")
    if any(k in name_clean for k in ["efficient", "eff", "hybrid", "b3"]):
        candidates.extend([
            models_dir / "efficientnetv2_hybrid_attention.keras",
            models_dir / "dermavision_b3_final_best.keras",
            fallback_dir / "efficientnetv2_hybrid_attention.keras",
            fallback_dir / "dermavision_b3_final_best.keras",
        ])
    elif "convnext" in name_clean:
        candidates.extend([
            models_dir / "convnext_best.keras",
            fallback_dir / "convnext_best.keras",
        ])
    elif "vit" in name_clean or "transformer" in name_clean:
        candidates.extend([
            models_dir / "vit_best.keras",
            fallback_dir / "vit_best.keras",
        ])

    # Smoke test candidates
    if hasattr(config, "SMOKE_TEST_MODEL_PATHS") and model_name in config.SMOKE_TEST_MODEL_PATHS:
        candidates.append(Path(config.SMOKE_TEST_MODEL_PATHS[model_name]))
    candidates.append(models_dir / "smoke_test" / f"smoke_{model_name}.keras")
    candidates.append(fallback_dir / "smoke_test" / f"smoke_{model_name}.keras")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def load_single_model(model_name: str,
                      model_path: Optional[Union[str, Path]] = None,
                      custom_objects: Optional[Dict[str, Any]] = None,
                      safe_mode: bool = False,
                      compile_model: bool = False) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Safely load a single trained model checkpoint without crashing other components.

    Returns:
        Tuple[model, status_dict]
    """
    if tf is None:
        return None, {
            "loaded": False,
            "model_name": model_name,
            "path": None,
            "error": "TensorFlow is not installed in the active environment.",
            "is_production": False,
        }

    resolved_path = find_model_checkpoint(model_name, model_path)
    if resolved_path is None:
        return None, {
            "loaded": False,
            "model_name": model_name,
            "path": None,
            "error": f"No checkpoint file found for model '{model_name}'.",
            "is_production": False,
        }

    objs = get_custom_objects()
    if custom_objects:
        objs.update(custom_objects)

    is_smoke = "smoke" in resolved_path.stem.lower() or "smoke_test" in str(resolved_path).lower()

    try:
        model = tf.keras.models.load_model(
            str(resolved_path),
            custom_objects=objs,
            safe_mode=safe_mode,
            compile=compile_model,
        )
        return model, {
            "loaded": True,
            "model_name": model_name,
            "path": str(resolved_path),
            "input_shape": getattr(model, "input_shape", None),
            "output_shape": getattr(model, "output_shape", None),
            "is_production": not is_smoke,
            "error": None,
        }
    except Exception as e:
        return None, {
            "loaded": False,
            "model_name": model_name,
            "path": str(resolved_path),
            "error": str(e),
            "is_production": False,
        }


class WeightedEnsemble:
    """
    Weighted Probability Averaging Ensemble for multiclass skin cancer diagnosis.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the ensemble with model weights.

        Args:
            weights: Dictionary with model names as keys and positive floats as values.
                     Defaults to config.ENSEMBLE_WEIGHTS.
        """
        self.weights = weights if weights is not None else config.ENSEMBLE_WEIGHTS.copy()
        self._normalize_weights()
        self.models: Dict[str, Any] = {}
        self.model_status: Dict[str, Dict[str, Any]] = {}

    def _normalize_weights(self):
        """Ensure all weights sum strictly to 1.0."""
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Total ensemble weights must be positive.")
        self.weights = {k: v / total for k, v in self.weights.items()}

    def load_models(self,
                    model_paths: Optional[Dict[str, Union[str, Path]]] = None,
                    safe_mode: bool = False) -> Dict[str, bool]:
        """
        Load trained model checkpoints into self.models dictionary.
        Supports fallback paths, custom objects, safe error handling without crashing.

        Args:
            model_paths: Optional mapping of model names to file paths.
            safe_mode: Passed to load_model (default False for Lambda deserialization).

        Returns:
            Dict[str, bool]: Summary of load success per model.
        """
        target_models = list(self.weights.keys())
        for m in ["efficientnet", "convnext", "vit"]:
            if m not in target_models:
                target_models.append(m)

        summary = {}
        for name in target_models:
            custom_path = model_paths.get(name) if model_paths else None
            model, status = load_single_model(name, custom_path, safe_mode=safe_mode)
            self.model_status[name] = status
            if model is not None:
                self.models[name] = model
                summary[name] = True
            else:
                summary[name] = False

        return summary

    def predict_probabilities(self, predictions_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate weighted average of prediction probabilities.

        Args:
            predictions_dict: Dictionary mapping model names to predicted probability arrays.
                              Each array has shape (N, num_classes).

        Returns:
            np.ndarray: Ensemble probability array of shape (N, num_classes).
        """
        ensemble_probs = None

        for model_name, weight in self.weights.items():
            if model_name not in predictions_dict:
                raise KeyError(f"Missing predictions for model '{model_name}'.")

            probs = np.array(predictions_dict[model_name])
            weighted_probs = weight * probs

            if ensemble_probs is None:
                ensemble_probs = np.zeros_like(weighted_probs)

            ensemble_probs += weighted_probs

        return ensemble_probs

    def predict_classes(self, predictions_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class indices and associated confidence scores.

        Args:
            predictions_dict: Dictionary mapping model names to probability matrices.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (predicted_class_indices, confidence_scores)
        """
        ensemble_probs = self.predict_probabilities(predictions_dict)
        pred_classes = np.argmax(ensemble_probs, axis=-1)
        confidences = np.max(ensemble_probs, axis=-1)
        return pred_classes, confidences

    def get_top_k(self, probabilities: Union[np.ndarray, List[float]], k: int = 3) -> List[Dict[str, Any]]:
        """Convenience method to retrieve top-k predictions."""
        return get_top_k_predictions(probabilities, k=k)

    def compute_uncertainty(self, probabilities: Union[np.ndarray, List[float]]) -> Dict[str, Any]:
        """Convenience method to compute uncertainty metrics."""
        return compute_prediction_uncertainty(probabilities)

    def predict_image(self,
                      image_input: Union[str, Path, np.ndarray, Any],
                      model_selection: str = "ensemble",
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run end-to-end inference on a single dermoscopic lesion image.

        Args:
            image_input: Filepath (str/Path), numpy array, or PIL Image.
            model_selection: 'ensemble' (default) or individual backbone name
                             ('efficientnet', 'convnext', 'vit').

        Returns:
            Dict[str, Any] containing:
                - 'top_k': List of top-3 differential diagnosis records
                - 'all_probabilities': Dict mapping all 7 class codes to probabilities
                - 'probabilities_vector': List of float probabilities [0..6]
                - 'predicted_class': Class code of top prediction (e.g. 'mel')
                - 'predicted_label': Human-readable label (e.g. 'Melanoma')
                - 'confidence': Probability of top prediction
                - 'uncertainty': Full uncertainty diagnostics dict
                - 'model_used': Architecture key used
                - 'active_weights': Dictionary of normalized weights applied
                - 'preprocessed_image': (H, W, 3) uint8 image for Grad-CAM / UI display
        """
        # Ensure models are loaded
        if len(self.models) == 0:
            self.load_models(safe_mode=False)

        sel_lower = model_selection.lower().strip()
        is_ensemble = any(k in sel_lower for k in ["ensemble", "weighted", "all"])

        predictions_dict = {}
        active_weights = {}
        orig_img = None

        if is_ensemble:
            for m_name, w in self.weights.items():
                if w > 0:
                    if m_name not in self.models:
                        m, _ = load_single_model(m_name, safe_mode=False)
                        if m is not None:
                            self.models[m_name] = m
                    if m_name in self.models:
                        batch_tensor, orig_img = preprocess_image_for_model(image_input, model_name=m_name)
                        preds = self.models[m_name].predict(batch_tensor, verbose=0)
                        predictions_dict[m_name] = preds
                        active_weights[m_name] = w

            # If no model with w > 0 loaded, fallback to any available production model
            if len(predictions_dict) == 0:
                for fallback_name in ["efficientnet", "convnext", "vit"]:
                    if fallback_name not in self.models:
                        m, _ = load_single_model(fallback_name, safe_mode=False)
                        if m is not None:
                            self.models[fallback_name] = m
                    if fallback_name in self.models:
                        batch_tensor, orig_img = preprocess_image_for_model(image_input, model_name=fallback_name)
                        predictions_dict[fallback_name] = self.models[fallback_name].predict(batch_tensor, verbose=0)
                        active_weights[fallback_name] = 1.0
                        break

            if len(predictions_dict) == 0:
                raise RuntimeError("No model checkpoints could be loaded for inference.")

            # Normalize active weights
            tot_w = sum(active_weights.values())
            norm_active_weights = {k: v / tot_w for k, v in active_weights.items()}

            probs = np.zeros(config.NUM_CLASSES, dtype=np.float64)
            for m_name, weight in norm_active_weights.items():
                probs += weight * np.asarray(predictions_dict[m_name], dtype=np.float64).squeeze()

        else:
            # Individual model selection
            target_key = "efficientnet"
            if "conv" in sel_lower:
                target_key = "convnext"
            elif "vit" in sel_lower or "trans" in sel_lower:
                target_key = "vit"

            if target_key not in self.models:
                m, _ = load_single_model(target_key, safe_mode=False)
                if m is not None:
                    self.models[target_key] = m
                else:
                    raise RuntimeError(f"Requested model '{target_key}' could not be loaded.")

            batch_tensor, orig_img = preprocess_image_for_model(image_input, model_name=target_key)
            preds = self.models[target_key].predict(batch_tensor, verbose=0)
            probs = np.asarray(preds, dtype=np.float64).squeeze()
            norm_active_weights = {target_key: 1.0}

        # Safe normalization of final probabilities
        probs = np.clip(probs, 0.0, 1.0)
        s = np.sum(probs)
        if s > 0:
            probs = probs / s
        else:
            probs = np.ones_like(probs) / len(probs)

        image_probs = probs.copy()
        multimodal_data = None
        if metadata is not None and isinstance(metadata, dict):
            try:
                from src.multimodal import fuse_image_and_demographics
                age = float(metadata.get("age", 65))
                sex = str(metadata.get("sex", "male"))
                loc = str(metadata.get("localization", "face"))
                prior_strength = float(metadata.get("prior_strength", 1.25))
                multimodal_data = fuse_image_and_demographics(
                    image_probabilities=image_probs,
                    age=age,
                    sex=sex,
                    localization=loc,
                    prior_strength=prior_strength,
                )
                top_k = multimodal_data["top_k"]
                all_probs = multimodal_data["multimodal_probabilities"]
                probs = multimodal_data["multimodal_vector"]
                uncertainty = multimodal_data["uncertainty"]
            except Exception as e:
                top_k = get_top_k_predictions(probs, k=3)
                all_probs = {code: float(probs[idx]) for idx, code in enumerate(config.CLASS_NAMES)}
                uncertainty = compute_prediction_uncertainty(probs)
        else:
            top_k = get_top_k_predictions(probs, k=3)
            all_probs = {code: float(probs[idx]) for idx, code in enumerate(config.CLASS_NAMES)}
            uncertainty = compute_prediction_uncertainty(probs)

        return {
            "top_k": top_k,
            "all_probabilities": all_probs,
            "probabilities_vector": [float(p) for p in probs],
            "predicted_class": top_k[0]["code"],
            "predicted_label": top_k[0]["label"],
            "confidence": float(top_k[0]["probability"]),
            "uncertainty": uncertainty,
            "model_used": model_selection,
            "active_weights": norm_active_weights,
            "preprocessed_image": orig_img,
            "image_probabilities": {code: float(image_probs[idx]) for idx, code in enumerate(config.CLASS_NAMES)},
            "multimodal": multimodal_data,
        }

    def predict_dataset(self,
                        image_inputs: Union[List[Union[str, Path]], np.ndarray],
                        model_selection: str = "ensemble",
                        batch_size: int = 32) -> np.ndarray:
        """
        Batch prediction over a list of image paths or a 4D image array.

        Returns:
            np.ndarray: (N, 7) probability matrix.
        """
        results = []
        for item in image_inputs:
            res = self.predict_image(item, model_selection=model_selection)
            results.append(res["probabilities_vector"])
        return np.array(results, dtype=np.float32)

    def optimize_weights_on_validation(self,
                                       val_predictions: Dict[str, np.ndarray],
                                       val_ground_truth: np.ndarray,
                                       metric: str = "log_loss") -> Dict[str, float]:
        """
        Find optimal ensemble weights using validation set predictions.
        Solves constrained optimization:
            min_{w} Loss(\\sum w_m * P_m, y_true)
            subject to \\sum w_m = 1, w_m >= 0

        Args:
            val_predictions: Mapping of model names to (N, 7) probability matrices.
            val_ground_truth: (N,) integer class indices or (N, 7) one-hot matrix.
            metric: 'log_loss' (default) or 'f1_macro'.

        Returns:
            Dict[str, float]: Optimized model weights.
        """
        from scipy.optimize import minimize

        model_names = sorted(list(val_predictions.keys()))
        M = len(model_names)
        if M == 0:
            raise ValueError("val_predictions dictionary is empty.")
        if M == 1:
            self.weights = {model_names[0]: 1.0}
            return self.weights.copy()

        N = len(val_predictions[model_names[0]])
        for name in model_names:
            if len(val_predictions[name]) != N:
                raise ValueError(f"Mismatched sample counts between models: {N} vs {len(val_predictions[name])}")

        gt = np.asarray(val_ground_truth)
        if gt.ndim == 1:
            gt_one_hot = np.zeros((N, config.NUM_CLASSES), dtype=np.float64)
            for i, c in enumerate(gt):
                gt_one_hot[i, int(c)] = 1.0
            gt_int = gt.astype(int)
        elif gt.ndim == 2:
            gt_one_hot = gt.astype(np.float64)
            gt_int = np.argmax(gt, axis=-1)
        else:
            raise ValueError(f"Unsupported ground truth shape: {gt.shape}")

        preds_stack = np.stack([np.asarray(val_predictions[name], dtype=np.float64) for name in model_names], axis=0)

        def objective(w):
            w_norm = np.maximum(w, 0.0)
            s = np.sum(w_norm)
            if s <= 0:
                return 1e9
            w_norm = w_norm / s

            blended = np.tensordot(w_norm, preds_stack, axes=(0, 0))
            blended = np.clip(blended, 1e-12, 1.0)
            blended = blended / np.sum(blended, axis=-1, keepdims=True)

            if metric == "f1_macro":
                preds = np.argmax(blended, axis=-1)
                f1_scores = []
                for c in range(config.NUM_CLASSES):
                    tp = np.sum((preds == c) & (gt_int == c))
                    fp = np.sum((preds == c) & (gt_int != c))
                    fn = np.sum((preds != c) & (gt_int == c))
                    denom = 2 * tp + fp + fn
                    f1_c = (2.0 * tp / denom) if denom > 0 else 0.0
                    f1_scores.append(f1_c)
                macro_f1 = float(np.mean(f1_scores))
                # Smooth log-loss tie breaker breaks discrete F1 plateaus
                log_loss = -np.mean(np.sum(gt_one_hot * np.log(blended), axis=-1))
                return -macro_f1 + 1e-3 * float(log_loss)
            else:
                loss = -np.mean(np.sum(gt_one_hot * np.log(blended), axis=-1))
                return float(loss)

        w0 = np.ones(M, dtype=np.float64) / M
        bounds = [(0.0, 1.0) for _ in range(M)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if not res.success:
            res = minimize(objective, w0, method="Nelder-Mead")

        opt_w = np.maximum(res.x, 0.0)
        opt_w = opt_w / np.sum(opt_w)

        self.weights = {name: float(round(opt_w[i], 4)) for i, name in enumerate(model_names)}
        self._normalize_weights()
        return self.weights.copy()

    optimize_weights = optimize_weights_on_validation


# Global default ensemble instance for convenient single-call predictions
_default_ensemble: Optional[WeightedEnsemble] = None


def get_default_ensemble() -> WeightedEnsemble:
    """Retrieve or lazily initialize the singleton WeightedEnsemble instance."""
    global _default_ensemble
    if _default_ensemble is None:
        _default_ensemble = WeightedEnsemble()
    return _default_ensemble


def predict_image(image_input: Union[str, Path, np.ndarray, Any],
                  model_selection: str = "ensemble") -> Dict[str, Any]:
    """Module-level convenience function for end-to-end image inference."""
    ensemble = get_default_ensemble()
    return ensemble.predict_image(image_input, model_selection=model_selection)


def preprocess_input_image(
    image: Any,
    target_size: Tuple[int, int] = config.IMAGE_SIZE,
    scale_mode: str = "normalized_01",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize an RGB image and scale pixels for inference / tests.
    Returns (batch[1, H, W, 3], resized_uint8).
    """
    from PIL import Image
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
    else:
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
            raise ValueError(f"Expected HxWx3 image array, got shape {rgb.shape}")
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb if rgb.max() > 1.5 else rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    width, height = int(target_size[0]), int(target_size[1])
    resized = np.asarray(
        Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    as_float = resized.astype(np.float32)
    if scale_mode == "normalized_01":
        tensor = as_float / 255.0
    elif scale_mode == "minus_one_one":
        tensor = (as_float / 255.0) * 2.0 - 1.0
    elif scale_mode == "raw_255":
        tensor = as_float
    else:
        tensor = as_float / 255.0

    return tensor[None, ...], resized


LOC_CATEGORIES = [
    "back",
    "lower extremity",
    "trunk",
    "upper extremity",
    "abdomen",
    "face",
    "chest",
    "foot",
    "unknown",
    "neck",
    "scalp",
    "hand",
    "ear",
    "genital",
    "acral"
]


class MultimodalEnsemble:
    """
    Multimodal clinical decision ensemble integrating deep features
    and patient metadata (age, sex, localization).
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else config.MODELS_DIR / "mlp_multimodal_best.keras"
        self.model = None
        self._load_model_if_available()

    def _load_model_if_available(self):
        if self.model_path.is_file():
            try:
                import keras
                self.model = keras.models.load_model(str(self.model_path), compile=False)
            except Exception:
                self.model = None

    def is_available(self) -> bool:
        return self.model is not None or self.model_path.is_file()

    def encode_metadata(self, age: float = 50.0, sex: str = "unknown", localization: str = "trunk") -> np.ndarray:
        """
        Encode age (1), sex (3: male, female, unknown), and localization (15) into (1, 19) float32 vector.
        Index contract:
          - 0: age scaled in [0, 1]
          - 1: male, 2: female, 3: unknown
          - 4..18: 15 localization categories (0=back -> 4)
        """
        vec = np.zeros((1, 19), dtype=np.float32)
        # 1. Normalized age
        age_val = float(age) if age is not None else 50.0
        vec[0, 0] = float(np.clip(age_val, 0.0, 100.0) / 100.0)

        # 2. Sex one-hot: male=1, female=2, unknown=3
        sex_str = str(sex).lower().strip()
        sex_map = {"male": 1, "female": 2, "unknown": 3}
        sex_idx = sex_map.get(sex_str, 3)
        vec[0, sex_idx] = 1.0

        # 3. Localization one-hot: 15 categories (back=0 -> index 4)
        loc_str = str(localization).lower().strip()
        loc_idx = LOC_CATEGORIES.index(loc_str) if loc_str in LOC_CATEGORIES else LOC_CATEGORIES.index("unknown")
        vec[0, 4 + loc_idx] = 1.0

        return vec

    def predict_image(self,
                      image: Any,
                      age: float = 50.0,
                      sex: str = "unknown",
                      localization: str = "trunk") -> Dict[str, Any]:
        """
        Predict on an image with clinical demographic metadata.
        """
        from src.multimodal import fuse_image_and_demographics

        meta_vec = self.encode_metadata(age=age, sex=sex, localization=localization)
        ensemble = get_default_ensemble()
        base_res = ensemble.predict_image(image, model_selection="efficientnet")
        img_probs = base_res["probabilities_vector"]

        fused = fuse_image_and_demographics(
            image_probabilities=img_probs,
            age=age,
            sex=sex,
            localization=localization
        )
        top1 = fused["top_k"][0]
        ens_probs = fused["multimodal_vector"]

        return {
            "class_name": top1["code"],
            "class_label": top1["label"],
            "confidence": float(top1["probability"]),
            "ensemble_probabilities": np.asarray(ens_probs, dtype=np.float32),
            "top_k": fused["top_k"],
            "metadata_vector": meta_vec,
            "multimodal": fused
        }

