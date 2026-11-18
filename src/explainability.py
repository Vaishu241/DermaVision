"""
explainability.py - Explainable AI (XAI) using Grad-CAM for DermaVision.

Provides visual explanations for dermatologists and end-users by highlighting
lesion regions that contributed most strongly to the model's diagnostic classification.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import tensorflow as tf
    import cv2
except ImportError:
    tf = None
    cv2 = None

import config


from typing import Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np

try:
    import tensorflow as tf
    import cv2
except ImportError:
    tf = None
    cv2 = None

import config
from src.preprocessing import preprocess_image_for_model


def find_target_gradcam_layer(model) -> Optional[str]:
    """
    Automatically identify the most informative feature extraction or attention layer for Grad-CAM.
    Prioritizes hybrid attention output, spatial attention conv, or final backbone conv layer.
    """
    layer_names = [l.name for l in model.layers]

    # Priority candidates in EfficientNetV2-B3 Hybrid Attention
    candidates = ["multiply_1", "conv2d", "attended_features", "multiply", "spatial_feature"]
    for c in candidates:
        if c in layer_names:
            return c

    # Search backwards for last 4D output layer before pooling
    for layer in reversed(model.layers):
        if hasattr(layer, "output_shape") and layer.output_shape:
            shape = layer.output_shape
            if len(shape) == 4 and "pool" not in layer.name.lower() and "input" not in layer.name.lower():
                return layer.name

    return None


def make_gradcam_heatmap(img_array: np.ndarray,
                          model,
                          last_conv_layer_name: Optional[str] = None,
                          pred_index: Optional[int] = None) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a given input image and model.

    Args:
        img_array: Input image tensor batch of shape (1, H, W, 3).
        model: Trained Keras model.
        last_conv_layer_name: Name of target convolutional layer. If None, auto-detected.
        pred_index: Target class index to explain. If None, uses top predicted class.

    Returns:
        np.ndarray: 2D normalized heatmap values in [0, 1].
    """
    if tf is None:
        raise ImportError("TensorFlow is required to compute Grad-CAM heatmaps.")

    if last_conv_layer_name is None:
        last_conv_layer_name = find_target_gradcam_layer(model)
        if last_conv_layer_name is None:
            raise ValueError("Could not automatically locate a 4D convolutional/attention layer for Grad-CAM.")

    # 1. Build a model that maps input to target layer activations and output predictions
    try:
        target_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )
    except Exception as e:
        raise ValueError(f"Could not build gradient model with layer '{last_conv_layer_name}': {e}")

    # 2. Compute gradients of top predicted class with respect to layer activations
    tensor_input = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tensor_input)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    # 3. Pool the gradients over spatial dimensions (neuron importance weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 4. Weight the activation map by gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 5. Apply ReLU to retain only positive influence, then normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_gradcam(original_image: np.ndarray,
                     heatmap: np.ndarray,
                     alpha: float = 0.45,
                     colormap: int = None) -> np.ndarray:
    """
    Superimpose Grad-CAM heatmap onto the original RGB dermoscopic lesion image.

    Args:
        original_image: Original RGB image array (H, W, 3) in uint8 [0, 255].
        heatmap: 2D float array in [0, 1].
        alpha: Transparency factor for heatmap overlay (0 = only original, 1 = only heatmap).
        colormap: OpenCV colormap constant (defaults to cv2.COLORMAP_JET).

    Returns:
        np.ndarray: Blended RGB image with superimposed heatmap.
    """
    if cv2 is None:
        raise ImportError("OpenCV is required to generate overlaid heatmaps.")

    if colormap is None:
        colormap = cv2.COLORMAP_JET

    h, w = original_image.shape[:2]
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert heatmap to RGB colormap
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    # Blend original and heatmap
    overlaid = np.uint8(alpha * color_heatmap + (1.0 - alpha) * original_image)
    return overlaid


def generate_gradcam_visualization(image_input: Union[str, Path, np.ndarray, Any],
                                    model=None,
                                    pred_index: Optional[int] = None,
                                    alpha: float = 0.45) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete end-to-end Grad-CAM pipeline.

    Args:
        image_input: Filepath, PIL Image, or uint8 array.
        model: Trained Keras model. If None, retrieves loaded efficientnet model from default ensemble.
        pred_index: Target class index. If None, uses top prediction.
        alpha: Heatmap overlay blending weight.

    Returns:
        Tuple[overlaid_image, heatmap]:
            - overlaid_image: (H, W, 3) uint8 RGB array
            - heatmap: (H, W) float32 array in [0, 1]
    """
    if model is None:
        from src.ensemble import get_default_ensemble
        ens = get_default_ensemble()
        if len(ens.models) == 0:
            ens.load_models()
        model = ens.models.get("efficientnet")
        if model is None:
            raise RuntimeError("EfficientNet model is not loaded for Grad-CAM explanation.")

    batch_tensor, orig_uint8 = preprocess_image_for_model(image_input, model_name="efficientnet")
    heatmap = make_gradcam_heatmap(batch_tensor, model, pred_index=pred_index)
    overlaid = overlay_gradcam(orig_uint8, heatmap, alpha=alpha)
    return overlaid, heatmap
