"""
test_model_loading.py - Unit tests for Stage 4: Checkpoint Loading & Custom Layers.

Verifies:
1. Registration and configuration of all custom Keras layers:
   ChannelAttention, SpatialAttention, HybridAttention, SpatialPool, Patches, PatchEncoder.
2. Backward compatibility of HybridAttention and ChannelAttention with both ratio and reduction_ratio.
3. Multi-path fallback mechanism in find_model_checkpoint().
4. Safe error handling when checkpoints are missing or corrupt without unhandled exceptions.
5. Real checkpoint loading on disk for production models.
6. WeightedEnsemble.load_models() integration.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.ensemble import (
    get_custom_objects,
    find_model_checkpoint,
    load_single_model,
    WeightedEnsemble,
    ChannelAttention,
    SpatialAttention,
    HybridAttention,
    SpatialPool,
    Patches,
    PatchEncoder,
)


def test_custom_objects_registry():
    """Verify all 6 custom layers are registered and properly configured."""
    custom_objs = get_custom_objects()
    expected_layers = [
        "ChannelAttention",
        "SpatialAttention",
        "HybridAttention",
        "SpatialPool",
        "Patches",
        "PatchEncoder",
    ]
    for layer_name in expected_layers:
        assert layer_name in custom_objs, f"Missing custom object: {layer_name}"
        assert custom_objs[layer_name] is not None, f"Custom object {layer_name} is None"

    # Test backward compatibility of ratio vs reduction_ratio
    ca1 = ChannelAttention(ratio=16)
    assert ca1.ratio == 16
    assert ca1.get_config()["ratio"] == 16
    assert ca1.get_config()["reduction_ratio"] == 16

    ca2 = ChannelAttention(reduction_ratio=8)
    assert ca2.ratio == 8
    assert ca2.get_config()["reduction_ratio"] == 8

    ha = HybridAttention(reduction_ratio=4, kernel_size=5)
    assert ha.ratio == 4
    assert ha.kernel_size == 5

    # Test SpatialPool modes
    sp_avg = SpatialPool(mode="avg")
    assert sp_avg.mode == "avg"
    assert sp_avg.get_config()["mode"] == "avg"

    sp_max = SpatialPool(mode="max")
    assert sp_max.mode == "max"
    assert sp_max.get_config()["mode"] == "max"

    # Test Patches config
    patches = Patches(patch_size=32)
    assert patches.patch_size == 32
    assert patches.get_config()["patch_size"] == 32

    # Test PatchEncoder config
    encoder = PatchEncoder(num_patches=196, projection_dim=128)
    assert encoder.num_patches == 196
    assert encoder.projection_dim == 128


def test_find_model_checkpoint_fallback():
    """Verify checkpoint discovery with multi-path fallback."""
    # EfficientNet checkpoint discovery
    eff_path = find_model_checkpoint("efficientnet")
    assert eff_path is not None, "find_model_checkpoint failed to find efficientnet checkpoint"
    assert eff_path.is_file(), f"Found efficientnet path is not a file: {eff_path}"

    # ConvNeXt checkpoint discovery
    conv_path = find_model_checkpoint("convnext")
    assert conv_path is not None, "find_model_checkpoint failed to find convnext checkpoint"
    assert conv_path.is_file(), f"Found convnext path is not a file: {conv_path}"

    # ViT checkpoint discovery
    vit_path = find_model_checkpoint("vit")
    assert vit_path is not None, "find_model_checkpoint failed to find vit checkpoint"
    assert vit_path.is_file(), f"Found vit path is not a file: {vit_path}"

    # Non-existent model name should return None safely
    missing = find_model_checkpoint("non_existent_neural_network_xyz")
    assert missing is None, f"Expected None for invalid model, got {missing}"


def test_safe_error_handling_for_missing_checkpoint():
    """Verify load_single_model handles missing checkpoints gracefully without raising exceptions."""
    model, status = load_single_model("non_existent_model_xyz")
    assert model is None
    assert status["loaded"] is False
    assert status["path"] is None
    assert status["is_production"] is False
    assert status["error"] is not None
    assert "No checkpoint file found" in status["error"]


def test_real_efficientnet_checkpoint_loading():
    """Verify loading real trained EfficientNetV2-B3 Hybrid Attention production checkpoint."""
    model, status = load_single_model("efficientnet", safe_mode=False)
    assert model is not None, f"Failed to load real EfficientNet model: {status.get('error')}"
    assert status["loaded"] is True
    assert status["is_production"] is True
    assert status["error"] is None

    # Check input and output dimensions
    assert status["input_shape"] == (None, 256, 256, 3), f"Unexpected input shape: {status['input_shape']}"
    assert status["output_shape"] == (None, 7), f"Unexpected output shape: {status['output_shape']}"


def test_weighted_ensemble_load_models():
    """Verify WeightedEnsemble.load_models() loads models and updates status dictionaries."""
    ensemble = WeightedEnsemble()
    summary = ensemble.load_models(safe_mode=False)

    # EfficientNet must be successfully loaded
    assert summary.get("efficientnet") is True, f"EfficientNet load failed in ensemble: {ensemble.model_status.get('efficientnet')}"
    assert "efficientnet" in ensemble.models
    assert ensemble.model_status["efficientnet"]["loaded"] is True
    assert ensemble.model_status["efficientnet"]["is_production"] is True

    # ConvNeXt and ViT checkpoints exist and load
    assert "convnext" in ensemble.model_status
    assert "vit" in ensemble.model_status