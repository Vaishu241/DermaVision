"""
test_ensemble.py - Sanity and unit tests for WeightedEnsemble.
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.ensemble import WeightedEnsemble, MultimodalEnsemble, preprocess_input_image


def test_ensemble_initialization_and_normalization():
    ens = WeightedEnsemble({"m1": 2.0, "m2": 2.0})
    assert abs(ens.weights["m1"] - 0.5) < 1e-5
    assert abs(ens.weights["m2"] - 0.5) < 1e-5
    assert abs(sum(ens.weights.values()) - 1.0) < 1e-5


def test_ensemble_probability_averaging():
    weights = {"m1": 0.6, "m2": 0.4}
    ens = WeightedEnsemble(weights)
    
    p1 = np.zeros((2, 7))
    p1[:, 0] = 1.0
    p2 = np.zeros((2, 7))
    p2[:, 1] = 1.0
    
    pred_dict = {"m1": p1, "m2": p2}
    ens_probs = ens.predict_probabilities(pred_dict)
    
    assert ens_probs.shape == (2, 7)
    assert np.allclose(ens_probs[:, 0], 0.6)
    assert np.allclose(ens_probs[:, 1], 0.4)
    assert np.allclose(np.sum(ens_probs, axis=-1), 1.0)


def test_ensemble_predict_classes():
    ens = WeightedEnsemble({"m1": 0.7, "m2": 0.3})
    p1 = np.array([[0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0]])
    p2 = np.array([[0.2, 0.6, 0.2, 0.0, 0.0, 0.0, 0.0]])
    
    pred_classes, confidences = ens.predict_classes({"m1": p1, "m2": p2})
    assert pred_classes[0] == 1
    expected_conf = 0.7 * 0.8 + 0.3 * 0.6
    assert abs(confidences[0] - expected_conf) < 1e-5


def test_preprocess_input_image():
    dummy_img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    batch_tensor, orig_uint8 = preprocess_input_image(dummy_img, target_size=(256, 256))
    
    assert batch_tensor.shape == (1, 256, 256, 3)
    assert batch_tensor.dtype == np.float32
    assert batch_tensor.min() >= 0.0 and batch_tensor.max() <= 1.0
    assert orig_uint8.shape == (256, 256, 3)
    assert orig_uint8.dtype == np.uint8


def test_weight_optimization():
    ens = WeightedEnsemble()
    y_true = np.array([0, 1, 2, 3, 4, 5, 6])
    p1 = np.eye(7)
    p2 = np.ones((7, 7)) / 7.0
    
    opt_weights = ens.optimize_weights_on_validation({"good": p1, "bad": p2}, y_true)
    assert opt_weights["good"] > opt_weights["bad"]
    assert abs(sum(opt_weights.values()) - 1.0) < 1e-4


def test_multimodal_metadata_encoding():
    me = MultimodalEnsemble()
    meta_vec = me.encode_metadata(age=55.0, sex="female", localization="back")
    assert meta_vec.shape == (1, 19)
    assert meta_vec.dtype == np.float32
    # Check sex one-hot (male, female, unknown)
    # Female should be at index 1+1 = 2
    assert meta_vec[0, 2] == 1.0
    # Check loc one-hot (back is index 0 in LOC_CATEGORIES)
    # Index 1 (age) + 3 (sex) + 0 = 4
    assert meta_vec[0, 4] == 1.0


def test_multimodal_pipeline_availability_and_inference():
    me = MultimodalEnsemble()
    if me.is_available():
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        res = me.predict_image(dummy_img, age=50.0, sex="unknown", localization="trunk")
        assert "class_name" in res
        assert "confidence" in res
        assert res["class_name"] in config.CLASS_NAMES
        assert res["ensemble_probabilities"].shape == (7,)
        assert abs(np.sum(res["ensemble_probabilities"]) - 1.0) < 1e-4

