"""
test_multimodal.py - Unit and Integration Tests for Multimodal Clinical Decision Support.
Verifies demographic prior estimation, Bayesian log-odds fusion, and BCC telangiectasia resolution.
"""

import os
from pathlib import Path
import numpy as np
import pytest

import config
from src.multimodal import (
    LOCALIZATION_CATEGORIES,
    LOCALIZATION_DISPLAY_MAP,
    SUN_EXPOSURE_TIERS,
    ClinicalPriorModel,
    get_clinical_prior_model,
    fuse_image_and_demographics,
)
from src.ensemble import WeightedEnsemble


def test_localization_categories_contract():
    """Verify all 15 localization categories are registered with display mappings."""
    assert len(LOCALIZATION_CATEGORIES) == 15
    for loc in LOCALIZATION_CATEGORIES:
        assert loc in LOCALIZATION_DISPLAY_MAP
        assert loc in SUN_EXPOSURE_TIERS
        assert len(LOCALIZATION_DISPLAY_MAP[loc]) > 0


def test_clinical_prior_model_probabilities():
    """Verify demographic prior outputs valid 7-class probability distributions."""
    model = get_clinical_prior_model()
    assert model.is_fitted

    # Test young female lower extremity (common for nv/vasc/df)
    p_young = model.predict_prior_probabilities(age=24, sex="female", localization="lower extremity")
    assert isinstance(p_young, np.ndarray)
    assert p_young.shape == (config.NUM_CLASSES,)
    assert np.isclose(np.sum(p_young), 1.0, atol=1e-5)
    assert np.all(p_young >= 0.0)

    # Test elderly male face (classic sun-damaged bcc/akiec site)
    p_elderly = model.predict_prior_probabilities(age=72, sex="male", localization="face")
    assert np.isclose(np.sum(p_elderly), 1.0, atol=1e-5)
    # Elderly face should have significantly higher malignant/pre-malignant prior than young extremity
    bcc_idx = config.LABEL_TO_IDX["bcc"]
    akiec_idx = config.LABEL_TO_IDX["akiec"]
    assert (p_elderly[bcc_idx] + p_elderly[akiec_idx]) > (p_young[bcc_idx] + p_young[akiec_idx])


def test_bayesian_multimodal_fusion_properties():
    """Verify mathematical properties of multimodal Bayesian fusion."""
    # Synthetic flat image distribution
    flat_img = np.ones(config.NUM_CLASSES) / config.NUM_CLASSES

    fused = fuse_image_and_demographics(
        image_probabilities=flat_img,
        age=65,
        sex="male",
        localization="face",
        prior_strength=1.0,
    )

    assert "multimodal_probabilities" in fused
    assert "top_k" in fused
    assert len(fused["top_k"]) == 3
    assert np.isclose(np.sum(fused["multimodal_vector"]), 1.0, atol=1e-5)
    assert fused["primary_prediction"]["code"] in config.CLASS_NAMES
    assert "clinical_shift_note" in fused


def test_bcc_telangiectasia_differentiation_regression():
    """
    CRITICAL CLINICAL REGRESSION TEST:
    Verify that an arborizing telangiectatic BCC lesion (which image-only predicts as VASC 89%)
    is correctly promoted to Basal Cell Carcinoma (BCC) as Rank #1 under clinical demographics.
    """
    # Exact probabilities produced by pure image model on bcc-images-09.jpg
    p_img = np.array([0.0175, 0.0332, 0.0007, 0.0026, 0.0511, 0.0022, 0.8927])

    # 1. Pure image prediction is vasc
    assert config.CLASS_NAMES[np.argmax(p_img)] == "vasc"

    # 2. Multimodal fusion with clinical profile: Age 68, Male, Face
    fused = fuse_image_and_demographics(
        image_probabilities=p_img,
        age=68,
        sex="male",
        localization="face",
        prior_strength=1.25,
        enable_telangiectasia_heuristics=True,
    )

    # Must resolve vascular confusion and promote BCC to #1
    assert fused["primary_prediction"]["code"] == "bcc", (
        f"Expected primary diagnosis 'bcc', got {fused['primary_prediction']['code']}"
    )
    assert fused["primary_prediction"]["probability"] > 0.70, (
        f"Expected BCC confidence > 70%, got {fused['primary_prediction']['probability']*100:.2f}%"
    )
    assert fused["primary_prediction"]["is_malignant"] is True
    assert fused["applied_telangiectasia_logic"] is True


def test_weighted_ensemble_multimodal_integration():
    """Verify that WeightedEnsemble.predict_image accepts metadata and returns multimodal outputs."""
    ensemble = WeightedEnsemble(weights={"efficientnet": 1.0, "convnext": 0.0, "vit": 0.0})

    # Test on real bcc sample
    sample_path = Path("samples/bcc_telangiectasia_face.jpg")
    if not sample_path.is_file():
        pytest.skip("Sample file not found.")

    res = ensemble.predict_image(
        image_input=str(sample_path),
        model_selection="efficientnet",
        metadata={"age": 68, "sex": "male", "localization": "face"},
    )

    assert "multimodal" in res
    assert res["multimodal"] is not None
    assert res["predicted_class"] == "bcc"
    assert res["top_k"][0]["code"] == "bcc"
    assert "image_probabilities" in res
    # Image alone had vasc
    assert res["image_probabilities"]["vasc"] > 0.50
    # Multimodal corrected to bcc
    assert res["all_probabilities"]["bcc"] > 0.70


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
