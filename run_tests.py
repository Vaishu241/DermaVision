"""
run_tests.py - Universal test runner for DermaVision.
Runs all sanity and unit tests without requiring external test dependencies.
"""

import sys
import time
from pathlib import Path

# Ensure root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_config import (
    test_config_classes,
    test_ensemble_weights,
    test_directory_paths
)
from tests.test_preprocessing import (
    test_image_path_matching,
    test_label_and_image_validation,
    test_group_aware_split_no_lesion_leakage,
    test_image_preprocessing_resize_and_normalization,
    test_model_specific_preprocessing_contracts,
    test_class_weights_calculation
)
from tests.test_labels_and_predictions import (
    test_deterministic_index_to_class_mapping,
    test_one_hot_predictions_for_all_7_classes,
    test_top_k_rank_ordering_and_probabilities,
    test_prediction_uncertainty_and_entropy,
    test_melanoma_vs_bkl_differentiation,
    test_weighted_ensemble_methods
)
from tests.test_model_loading import (
    test_custom_objects_registry,
    test_find_model_checkpoint_fallback,
    test_safe_error_handling_for_missing_checkpoint,
    test_real_efficientnet_checkpoint_loading,
    test_weighted_ensemble_load_models
)
from tests.test_ensemble_inference import (
    test_predict_image_synthetic_array,
    test_individual_model_inference,
    test_validation_weight_optimization_log_loss,
    test_validation_weight_optimization_macro_f1,
    test_predict_dataset_batching,
    test_real_melanoma_image_prediction_if_available
)
from tests.test_evaluate import (
    test_compute_classification_metrics,
    test_compute_melanoma_metrics,
    test_plot_generation,
    test_gradcam_overlay_synthesis,
    test_real_gradcam_with_model
)
from tests.test_multimodal import (
    test_localization_categories_contract,
    test_clinical_prior_model_probabilities,
    test_bayesian_multimodal_fusion_properties,
    test_bcc_telangiectasia_differentiation_regression,
    test_weighted_ensemble_multimodal_integration
)


def run_all_tests():
    tests = [
        ("Config: 7 Classes & Mappings", test_config_classes),
        ("Config: Ensemble Weights Sum to 1.0", test_ensemble_weights),
        ("Config: Directory Path Specifications", test_directory_paths),
        ("Preprocessing: Image Path Matching & Duplicate Detection", test_image_path_matching),
        ("Preprocessing: Label & Image Dimension Validation", test_label_and_image_validation),
        ("Preprocessing: Group-Aware Split (Zero Lesion Leakage)", test_group_aware_split_no_lesion_leakage),
        ("Preprocessing: 256x256 RGB float32 Normalization", test_image_preprocessing_resize_and_normalization),
        ("Preprocessing: Model-Specific Scaling & Channel Contracts", test_model_specific_preprocessing_contracts),
        ("Preprocessing: Inverse Class Frequency Weights", test_class_weights_calculation),
        ("Labels: Deterministic Neuron-to-Class 0..6 Contract", test_deterministic_index_to_class_mapping),
        ("Labels: 7-Class One-Hot Prediction Accuracy & Malignancy", test_one_hot_predictions_for_all_7_classes),
        ("Predictions: Top-K Differential Ranking & Probabilities", test_top_k_rank_ordering_and_probabilities),
        ("Uncertainty: Shannon Entropy & OOD Boundary Detection", test_prediction_uncertainty_and_entropy),
        ("Clinical: Melanoma vs BKL Differentiation Regression", test_melanoma_vs_bkl_differentiation),
        ("Ensemble: OOP Helper Methods Consistency", test_weighted_ensemble_methods),
        ("Model Loading: Custom Objects Registry & Layer Configs", test_custom_objects_registry),
        ("Model Loading: Multi-Path Checkpoint Discovery & Fallback", test_find_model_checkpoint_fallback),
        ("Model Loading: Safe Error Handling for Missing Checkpoint", test_safe_error_handling_for_missing_checkpoint),
        ("Model Loading: Real Production Checkpoint Verification", test_real_efficientnet_checkpoint_loading),
        ("Model Loading: WeightedEnsemble Integration & Status", test_weighted_ensemble_load_models),
        ("Ensemble Inference: End-to-End Prediction Pipeline", test_predict_image_synthetic_array),
        ("Ensemble Inference: Individual Model Routing", test_individual_model_inference),
        ("Ensemble Weights: Optimization via Log-Loss Minimization", test_validation_weight_optimization_log_loss),
        ("Ensemble Weights: Optimization via Macro-F1 Maximization", test_validation_weight_optimization_macro_f1),
        ("Ensemble Inference: Batch Dataset Prediction", test_predict_dataset_batching),
        ("Clinical Verification: Real Melanoma Test Lesion Diagnosis", test_real_melanoma_image_prediction_if_available),
        ("Evaluation: Multiclass Accuracy, Precision, Recall, Macro F1", test_compute_classification_metrics),
        ("Evaluation: Dedicated Melanoma Sensitivity & Specificity", test_compute_melanoma_metrics),
        ("Evaluation: Confusion Matrix & ROC Curves Generation", test_plot_generation),
        ("Explainability: Grad-CAM Overlay Dimensions & Value Clipping", test_gradcam_overlay_synthesis),
        ("Explainability: Real Model Grad-CAM Saliency Heatmap", test_real_gradcam_with_model),
        ("Multimodal: Localization Categories & Display Mappings", test_localization_categories_contract),
        ("Multimodal: Demographic Prior Estimation Bounds", test_clinical_prior_model_probabilities),
        ("Multimodal: Bayesian Fusion & Mathematical Normalization", test_bayesian_multimodal_fusion_properties),
        ("Multimodal: BCC Telangiectasia Differential Regression", test_bcc_telangiectasia_differentiation_regression),
        ("Multimodal: WeightedEnsemble End-to-End Integration", test_weighted_ensemble_multimodal_integration),
    ]

    print("\n" + "=" * 70)
    print("               DERMAVISION TEST SUITE EXECUTION")
    print("=" * 70)

    passed = 0
    failed = 0
    start_time = time.time()

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Summary: {passed} passed, {failed} failed in {elapsed:.2f}s")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
