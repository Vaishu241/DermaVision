"""
multimodal.py - Multimodal Clinical Decision Support for DermaVision.

Integrates Deep Learning Computer Vision with Patient Clinical Demographics:
1. Patient Age (continuous / clinical risk tiers)
2. Biological Sex (Male / Female / Unknown)
3. Anatomical Area / Region (15 anatomical sites, sun-exposure stratification)

Combines image probabilities P(disease | image) with clinical demographic priors
P(disease | age, sex, localization) via Bayesian Log-Odds Fusion and
dermatological arborizing-vessel calibration for Basal Cell Carcinoma.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

import config
from src.ensemble import get_top_k_predictions, compute_prediction_uncertainty, map_index_to_class


# ---------------------------------------------------------------------------
# Anatomical Region Definitions & Clinical Metadata Constants
# ---------------------------------------------------------------------------

LOCALIZATION_CATEGORIES = [
    "face",
    "scalp",
    "neck",
    "ear",
    "back",
    "chest",
    "trunk",
    "abdomen",
    "upper extremity",
    "lower extremity",
    "hand",
    "foot",
    "genital",
    "acral",
    "unknown"
]

LOCALIZATION_DISPLAY_MAP = {
    "face": "Face (Forehead, Cheeks, Nose, Temple)",
    "scalp": "Scalp",
    "neck": "Neck",
    "ear": "Ear / Helix / Pre-auricular",
    "back": "Back (Upper / Lower Thoracic)",
    "chest": "Chest / Sternal Area",
    "trunk": "Trunk / Flank",
    "abdomen": "Abdomen",
    "upper extremity": "Upper Extremity (Arm, Forearm, Shoulder)",
    "lower extremity": "Lower Extremity (Thigh, Calf, Knee)",
    "hand": "Hand (Dorsum, Palm)",
    "foot": "Foot (Dorsum, Sole)",
    "genital": "Genital / Inguinal",
    "acral": "Acral (Fingers, Toes, Nails)",
    "unknown": "Other / Unspecified Region"
}

SUN_EXPOSURE_TIERS = {
    "face": "Chronic High Sun Exposure",
    "scalp": "Chronic High Sun Exposure",
    "neck": "Chronic High Sun Exposure",
    "ear": "Chronic High Sun Exposure",
    "back": "Intermittent Sun Exposure",
    "chest": "Intermittent Sun Exposure",
    "trunk": "Intermittent Sun Exposure",
    "upper extremity": "Moderate Sun Exposure",
    "lower extremity": "Moderate Sun Exposure",
    "hand": "High Sun Exposure",
    "foot": "Low Sun Exposure",
    "abdomen": "Low Sun Exposure",
    "genital": "Non Sun-Exposed",
    "acral": "Extremity / Distal",
    "unknown": "Unspecified Exposure"
}


# ---------------------------------------------------------------------------
# Clinical Demographic Prior Model
# ---------------------------------------------------------------------------

class ClinicalPriorModel:
    """
    Epidemiological classifier and empirical prior estimator trained on HAM10000 metadata.
    Estimates P(disease | age, sex, localization).
    """

    def __init__(self, data_csv_path: Optional[Union[str, Path]] = None):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.classifier = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=6,
            class_weight="balanced",
            random_state=42
        )
        self.class_priors = np.ones(config.NUM_CLASSES) / config.NUM_CLASSES
        self.median_age = 50.0
        self.is_fitted = False

        self._fit_on_cohort(data_csv_path)

    def _fit_on_cohort(self, custom_path: Optional[Union[str, Path]] = None):
        """Fit empirical prior on training and validation metadata."""
        candidates = []
        if custom_path:
            candidates.append(Path(custom_path))
        candidates.extend([
            config.TRAIN_CSV_PATH,
            config.DATA_DIR / "processed" / "train.csv",
            Path(__file__).resolve().parent.parent / "data" / "processed" / "train.csv",
            Path("C:/Users/muvva/.gemini/antigravity/scratch/DermaVision/data/processed/train.csv"),
            Path("C:/Users/muvva/Downloads/DermaVision/HAM10000_metadata.csv")
        ])

        target_df = None
        for p in candidates:
            if p.is_file():
                try:
                    df = pd.read_csv(p)
                    if "dx" in df.columns or "target" in df.columns:
                        target_df = df
                        break
                except Exception:
                    continue

        if target_df is None:
            self.is_fitted = True
            return

        # Prepare labels
        if "target" in target_df.columns:
            y = target_df["target"].values
        elif "dx" in target_df.columns:
            y = target_df["dx"].map(config.LABEL_TO_IDX).values
        else:
            return

        # Prepare features
        cat_df = target_df[["sex", "localization"]].fillna("unknown")
        X_cat = self.encoder.fit_transform(cat_df)

        self.median_age = float(target_df["age"].median()) if "age" in target_df.columns and not target_df["age"].isna().all() else 50.0
        age_clean = target_df["age"].fillna(self.median_age).values / 100.0
        X = np.hstack([X_cat, age_clean.reshape(-1, 1)])

        # Compute empirical class priors
        counts = np.bincount(y, minlength=config.NUM_CLASSES)
        self.class_priors = (counts + 1.0) / (np.sum(counts) + config.NUM_CLASSES)

        # Fit gradient booster
        self.classifier.fit(X, y)
        self.is_fitted = True

    def predict_prior_probabilities(self, age: float, sex: str, localization: str) -> np.ndarray:
        """
        Estimate 7-class demographic prior P(class | age, sex, localization).

        Returns:
            np.ndarray of shape (7,) summing to 1.0.
        """
        if not self.is_fitted:
            return self.class_priors.copy()

        sex_clean = str(sex).lower().strip()
        if sex_clean not in ["male", "female"]:
            sex_clean = "unknown"

        loc_clean = str(localization).lower().strip()
        if loc_clean not in LOCALIZATION_CATEGORIES:
            loc_clean = "unknown"

        age_val = float(age) if age is not None else self.median_age
        age_scaled = np.clip(age_val, 0.0, 100.0) / 100.0

        meta_df = pd.DataFrame([{"sex": sex_clean, "localization": loc_clean}])
        X_cat = self.encoder.transform(meta_df[["sex", "localization"]])
        X = np.hstack([X_cat, np.array([[age_scaled]])])

        probs = self.classifier.predict_proba(X)[0]
        probs = np.clip(probs, 1e-6, 1.0)
        return probs / np.sum(probs)


_GLOBAL_CLINICAL_MODEL: Optional[ClinicalPriorModel] = None

def get_clinical_prior_model() -> ClinicalPriorModel:
    global _GLOBAL_CLINICAL_MODEL
    if _GLOBAL_CLINICAL_MODEL is None:
        _GLOBAL_CLINICAL_MODEL = ClinicalPriorModel()
    return _GLOBAL_CLINICAL_MODEL


# ---------------------------------------------------------------------------
# Multimodal Bayesian Fusion & Clinical Reasoning Engine
# ---------------------------------------------------------------------------

def fuse_image_and_demographics(
    image_probabilities: Union[np.ndarray, List[float]],
    age: float = 65.0,
    sex: str = "male",
    localization: str = "face",
    prior_strength: float = 1.25,
    enable_telangiectasia_heuristics: bool = True
) -> Dict[str, Any]:
    """
    Fuse computer vision probabilities with patient clinical demographics.
    """
    p_img = np.asarray(image_probabilities, dtype=np.float64).squeeze()
    if p_img.ndim != 1 or len(p_img) != config.NUM_CLASSES:
        raise ValueError(f"Expected probability array of length {config.NUM_CLASSES}, got {p_img.shape}")

    p_img = np.clip(p_img, 1e-12, 1.0)
    p_img = p_img / np.sum(p_img)

    model = get_clinical_prior_model()
    p_meta = model.predict_prior_probabilities(age=age, sex=sex, localization=localization)
    p_prior = model.class_priors

    eps = 1e-8
    log_img = np.log(p_img + eps)
    log_meta = np.log(p_meta + eps)
    log_prior = np.log(p_prior + eps)

    evidence_lift = log_meta - log_prior
    log_fusion = log_img + prior_strength * evidence_lift

    vasc_idx = config.LABEL_TO_IDX["vasc"]
    bcc_idx = config.LABEL_TO_IDX["bcc"]
    loc_clean = str(localization).lower().strip()
    sun_damaged_sites = ["face", "scalp", "neck", "ear", "back", "chest", "trunk"]

    applied_heuristic = False
    heuristic_reason = ""

    if enable_telangiectasia_heuristics:
        if p_img[vasc_idx] > 0.25 and float(age) >= 45 and loc_clean in sun_damaged_sites:
            applied_heuristic = True
            vasc_excess = p_img[vasc_idx]
            age_factor = min(1.0, max(0.2, (float(age) - 40.0) / 30.0))
            transfer_boost = 3.2 * age_factor * vasc_excess

            log_fusion[bcc_idx] += transfer_boost
            log_fusion[vasc_idx] -= 2.0 * transfer_boost
            heuristic_reason = (
                f"Clinical Decision Support detected prominent telangiectatic branching vessels on chronic sun-exposed skin "
                f"({loc_clean.capitalize()}, Age {int(age)}). In older adults, arborizing vessels are a classic hallmark "
                f"of nodular Basal Cell Carcinoma (BCC) rather than benign vascular lesions."
            )

    log_fusion_stable = log_fusion - np.max(log_fusion)
    fused_probs = np.exp(log_fusion_stable)
    fused_probs = fused_probs / np.sum(fused_probs)

    top_k = get_top_k_predictions(fused_probs, k=3)
    image_top = get_top_k_predictions(p_img, k=1)[0]
    primary = top_k[0]

    if applied_heuristic:
        shift_note = heuristic_reason
    elif primary["code"] != image_top["code"]:
        shift_note = (
            f"Patient clinical profile (Age {int(age)}, {sex.capitalize()}, {loc_clean.capitalize()}) "
            f"re-ranked the differential diagnosis: elevated {primary['label']} ({primary['probability']*100:.1f}%) "
            f"over image-only suspect {image_top['label']} ({image_top['probability']*100:.1f}%)."
        )
    else:
        shift_note = (
            f"Clinical demographics (Age {int(age)}, {sex.capitalize()}, {loc_clean.capitalize()}) "
            f"reinforced the diagnosis of {primary['label']} with consistent epidemiological alignment."
        )

    uncertainty = compute_prediction_uncertainty(fused_probs)

    fused_dict = {config.CLASS_NAMES[i]: float(fused_probs[i]) for i in range(config.NUM_CLASSES)}
    meta_dict = {config.CLASS_NAMES[i]: float(p_meta[i]) for i in range(config.NUM_CLASSES)}
    image_dict = {config.CLASS_NAMES[i]: float(p_img[i]) for i in range(config.NUM_CLASSES)}

    return {
        "multimodal_probabilities": fused_dict,
        "multimodal_vector": fused_probs,
        "image_probabilities": image_dict,
        "clinical_prior_probabilities": meta_dict,
        "top_k": top_k,
        "primary_prediction": primary,
        "image_top_prediction": image_top,
        "clinical_shift_note": shift_note,
        "applied_telangiectasia_logic": applied_heuristic,
        "uncertainty": uncertainty,
        "demographics": {
            "age": float(age),
            "sex": sex,
            "localization": loc_clean,
            "sun_exposure": SUN_EXPOSURE_TIERS.get(loc_clean, "Unspecified")
        }
    }
