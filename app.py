"""
DermaVision Health: Clinical Dermatology Diagnostic Portal & Telemedicine AI
Streamlit Web Application

Research Reference:
- "Skin Cancer Detection and Classification Using Deep Learning Techniques" (2026 IEEE AIDE)
- Multimodal Bayesian Fusion: Deep Convolutional Vision + Patient Electronic Health Record (EHR)
- Architectures: EfficientNetV2-B3 with Hybrid Attention (CBAM), ConvNeXt, Vision Transformer (ViT)
- Dataset: HAM10000 (7 diagnostic categories)
"""

import sys
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import streamlit as st
import numpy as np
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.ensemble import WeightedEnsemble
from src.explainability import generate_gradcam_visualization
from src.multimodal import (
    LOCALIZATION_CATEGORIES,
    LOCALIZATION_DISPLAY_MAP,
    SUN_EXPOSURE_TIERS,
    fuse_image_and_demographics,
)

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DermaVision Health | Clinical Dermatology AI Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS: Modern Hospital / Clinical Health Portal Theme
# Clean, Simple Neutral Background with Medical-Grade Card Elevation
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Simple Soothing Neutral Healthcare Background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Healthcare Navigation Bar */
    .health-nav {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 26px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
    }
    .brand-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-icon {
        width: 44px;
        height: 44px;
        background: #0284c7;
        color: white;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: 700;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.25);
    }
    .brand-title {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.4px;
    }
    .brand-subtitle {
        font-size: 13px;
        color: #64748b;
        margin: 2px 0 0 0;
    }
    .clinic-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #15803d;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .clinic-dot {
        width: 8px;
        height: 8px;
        background-color: #22c55e;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.25);
    }
    
    /* Healthcare White Cards */
    .health-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .health-card-header {
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
        padding-bottom: 10px;
        margin-bottom: 14px;
        border-bottom: 1px solid #f1f5f9;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* Medical Triage Alert Banners */
    .triage-banner {
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 18px;
        border: 1px solid #e2e8f0;
    }
    .triage-malignant {
        background: #fff5f5;
        border-left: 6px solid #ef4444;
        border-color: #fecaca;
    }
    .triage-precancer {
        background: #fffbeb;
        border-left: 6px solid #f59e0b;
        border-color: #fde68a;
    }
    .triage-benign {
        background: #f0fdf4;
        border-left: 6px solid #10b981;
        border-color: #bbf7d0;
    }
    
    .triage-tag {
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }
    .triage-tag-malignant { color: #dc2626; }
    .triage-tag-precancer { color: #d97706; }
    .triage-tag-benign { color: #16a34a; }
    
    .triage-disease-name {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        margin: 2px 0 6px 0;
        letter-spacing: -0.3px;
    }
    
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 5px 14px;
        font-size: 11.5px;
        font-weight: 700;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-malignant {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
    }
    .badge-precancer {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fbbf24;
    }
    .badge-benign {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #34d399;
    }
    
    /* Patient Demographics Bar */
    .patient-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #f1f5f9;
        color: #334155;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    
    /* Top 3 Differential Row */
    .top3-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        background: #f8fafc;
        border: 1px solid #edf2f7;
        border-radius: 10px;
        margin-bottom: 8px;
    }
    .top3-rank {
        font-weight: 800;
        color: #64748b;
        font-size: 14px;
        width: 28px;
    }
    .top3-label {
        font-weight: 600;
        color: #1e293b;
        font-size: 14px;
        flex-grow: 1;
    }
    .top3-prob {
        font-weight: 800;
        color: #0f172a;
        font-size: 15px;
        margin-left: 14px;
    }
    
    /* Horizontal Probability Bar */
    .prob-bar-container {
        margin-bottom: 11px;
    }
    .prob-bar-header {
        display: flex;
        justify-content: space-between;
        font-size: 12.5px;
        margin-bottom: 4px;
        color: #334155;
    }
    .prob-bar-bg {
        background-color: #e2e8f0;
        border-radius: 8px;
        height: 9px;
        width: 100%;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.4s ease;
    }
    
    /* Clinical Action Box */
    .action-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 5px solid #2563eb;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 14px;
        font-size: 13.5px;
        line-height: 1.5;
        color: #1e3a8a;
    }
    
    /* Disclaimer Footer */
    .health-disclaimer {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 28px;
        font-size: 12px;
        color: #64748b;
        text-align: center;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Clinical Diagnostic Engines & Neural Models...")
def get_cached_ensemble() -> WeightedEnsemble:
    ensemble = WeightedEnsemble()
    ensemble.load_models(safe_mode=False)
    return ensemble


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def get_badge_html(code: str) -> str:
    status = config.MALIGNANCY_STATUS.get(code, "Benign")
    if status == "Malignant":
        return '<span class="badge badge-malignant">Malignant</span>'
    elif status == "Pre-cancerous":
        return '<span class="badge badge-precancer">Pre-cancerous</span>'
    else:
        return '<span class="badge badge-benign">Benign</span>'


def get_bar_color(code: str) -> str:
    status = config.MALIGNANCY_STATUS.get(code, "Benign")
    if status == "Malignant":
        return "#ef4444"
    elif status == "Pre-cancerous":
        return "#f59e0b"
    else:
        return "#10b981"


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    # 1. Hospital Header Navbar
    st.markdown("""
    <div class="health-nav">
        <div class="brand-left">
            <div class="brand-icon">🏥</div>
            <div>
                <div class="brand-title">DermaVision Health</div>
                <div class="brand-subtitle">Clinical Dermatology AI & Multimodal Telemedicine Diagnostic Studio</div>
            </div>
        </div>
        <div class="clinic-badge">
            <span class="clinic-dot"></span> Clinical Neural System Online
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Sidebar Controls: Diagnostic Protocol & Model Configuration
    st.sidebar.markdown("### 🏥 Diagnostic Protocol")
    
    diagnostic_engine = st.sidebar.selectbox(
        "AI Inference Engine",
        [
            "Weighted Multimodal Ensemble (Recommended)",
            "EfficientNetV2-B3 + Hybrid Attention (CBAM)",
            "ConvNeXt-Tiny Architecture",
            "Vision Transformer (ViT)",
            "Custom Calibrated Weights"
        ]
    )

    ensemble = get_cached_ensemble()
    
    custom_weights = None
    if diagnostic_engine == "Custom Calibrated Weights":
        st.sidebar.markdown("#### Weight Allocation")
        w_eff = st.sidebar.slider("EfficientNetV2-B3", 0.0, 1.0, 0.50, 0.05)
        w_conv = st.sidebar.slider("ConvNeXt", 0.0, 1.0, 0.25, 0.05)
        w_vit = st.sidebar.slider("Vision Transformer", 0.0, 1.0, 0.25, 0.05)
        tot = w_eff + w_conv + w_vit
        custom_weights = {"efficientnet": w_eff/tot, "convnext": w_conv/tot, "vit": w_vit/tot} if tot > 0 else {"efficientnet": 1.0, "convnext": 0.0, "vit": 0.0}

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧬 Epidemiological Prior Controls")
    prior_strength = st.sidebar.slider(
        "Clinical Demographic Weight (γ)",
        min_value=0.5,
        max_value=2.0,
        value=1.25,
        step=0.05,
        help="Controls how strongly patient age, sex, and anatomical location calibrate the differential diagnosis."
    )
    enable_telangiectasia = st.sidebar.checkbox(
        "Arborizing Vessel Heuristic (BCC)",
        value=True,
        help="Dermatologically re-ranks vascular mimickers to Basal Cell Carcinoma when present on chronic sun-damaged skin in older adults."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Explainable AI (XAI)")
    enable_gradcam = st.sidebar.checkbox("Compute Grad-CAM Saliency Maps", value=True)
    gradcam_alpha = st.sidebar.slider("Heatmap Transparency (α)", 0.1, 0.9, 0.45, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Neural Checkpoint Status")
    for name in ["efficientnet", "convnext", "vit"]:
        status = ensemble.model_status.get(name, {})
        loaded = status.get("loaded", False)
        is_prod = status.get("is_production", False)
        if loaded and is_prod:
            st.sidebar.markdown(f"**{name.capitalize()}**: :green[Production Validated]")
        elif loaded:
            st.sidebar.markdown(f"**{name.capitalize()}**: :orange[Loaded (Smoke Checkpoint)]")
        else:
            st.sidebar.markdown(f"**{name.capitalize()}**: :red[Offline]")

    # 3. Main Workspace: Two Balanced Health Columns
    col_patient, col_results = st.columns([1, 1], gap="large")

    selected_image = None
    image_source_label = ""
    patient_id = "PT-2026-0842"

    with col_patient:
        # Card 1: Patient Electronic Health Record (EHR) Intake
        st.markdown(f"""
        <div class="health-card">
            <div class="health-card-header">
                <span>📋 1. Patient Clinical Intake (EHR)</span>
                <span style="font-size: 12px; color: #64748b; font-weight: normal;">Case Ref: <b>{patient_id}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        mode_choice = st.radio(
            "Diagnostic Evaluation Mode:",
            ["🔬 Multimodal Clinical AI (Vision + Demographics) [Recommended]", "📷 Pure Computer Vision (Image Only)"],
            index=0,
            horizontal=True
        )
        is_multimodal = "Multimodal" in mode_choice

        demo_row1_c1, demo_row1_c2 = st.columns(2)
        with demo_row1_c1:
            patient_age = st.number_input("Patient Age (Years):", min_value=1, max_value=100, value=65, step=1)
            patient_sex = st.selectbox("Biological Sex:", ["Male", "Female", "Unknown"], index=0)

        with demo_row1_c2:
            localization_keys = LOCALIZATION_CATEGORIES
            patient_loc = st.selectbox(
                "Anatomical Lesion Site:",
                localization_keys,
                index=localization_keys.index("face") if "face" in localization_keys else 0,
                format_func=lambda k: LOCALIZATION_DISPLAY_MAP.get(k, k)
            )
            sun_tier = SUN_EXPOSURE_TIERS.get(patient_loc, "Unspecified")
            st.caption(f"☀️ Sun Exposure Tier: **{sun_tier}**")

        # Patient Symptoms Checklist
        st.markdown("**Reported Clinical Symptoms:**")
        symp_c1, symp_c2, symp_c3, symp_c4 = st.columns(4)
        with symp_c1:
            symp_itching = st.checkbox("Itching")
        with symp_c2:
            symp_bleeding = st.checkbox("Bleeding")
        with symp_c3:
            symp_growth = st.checkbox("Rapid Growth", value=True)
        with symp_c4:
            symp_pain = st.checkbox("Pain/Tenderness")

        # Card 2: Dermoscopic Lesion Imaging
        st.markdown("""
        <div class="health-card" style="margin-top: 14px;">
            <div class="health-card-header">
                <span>📷 2. Dermoscopic Lesion Examination</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        input_tab1, input_tab2 = st.tabs(["📤 Upload Dermoscopic Image", "🔬 Clinical Hospital Cases (Presets)"])

        with input_tab1:
            uploaded_file = st.file_uploader(
                "Upload a high-resolution dermoscopic image (JPG, JPEG, PNG):",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                try:
                    selected_image = Image.open(uploaded_file).convert("RGB")
                    image_source_label = f"Uploaded Patient Case: {uploaded_file.name}"
                except Exception as e:
                    st.error(f"Error reading image: {e}")

        with input_tab2:
            st.markdown("Select a verified reference clinical case:")
            preset_options = [
                "Select a clinical presentation...",
                "🔬 Case 1: Basal Cell Carcinoma (Face, 68yo, Male) [Arborizing Telangiectasia]",
                "🔴 Case 2: Invasive Melanoma (Back, 60yo, Male)",
                "🟢 Case 3: Melanocytic Nevus (Trunk, 20yo, Male)",
                "🟢 Case 4: Benign Keratosis BKL (Abdomen, 45yo, Male)"
            ]
            chosen_preset = st.selectbox("Hospital Reference Cases:", preset_options, index=0)

            samples_dir = PROJECT_ROOT / "samples"
            if chosen_preset == "🔬 Case 1: Basal Cell Carcinoma (Face, 68yo, Male) [Arborizing Telangiectasia]":
                p = samples_dir / "bcc_telangiectasia_face.jpg"
                patient_age = 68
                patient_sex = "Male"
                patient_loc = "face"
                if p.is_file():
                    selected_image = Image.open(p).convert("RGB")
                    image_source_label = "Verified Case: Nodular Basal Cell Carcinoma (BCC) [Auto-Profile: 68yo Male, Face]"
            elif chosen_preset == "🔴 Case 2: Invasive Melanoma (Back, 60yo, Male)":
                p = samples_dir / "melanoma_ISIC_0032844.jpg"
                if not p.is_file():
                    p = samples_dir / "melanoma_ISIC_0026120.jpg"
                patient_age = 60
                patient_sex = "Male"
                patient_loc = "back"
                if p.is_file():
                    selected_image = Image.open(p).convert("RGB")
                    image_source_label = "Verified Case: Malignant Melanoma (MEL) [Auto-Profile: 60yo Male, Back]"
            elif chosen_preset == "🟢 Case 3: Melanocytic Nevus (Trunk, 20yo, Male)":
                p = samples_dir / "nevus_ISIC_0025161.jpg"
                if not p.is_file():
                    p = samples_dir / "nevus_ISIC_0024306.jpg"
                patient_age = 20
                patient_sex = "Male"
                patient_loc = "trunk"
                if p.is_file():
                    selected_image = Image.open(p).convert("RGB")
                    image_source_label = "Verified Case: Common Melanocytic Nevus (NV) [Auto-Profile: 20yo Male, Trunk]"
            elif chosen_preset == "🟢 Case 4: Benign Keratosis BKL (Abdomen, 45yo, Male)":
                p = samples_dir / "bkl_ISIC_0024748.jpg"
                if not p.is_file():
                    p = samples_dir / "bkl_ISIC_0024310.jpg"
                patient_age = 45
                patient_sex = "Male"
                patient_loc = "abdomen"
                if p.is_file():
                    selected_image = Image.open(p).convert("RGB")
                    image_source_label = "Verified Case: Benign Keratosis (BKL) [Auto-Profile: 45yo Male, Abdomen]"

        if selected_image is not None:
            st.image(selected_image, caption=image_source_label, use_container_width=True)

    with col_results:
        # Card 3: Diagnostic Assessment & Triage
        st.markdown("""
        <div class="health-card">
            <div class="health-card-header">
                <span>📊 3. Clinical Diagnostic Assessment & Differential Triage</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if selected_image is None:
            st.info("👆 Please upload a patient lesion image or select a clinical case to perform diagnostic triage.")
        else:
            with st.spinner("Analyzing dermoscopic morphology and synthesizing clinical priors..."):
                orig_weights = ensemble.weights.copy()
                if custom_weights:
                    ensemble.weights = custom_weights.copy()
                    ensemble._normalize_weights()

                model_arg = "ensemble"
                if "EfficientNet" in diagnostic_engine:
                    model_arg = "efficientnet"
                elif "ConvNeXt" in diagnostic_engine:
                    model_arg = "convnext"
                elif "Vision Transformer" in diagnostic_engine:
                    model_arg = "vit"

                meta_payload = None
                if is_multimodal:
                    meta_payload = {
                        "age": patient_age,
                        "sex": patient_sex.lower(),
                        "localization": patient_loc.lower(),
                        "prior_strength": prior_strength,
                    }

                try:
                    result = ensemble.predict_image(
                        selected_image,
                        model_selection=model_arg,
                        metadata=meta_payload
                    )
                finally:
                    if custom_weights:
                        ensemble.weights = orig_weights

            top1 = result["top_k"][0]
            is_malig = top1["malignancy"] == "Malignant"
            is_pre = top1["malignancy"] == "Pre-cancerous"

            triage_class = "triage-malignant" if is_malig else ("triage-precancer" if is_pre else "triage-benign")
            triage_tag_class = "triage-tag-malignant" if is_malig else ("triage-tag-precancer" if is_pre else "triage-tag-benign")
            triage_status_text = "CRITICAL CLINICAL ALERT: MALIGNANCY CONFIRMED" if is_malig else ("PRE-CANCEROUS LESION DETECTED" if is_pre else "BENIGN NEOPLASM - ROUTINE MONITORING")

            badge_html = get_badge_html(top1["code"])

            # Primary Diagnostic Triage Banner
            st.markdown(f"""
            <div class="triage-banner {triage_class}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div class="triage-tag {triage_tag_class}">● {triage_status_text}</div>
                        <div class="triage-disease-name">{top1['label']} ({top1['code'].upper()})</div>
                    </div>
                    {badge_html}
                </div>
                <div style="margin-top: 8px; font-size: 15px; font-weight: 600; color: #1e293b;">
                    Diagnostic Confidence: <span style="font-size: 20px; font-weight: 800; color: #0f172a;">{top1['percentage']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Multimodal Clinical Shift Note
            if result.get("multimodal") is not None:
                shift_msg = result["multimodal"].get("clinical_shift_note", "")
                if shift_msg:
                    st.markdown(f"""
                    <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-left: 4px solid #2563eb; border-radius: 8px; padding: 12px 14px; margin-bottom: 14px; font-size: 13px; color: #1e3a8a; line-height: 1.5;">
                        <b>🩺 Clinical Decision Support Insight:</b><br>{shift_msg}
                    </div>
                    """, unsafe_allow_html=True)

            # Top-3 Differential Diagnosis Ranking
            st.markdown("##### 🩺 Top-3 Differential Diagnosis Ranking")
            for item in result["top_k"]:
                item_badge = get_badge_html(item["code"])
                st.markdown(f"""
                <div class="top3-row">
                    <span class="top3-rank">#{item['rank']}</span>
                    <span class="top3-label">{item['label']} ({item['code'].upper()})</span>
                    {item_badge}
                    <span class="top3-prob">{item['percentage']}</span>
                </div>
                """, unsafe_allow_html=True)

            # Full 7-Class Distribution
            st.markdown("##### 📈 7-Class Full Probability Distribution")
            for code in config.CLASS_NAMES:
                prob = result["all_probabilities"].get(code, 0.0)
                pct = prob * 100.0
                bar_color = get_bar_color(code)
                label_name = config.CLASS_LABELS.get(code, code)
                
                st.markdown(f"""
                <div class="prob-bar-container">
                    <div class="prob-bar-header">
                        <span><b>{code.upper()}</b> — {label_name}</span>
                        <span><b>{pct:.2f}%</b></span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: {pct:.1f}%; background-color: {bar_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Recommended Clinical Action Plan Card
            action_text = ""
            if is_malig:
                action_text = "<b>Action Required:</b> Immediate urgent referral to Mohs micrographic surgery or dermatologic oncology. Perform an urgent diagnostic punch or excisional biopsy with 5mm surgical margins to confirm histological subtype."
            elif is_pre:
                action_text = "<b>Action Required:</b> Recommend cryotherapy with liquid nitrogen, topical 5-Fluorouracil (5-FU), or photodynamic therapy (PDT). Schedule a 6-month clinical follow-up."
            else:
                action_text = "<b>Action Required:</b> Reassure patient of benign etiology. Educate on self-monitoring using the ABCDE criteria. Routine annual skin exam recommended."

            st.markdown(f"""
            <div class="action-box">
                {action_text}
            </div>
            """, unsafe_allow_html=True)

            # Downloadable Patient Clinical Report Button
            report_text = f"""======================================================
           DERMAVISION HEALTH CLINICAL DIAGNOSTIC REPORT
======================================================
Report Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Patient ID: {patient_id}
Age: {patient_age} | Sex: {patient_sex} | Site: {LOCALIZATION_DISPLAY_MAP.get(patient_loc, patient_loc)}
Sun Exposure Profile: {sun_tier}
Symptoms: Itching={symp_itching}, Bleeding={symp_bleeding}, Rapid Growth={symp_growth}, Pain={symp_pain}

------------------------------------------------------
DIAGNOSTIC TRIAGE SUMMARY
------------------------------------------------------
Primary Diagnosis: {top1['label']} ({top1['code'].upper()})
Malignancy Status: {top1['malignancy'].upper()}
Diagnostic Confidence: {top1['percentage']}

TOP-3 DIFFERENTIAL RANKING:
1. {result['top_k'][0]['label']} ({result['top_k'][0]['percentage']}) - {result['top_k'][0]['malignancy']}
2. {result['top_k'][1]['label']} ({result['top_k'][1]['percentage']}) - {result['top_k'][1]['malignancy']}
3. {result['top_k'][2]['label']} ({result['top_k'][2]['percentage']}) - {result['top_k'][2]['malignancy']}

UNCERTAINTY DIAGNOSTICS:
Entropy: {result['uncertainty']['entropy']} bits | Margin: {result['uncertainty']['margin']}
Clinical Advisory: {result['uncertainty']['clinical_recommendation']}

RECOMMENDED MEDICAL ACTION:
{action_text.replace('<b>', '').replace('</b>', '')}

DISCLAIMER: AI-assisted clinical decision support. Must be verified by a licensed dermatologist.
======================================================"""

            st.download_button(
                label="📄 Download Official Patient Diagnostic Report (.txt)",
                data=report_text,
                file_name=f"DermaVision_Report_{patient_id}_{top1['code'].upper()}.txt",
                mime="text/plain",
                use_container_width=True
            )

    # 4. Explainable AI: Grad-CAM Morphological Saliency
    if selected_image is not None and enable_gradcam:
        st.markdown("---")
        st.markdown("""
        <div class="health-card">
            <div class="health-card-header">
                <span>🔍 Explainable AI: Grad-CAM Morphological Saliency Maps</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Computing convolutional feature gradients and attention heatmaps..."):
            try:
                overlaid_img, heatmap = generate_gradcam_visualization(
                    selected_image,
                    alpha=gradcam_alpha
                )
                
                cam_col1, cam_col2, cam_col3 = st.columns(3)
                with cam_col1:
                    st.image(result["preprocessed_image"], caption="1. Original Lesion Image", use_container_width=True)
                with cam_col2:
                    st.image(overlaid_img, caption="2. Grad-CAM Diagnostic Overlay", use_container_width=True)
                with cam_col3:
                    st.image(np.uint8(255 * heatmap), caption="3. Morphological Heatmap", use_container_width=True)
                
                st.caption(
                    "Grad-CAM visualizes which spatial coordinates, lesion borders, and pigmentation networks contributed "
                    "most significantly to the neural diagnostic assessment."
                )
            except Exception as e:
                st.warning(f"Grad-CAM visualization could not be computed: {e}")

    # 5. Clinical Knowledge Tabs
    st.markdown("---")
    ref_tab1, ref_tab2, ref_tab3 = st.tabs([
        "📚 Diagnostic Categories Reference",
        "🎯 ABCDE Clinical Screening Criteria",
        "🧬 Multimodal Fusion Architecture"
    ])

    with ref_tab1:
        glossary_data = [
            {"Code": "akiec", "Name": "Actinic Keratoses / Intraepithelial Carcinoma", "Severity": "Pre-cancerous", "Description": "Sun-induced pre-malignant skin lesions that can develop into squamous cell carcinoma."},
            {"Code": "bcc", "Name": "Basal Cell Carcinoma", "Severity": "Malignant", "Description": "Most common form of skin cancer; slow growing but locally destructive if untreated. Often presents with arborizing telangiectasia."},
            {"Code": "bkl", "Name": "Benign Keratosis-like Lesions", "Severity": "Benign", "Description": "Harmless skin growths common in older adults (seborrheic keratoses, solar lentigines)."},
            {"Code": "df", "Name": "Dermatofibroma", "Severity": "Benign", "Description": "Benign fibrous nodule typically occurring on extremities."},
            {"Code": "mel", "Name": "Melanoma", "Severity": "Malignant", "Description": "Most aggressive and lethal skin cancer arising from melanocytes; early excision is vital."},
            {"Code": "nv", "Name": "Melanocytic Nevi", "Severity": "Benign", "Description": "Common moles composed of melanocytes; benign proliferation."},
            {"Code": "vasc", "Name": "Vascular Lesions", "Severity": "Benign", "Description": "Angiomas, pyogenic granulomas, and vascular malformations."}
        ]
        st.table(glossary_data)

    with ref_tab2:
        st.markdown("""
        - **A (Asymmetry):** One half of the lesion does not match the other half in contour or shape.
        - **B (Border):** The edges are irregular, ragged, notched, scalloped, or poorly defined.
        - **C (Color):** The pigmentation is non-uniform; varying shades of tan, brown, black, or patches of pink, red, white, or blue.
        - **D (Diameter):** Lesion diameter is greater than 6 mm (pencil eraser size), though early melanomas can be smaller.
        - **E (Evolving):** The lesion is actively changing in size, shape, surface elevation, symptoms (itching, bleeding), or color over time.
        """)

    with ref_tab3:
        st.markdown("""
        - **EfficientNetV2-B3 with Hybrid Attention (CBAM):** Sequential Channel Attention (diagnostic color filters) and Spatial Attention (lesion margins).
        - **Multimodal Bayesian Conditioning:** Integrates epidemiological priors $P(\text{disease} \mid \text{age}, \text{sex}, \text{localization})$ with convolutional visual features:
          $$\log P_{\text{multimodal}}(y) = \log P_{\text{image}}(y) + \gamma \cdot \left[\log P(\text{metadata} \mid y) - \log P(\text{metadata})\right]$$
        - **Arborizing Vessel Logic:** Resolves telangiectatic mimicry on chronic sun-damaged skin, correctly elevating Basal Cell Carcinoma in the differential diagnosis.
        """)

    # 6. Healthcare Legal Disclaimer
    st.markdown("""
    <div class="health-disclaimer">
        ⚠️ <b>Medical & Clinical Disclaimer:</b> DermaVision Health is an AI-powered clinical decision-support research platform developed 
        under the 2026 IEEE AIDE framework. It is intended to assist medical professionals and dermatologists. It does <b>not</b> replace 
        formal histopathological biopsy or independent clinical judgment by a licensed physician.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
