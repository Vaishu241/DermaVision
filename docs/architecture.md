# DermaVision Architecture & Research Methodology

Based on: *"Skin Cancer Detection and Classification Using Deep Learning Techniques"* (2026 IEEE AIDE).

---

## 1. Pipeline Overview

```
                      +-----------------------------+
                      |   HAM10000 Dataset (Raw)    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Preprocessing & Augmentation|
                      |  - Patient Leakage Handling |
                      |  - CLAHE / Color Normalization|
                      |  - Spatial Transformations  |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
         v                           v                           v
+------------------+       +-------------------+       +--------------------+
|     ConvNeXt     |       | EfficientNetV2-B3 |       | Vision Transformer |
|     Backbone     |       | + Hybrid Attention|       |       (ViT)        |
+--------+---------+       +---------+---------+       +---------+----------+
         |                           |                           |
  p1 = f(x)                   p2 = f(x)                   p3 = f(x)
         |                           |                           |
         +---------------------------+---------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Weighted Probability Average|
                      | P_ens = w1*p1 + w2*p2 + w3*p3|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Evaluation & Explainability |
                      |  - Multiclass Metrics       |
                      |  - Grad-CAM Heatmaps        |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Streamlit Web Application   |
                      +-----------------------------+
```

---

## 2. Models & Components

### A. EfficientNetV2-B3 with Hybrid Attention (CBAM)
- **Backbone**: EfficientNetV2-B3 provides progressive learning with faster training speed and parameter efficiency.
- **Hybrid Attention**: Combines Channel Attention (focusing on 'what' features are significant across feature maps) and Spatial Attention (focusing on 'where' the lesion margins and pigment networks are located).

### B. ConvNeXt
- A modernized convolutional architecture utilizing depthwise separable convolutions, inverted bottlenecks, and larger 7x7 kernels that compete with transformer vision models while retaining convolutional inductive bias.

### C. Vision Transformer (ViT)
- Divides dermoscopic images into non-overlapping patches (e.g., 16x16), projects them linearly into sequence embeddings, and applies multi-head self-attention to capture long-range global context and border irregularities.

### D. Weighted Ensemble
- Aggregates softmax probabilities:
  $$\hat{y} = \arg\max \sum_{m=1}^{M} w_m \cdot P_m(y \mid x)$$
  where $\sum w_m = 1$, mitigating individual model bias and improving multiclass classification robustness across rare classes (such as `df` and `vasc`).

### E. Explainability (Grad-CAM)
- Generates coarse 2D heatmaps of the gradients flowing into the final convolutional layer, highlighting the precise dermatoscopic structures (e.g. pigment network, streaks, blue-white veil) that influenced the classification.
