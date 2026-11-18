# Model Checkpoints Directory

This directory stores trained model weights, architectures, and serialization artifacts for **DermaVision**.

## Expected Models

1. **`efficientnetv2_hybrid_attention.keras`**
   - EfficientNetV2-B3 backbone with custom Hybrid Attention (Channel + Spatial Attention / CBAM).
2. **`convnext_best.keras`**
   - ConvNeXt backbone fine-tuned for 7-class lesion classification.
3. **`vit_best.keras`**
   - Vision Transformer model trained on patch embeddings of dermoscopic images.

## Checkpoint Subdirectory
- `checkpoints/`: Intermediate training checkpoints created by `tf.keras.callbacks.ModelCheckpoint` during training runs.

*Do not fabricate or commit mock weight files. Weight files will be generated upon executing the respective training scripts.*
