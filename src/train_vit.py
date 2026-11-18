"""
train_vit.py - Vision Transformer (ViT) training module for Skin Cancer Detection.

Architecture:
- Patch Extraction (16x16 square non-overlapping patches) and Linear Projection
- Learnable Positional Embeddings
- Multi-Head Self-Attention Transformer Encoder Blocks (4 blocks, 4 heads)
- Multilayer Perceptron (MLP) classification head for 7 HAM10000 classes

Training Pipeline:
- Streams verified HAM10000 images from train.csv and val.csv
- Balances loss using inverse class frequency weights
- Data augmentation applied strictly to training batches
- Callbacks: ModelCheckpoint (best val_loss), EarlyStopping, ReduceLROnPlateau
- Evaluation executed on validation set (test set preserved for final evaluation)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:
    tf = None
    layers = None
    models = None

import config
from src.preprocessing import create_tf_dataset, compute_class_weights


@tf.keras.utils.register_keras_serializable(package="DermaVision")
class Patches(layers.Layer if layers else object):
    """Custom Keras layer to extract non-overlapping square patches from images."""
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
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


@tf.keras.utils.register_keras_serializable(package="DermaVision")
class PatchEncoder(layers.Layer if layers else object):
    """Project patches to latent projection dimension and add learnable positional embeddings."""
    def __init__(self, num_patches: int, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config


def build_vit_model(input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
                    num_classes: int = config.NUM_CLASSES,
                    patch_size: int = 16,
                    projection_dim: int = 64,
                    num_heads: int = 4,
                    transformer_layers: int = 4,
                    mlp_head_units: list = [256, 128]):
    """
    Construct Vision Transformer model for skin lesion diagnosis.

    Args:
        input_shape: Image shape (256, 256, 3).
        num_classes: Number of lesion classes (7).
        patch_size: Dimension of image patches (16x16).
        projection_dim: Embedding dimension.
        num_heads: Attention heads in transformer encoder.
        transformer_layers: Number of transformer blocks.
        mlp_head_units: Hidden units in final classification MLP.

    Returns:
        tf.keras.Model: Vision Transformer model.
    """
    if tf is None:
        raise ImportError("TensorFlow is required to build Vision Transformer.")

    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

    inputs = layers.Input(shape=input_shape, name="input_image")
    # Patch extraction and projection
    patches = Patches(patch_size=patch_size, name="patches")(inputs)
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim, name="patch_encoder")(patches)

    # Transformer Blocks
    x = encoded_patches
    for i in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6, name=f"norm_1_{i}")(x)
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1, name=f"mha_{i}"
        )(x1, x1)
        # Skip connection 1
        x2 = layers.Add(name=f"add_1_{i}")([attention_output, x])
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6, name=f"norm_2_{i}")(x2)
        # MLP Block
        x3 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu, name=f"mlp_dense_1_{i}")(x3)
        x3 = layers.Dropout(0.1, name=f"mlp_dropout_1_{i}")(x3)
        x3 = layers.Dense(projection_dim, activation=tf.nn.gelu, name=f"mlp_dense_2_{i}")(x3)
        x3 = layers.Dropout(0.1, name=f"mlp_dropout_2_{i}")(x3)
        # Skip connection 2
        x = layers.Add(name=f"add_2_{i}")([x3, x2])

    # Sequence representation via LayerNormalization & Global Pooling
    representation = layers.LayerNormalization(epsilon=1e-6, name="head_norm")(x)
    representation = layers.Flatten(name="flatten")(representation)
    representation = layers.Dropout(0.3, name="rep_dropout")(representation)

    # MLP Classification Head
    for j, units in enumerate(mlp_head_units):
        representation = layers.Dense(units, activation=tf.nn.gelu, name=f"fc_{j}")(representation)
        representation = layers.Dropout(0.2, name=f"fc_dropout_{j}")(representation)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(representation)

    model = models.Model(inputs=inputs, outputs=outputs, name="Vision_Transformer_ViT")
    return model


def train(epochs: int = config.EPOCHS,
          batch_size: int = config.BATCH_SIZE,
          learning_rate: float = config.LEARNING_RATE,
          save_path: Path = config.MODEL_PATHS["vit"],
          smoke_test: bool = False):
    """
    Execute training pipeline for Vision Transformer (ViT).

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial AdamW learning rate.
        save_path: Destination path for best model checkpoint (.keras).
        smoke_test: If True, runs 1 epoch on a tiny subset for quick CPU verification.
    """
    print("\n" + "=" * 70)
    print("  [DermaVision] Training: Vision Transformer (ViT)")
    print("=" * 70)

    # 1. Verify dataset split files exist
    if not config.TRAIN_CSV_PATH.is_file() or not config.VAL_CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset split CSVs missing. Expected:\n"
            f"  - {config.TRAIN_CSV_PATH}\n"
            f"  - {config.VAL_CSV_PATH}\n"
            f"Run 'python -m src.preprocessing' first to generate splits."
        )

    # 2. Load and verify metadata
    print("Loading split CSVs...")
    train_df = pd.read_csv(config.TRAIN_CSV_PATH)
    val_df = pd.read_csv(config.VAL_CSV_PATH)

    train_df = train_df[train_df['image_path'].apply(lambda p: os.path.isfile(str(p)))].reset_index(drop=True)
    val_df = val_df[val_df['image_path'].apply(lambda p: os.path.isfile(str(p)))].reset_index(drop=True)

    print(f"Verified training images:   {len(train_df)}")
    print(f"Verified validation images: {len(val_df)}")

    # 3. Smoke Test Configuration
    if smoke_test:
        print("\n*** RUNNING IN SMOKE TEST MODE (CPU Verification) ***")
        train_df = train_df.head(64)
        val_df = val_df.head(32)
        epochs = 1
        batch_size = min(batch_size, 16)
        print(f"Smoke test subset: {len(train_df)} train, {len(val_df)} val, epochs={epochs}, batch_size={batch_size}\n")

    # 4. Calculate Class Weights
    class_weights = compute_class_weights(train_df)
    print("Computed Training Class Weights:")
    for c_idx, w in sorted(class_weights.items()):
        print(f"  {config.IDX_TO_LABEL[c_idx]:<6}: {w:.4f}")

    # 5. Build tf.data Streaming Pipelines
    train_ds = create_tf_dataset(train_df, batch_size=batch_size, is_training=True, image_size=config.IMAGE_SIZE)
    val_ds = create_tf_dataset(val_df, batch_size=batch_size, is_training=False, image_size=config.IMAGE_SIZE)

    # 6. Build and Compile Model
    print("\nBuilding Vision Transformer...")
    model = build_vit_model(
        input_shape=config.INPUT_SHAPE,
        num_classes=config.NUM_CLASSES,
        patch_size=16,
        projection_dim=64,
        num_heads=4,
        transformer_layers=4
    )

    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=True)
        ]
    )

    # 7. Setup Callbacks
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 8. Training
    print(f"\nStarting Vision Transformer Training ({epochs} epochs)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # 9. Final Validation Evaluation
    print("\nEvaluating best model on Validation Set...")
    try:
        if save_path.is_file():
            eval_model = tf.keras.models.load_model(
                str(save_path),
                custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder}
            )
        else:
            eval_model = model
    except Exception as e:
        print(f"Notice: Evaluating in-memory restored model ({e})")
        eval_model = model

    val_metrics = eval_model.evaluate(val_ds, verbose=1)
    print(f"Validation Loss:     {val_metrics[0]:.4f}")
    print(f"Validation Accuracy: {val_metrics[1]:.4f}")
    print(f"Validation AUC:      {val_metrics[2]:.4f}")
    print(f"Best model weights saved to: {save_path.resolve()}")

    return eval_model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Transformer Model")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch quick CPU smoke test on small subset")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        smoke_test=args.smoke_test
    )
