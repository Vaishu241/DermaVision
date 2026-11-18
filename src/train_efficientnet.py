"""
train_efficientnet.py - Training EfficientNetV2-B3 with Hybrid Attention (CBAM) for Skin Cancer Detection.

Architecture:
- Backbone: EfficientNetV2-B3 (pretrained on ImageNet)
- Attention Mechanism: Hybrid Attention combining Channel Attention and Spatial Attention (Keras 3 compatible)
- Custom Classification Head: Dropout, Dense, Softmax for 7-class prediction

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


def build_channel_attention(input_tensor, ratio: int = 8):
    """
    Channel Attention Module: Focuses on 'what' features are meaningful.
    Aggregates spatial dimensions using GlobalAveragePooling and GlobalMaxPooling.
    """
    channel = input_tensor.shape[-1]
    shared_dense_one = layers.Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_dense_two = layers.Dense(channel, kernel_initializer='he_normal', use_bias=True)

    avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_tensor)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)
    return layers.Multiply()([input_tensor, cbam_feature])


def build_spatial_attention(input_tensor, kernel_size: int = 7):
    """
    Spatial Attention Module: Focuses on 'where' informative lesion regions are located.
    Uses Lambda layers for Keras 3 compatibility with symbolic KerasTensor.
    """
    avg_pool = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True), name="spatial_avg_pool")(input_tensor)
    max_pool = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True), name="spatial_max_pool")(input_tensor)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    spatial_feature = layers.Conv2D(1, (kernel_size, kernel_size), strides=1, padding='same',
                                    activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    return layers.Multiply()([input_tensor, spatial_feature])


def build_hybrid_attention_module(input_tensor):
    """
    Sequential Hybrid Attention: Applies Channel Attention followed by Spatial Attention.
    """
    x = build_channel_attention(input_tensor)
    x = build_spatial_attention(x)
    return x


def build_efficientnet_model(input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
                             num_classes: int = config.NUM_CLASSES,
                             freeze_backbone: bool = True):
    """
    Construct EfficientNetV2-B3 with Hybrid Attention and classification head.

    Args:
        input_shape: Dimensions of input images (H, W, C), default (256, 256, 3).
        num_classes: Number of target lesion classes (7 for HAM10000).
        freeze_backbone: Whether to freeze initial ImageNet backbone weights.

    Returns:
        tf.keras.Model: Configured model ready for training.
    """
    if tf is None:
        raise ImportError("TensorFlow is required to build EfficientNetV2.")

    # 1. Base Pretrained Backbone
    base_model = tf.keras.applications.EfficientNetV2B3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = not freeze_backbone

    # 2. Input and Backbone Feature Extraction
    inputs = layers.Input(shape=input_shape, name="input_image")
    feature_map = base_model(inputs)

    # 3. Hybrid Attention Mechanism
    attended_features = build_hybrid_attention_module(feature_map)

    # 4. Dense Classification Head
    x = layers.GlobalAveragePooling2D(name="avg_pool")(attended_features)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.3, name="top_dropout")(x)
    x = layers.Dense(256, activation='relu', name="fc_1")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(0.2, name="fc_dropout")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetV2_B3_HybridAttention")
    return model


def train(epochs: int = config.EPOCHS,
          batch_size: int = config.BATCH_SIZE,
          learning_rate: float = config.LEARNING_RATE,
          save_path: Path = config.MODEL_PATHS["efficientnet"],
          smoke_test: bool = False,
          fine_tune_epochs: int = 0):
    """
    Execute training pipeline for EfficientNetV2-B3 with Hybrid Attention.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial Adam learning rate.
        save_path: Destination path for best model checkpoint (.keras).
        smoke_test: If True, runs 1 epoch on a tiny subset for quick CPU verification.
        fine_tune_epochs: Number of fine-tuning epochs with unfrozen top backbone layers.
    """
    print("\n" + "=" * 70)
    print("  [DermaVision] Training: EfficientNetV2-B3 + Hybrid Attention")
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
    print(f"Loading split CSVs...")
    train_df = pd.read_csv(config.TRAIN_CSV_PATH)
    val_df = pd.read_csv(config.VAL_CSV_PATH)

    # Verify physical file existence
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
    print("\nBuilding EfficientNetV2-B3 with Hybrid Attention...")
    model = build_efficientnet_model(input_shape=config.INPUT_SHAPE, num_classes=config.NUM_CLASSES, freeze_backbone=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
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

    # 8. Initial Head Training
    print(f"\nStarting Phase 1 Training ({epochs} epochs)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # 9. Optional Fine-Tuning Phase
    if fine_tune_epochs > 0 and not smoke_test:
        print(f"\nStarting Phase 2 Fine-Tuning ({fine_tune_epochs} epochs)...")
        # Unfreeze top layers of EfficientNetV2
        base_model = model.layers[1]
        base_model.trainable = True
        # Fine-tune only the last 30 layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.1),
            loss=loss_fn,
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc", multi_label=True)
            ]
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

    # 10. Final Validation Evaluation
    print("\nEvaluating best model on Validation Set...")
    if save_path.is_file():
        eval_model = tf.keras.models.load_model(str(save_path))
    else:
        eval_model = model

    val_metrics = eval_model.evaluate(val_ds, verbose=1)
    print(f"Validation Loss:     {val_metrics[0]:.4f}")
    print(f"Validation Accuracy: {val_metrics[1]:.4f}")
    print(f"Validation AUC:      {val_metrics[2]:.4f}")
    print(f"Best model weights saved to: {save_path.resolve()}")

    return eval_model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNetV2-B3 with Hybrid Attention")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch quick CPU smoke test on small subset")
    parser.add_argument("--fine-tune", type=int, default=0, help="Number of fine-tuning epochs")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        smoke_test=args.smoke_test,
        fine_tune_epochs=args.fine_tune
    )
