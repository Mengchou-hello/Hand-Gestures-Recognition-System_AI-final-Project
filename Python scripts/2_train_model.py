import tensorflow as tf
import keras
import numpy as np
from keras import layers, models
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, log_loss
import os

TRAIN_DIR  = "../dataset_split/train"
VAL_DIR    = "../dataset_split/val"
TEST_DIR   = "../dataset_split/test"
MODEL_PATH = "../models/gesture_cnn.keras"
IMG_SIZE   = 128
BATCH      = 32
CLASSES    = ["good_luck", "i_love_you", "i_want_to_talk_to_you", "victory"]
SEED       = 42
SPLIT_DIRS = {
    "train": TRAIN_DIR,
    "val": VAL_DIR,
    "test": TEST_DIR,
}

os.makedirs("../models", exist_ok=True)

if not all(os.path.isdir(split_dir) for split_dir in SPLIT_DIRS.values()):
    raise SystemExit(
        "Dataset split not found. Run '0_split_dataset.py' first to create train/val/test folders."
    )

print("Loading datasets...")

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=True,
    seed=SEED,
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=False,
)

print(f"Class names: {train_ds.class_names}")

# Augmentation + pretrained-model preprocessing pipeline
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1),
], name="augmentation")

def prepare_train(x, y):
    x = tf.cast(x, tf.float32)
    x = augment(x, training=True)
    x = preprocess_input(x)
    return x, y

def prepare_eval(x, y):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)
    return x, y


def load_eval_split(split_name):
    dataset = keras.utils.image_dataset_from_directory(
        SPLIT_DIRS[split_name],
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        label_mode="categorical",
        shuffle=False,
    )
    dataset = dataset.map(prepare_eval, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def print_accuracy_report():
    print("\n=== Accuracy Report ===")
    print(f"{'Split':<8} {'Samples':>8} {'Loss':>10} {'Accuracy':>10}")
    print("-" * 40)

    best_model = keras.models.load_model(MODEL_PATH)
    for split_name in ("train", "val", "test"):
        dataset = load_eval_split(split_name)
        all_probs = []
        all_targets = []

        for batch_x, batch_y in dataset:
            probs = best_model.predict(batch_x, verbose=0)
            all_probs.append(probs)
            all_targets.append(np.argmax(batch_y.numpy(), axis=1))

        probs = np.concatenate(all_probs, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        predictions = np.argmax(probs, axis=1)
        sample_count = len(targets)
        loss = log_loss(targets, probs, labels=list(range(len(CLASSES))))
        accuracy = accuracy_score(targets, predictions)
        print(f"{split_name:<8} {sample_count:>8} {loss:>10.4f} {accuracy:>9.2%}")

train_ds = train_ds.map(prepare_train, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(prepare_eval, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# Build model
base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                   include_top=False, weights="imagenet")
base.trainable = False

inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation="swish")(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
model   = models.Model(inputs, outputs)

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy", verbose=1),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, monitor="val_loss", verbose=1),
]

loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# Phase 1 — train head only
print("\n=== Phase 1: Training classifier head ===")
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=loss_fn, metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

# Phase 2 — fine-tune deeper layers
print("\n=== Phase 2: Fine-tuning top layers ===")
base.trainable = True
for layer in base.layers[:-80]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(3e-5),
              loss=loss_fn, metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=callbacks)

print(f"\nModel saved to {MODEL_PATH}")
print("\nRunning accuracy report...")
print_accuracy_report()
