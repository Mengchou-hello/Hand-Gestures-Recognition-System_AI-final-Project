import os

import keras
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

matplotlib.use("Agg")

TEST_DIR = "../dataset_split/test"
MODEL_PATH = "../models/gesture_cnn.keras"
CONFUSION_MATRIX_PATH = "../confusion_matrix.png"
IMG_SIZE = 128
CLASSES = ["good_luck", "i_love_you", "i_want_to_talk_to_you", "victory"]


if not os.path.isdir(TEST_DIR):
    raise SystemExit(
        "Test split not found. Run '0_split_dataset.py' first to create train/val/test folders."
    )

if not os.path.isfile(MODEL_PATH):
    raise SystemExit(
        "Saved model not found. Run '2_train_model.py' first."
    )

print("Step 1: Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded OK")

print("Step 2: Loading test dataset...")
for idx, class_name in enumerate(CLASSES):
    class_dir = os.path.join(TEST_DIR, class_name)
    if not os.path.isdir(class_dir):
        raise SystemExit(f"Missing class folder in test split: {class_dir}")

    files = sorted(
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"{class_name}: {len(files)} images")


def prepare_eval(x, y):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)
    return x, y


dataset = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    label_mode="categorical",
    shuffle=False,
)
dataset = dataset.map(prepare_eval, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

all_probs = []
all_targets = []
for batch_x, batch_y in dataset:
    all_probs.append(model.predict(batch_x, verbose=0))
    all_targets.append(np.argmax(batch_y.numpy(), axis=1))

probs = np.concatenate(all_probs, axis=0)
y = np.concatenate(all_targets, axis=0)

print(f"Total test samples: {len(y)}")

print("Step 3: Evaluating model...")
y_pred = np.argmax(probs, axis=1)
accuracy = accuracy_score(y, y_pred)
loss = log_loss(y, probs, labels=list(range(len(CLASSES))))

print("\n=== Test Summary ===")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4%}")

print("\n=== Classification Report ===")
print(classification_report(y, y_pred, target_names=CLASSES, digits=4))

print("=== Confusion Matrix ===")
cm = confusion_matrix(y, y_pred)
header = "pred->   " + " ".join(f"{name[:8]:>8}" for name in CLASSES)
print(header)
for idx, row in enumerate(cm):
    print(f"{CLASSES[idx][:8]:<8} " + " ".join(f"{value:>8}" for value in row))

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
print(f"\nSaved confusion matrix to {CONFUSION_MATRIX_PATH}")

print("\nEvaluation complete.")
