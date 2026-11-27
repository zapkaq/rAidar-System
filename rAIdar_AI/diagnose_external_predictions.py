from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# Configuration
TEST_PATH = Path("external_test")
MODEL_PATH = "deepfake_detector_tigas_balanced.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
THRESHOLD = 0.5

print("Loading external test images...")

# Load images using the same preprocessing as the existing evaluation.
test_ds = tf.keras.utils.image_dataset_from_directory(
    str(TEST_PATH),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = test_ds.class_names

if class_names != ["fake", "real"]:
    raise ValueError(f"Unexpected class order: {class_names}")

# Reconstruct the sorted file order used by the dataset loader.
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

file_paths = []
actual_labels = []

for class_index, class_name in enumerate(class_names):
    class_dir = TEST_PATH / class_name

    paths = sorted(
        (
            path for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in valid_extensions
        ),
        key=lambda path: str(path.relative_to(class_dir)).lower()
    )

    file_paths.extend(paths)
    actual_labels.extend([class_index] * len(paths))

actual_labels = np.array(actual_labels, dtype=int)

if len(file_paths) != len(actual_labels):
    raise ValueError("Image filename and label counts do not match.")

if len(file_paths) != sum(1 for _ in test_ds.unbatch()):
    raise ValueError("Image count does not match the loaded dataset.")

print(f"Total images: {len(file_paths)}")
print(f"Loading model: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

prob_real = model.predict(test_ds, verbose=1).reshape(-1)

if len(prob_real) != len(file_paths):
    raise ValueError("Prediction count does not match filename count.")

predicted_labels = (prob_real >= THRESHOLD).astype(int)

print("\n" + "=" * 100)
print("INDIVIDUAL IMAGE PREDICTIONS")
print("=" * 100)

print(
    f"{'ACTUAL':<8} {'PREDICTED':<10} "
    f"{'REAL SCORE':<12} FILENAME"
)
print("-" * 100)

for path, actual, predicted, score in zip(
    file_paths, actual_labels, predicted_labels, prob_real
):
    actual_name = class_names[actual].upper()
    predicted_name = class_names[predicted].upper()

    print(
        f"{actual_name:<8} {predicted_name:<10} "
        f"{score:<12.4f} {path.name}"
    )

print("\n" + "=" * 100)
print("MISCLASSIFIED IMAGES")
print("=" * 100)

errors = 0

for path, actual, predicted, score in zip(
    file_paths, actual_labels, predicted_labels, prob_real
):
    if actual != predicted:
        errors += 1
        print(
            f"Actual: {class_names[actual].upper():<5} | "
            f"Predicted: {class_names[predicted].upper():<5} | "
            f"REAL score: {score:.4f} | "
            f"File: {path.name}"
        )

if errors == 0:
    print("No misclassified images.")

print(f"\nTotal misclassified: {errors} / {len(actual_labels)}")

print("\nConfusion matrix (rows=actual, columns=predicted)")
print("Order: FAKE, REAL")
print(confusion_matrix(actual_labels, predicted_labels, labels=[0, 1]))

print("\nDiagnostic complete. No model was trained or modified.")