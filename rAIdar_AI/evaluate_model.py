import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

MODEL_PATH = "deepfake_detector.keras"

TEST_PATH = r"C:\Users\Adrian\OneDrive\ドキュメント\rAIdar_Dataset\real_vs_fake\real-vs-fake\test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


# =========================
# LOAD MODEL
# =========================

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")
print()


# =========================
# GET FILES
# =========================

fake_path = os.path.join(TEST_PATH, "fake")
real_path = os.path.join(TEST_PATH, "real")

fake_files = [
    os.path.join(fake_path, f)
    for f in os.listdir(fake_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

real_files = [
    os.path.join(real_path, f)
    for f in os.listdir(real_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

files = fake_files + real_files

labels = np.array(
    [0] * len(fake_files) +
    [1] * len(real_files)
)

print(f"Fake images: {len(fake_files)}")
print(f"Real images: {len(real_files)}")
print(f"Total images: {len(files)}")
print()


# =========================
# PREDICT IN BATCHES
# =========================

all_predictions = []

total = len(files)

for start in range(0, total, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total)

    batch_files = files[start:end]

    batch_images = []

    for filepath in batch_files:

        try:
            image = Image.open(filepath).convert("RGB")
            image = image.resize(IMAGE_SIZE)

            # IMPORTANT:
            # Do NOT divide by 255.
            # The current model contains its own preprocessing.

            image_array = np.asarray(
                image,
                dtype=np.float32
            )

            batch_images.append(image_array)

        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    batch_images = np.array(
        batch_images,
        dtype=np.float32
    )

    predictions = model.predict(
        batch_images,
        verbose=0
    ).flatten()

    all_predictions.extend(predictions)

    if end % 500 == 0 or end == total:
        print(f"Processed {end}/{total}")


# =========================
# CONVERT PREDICTIONS
# =========================

raw_predictions = np.array(all_predictions)

# 0 = FAKE
# 1 = REAL
#
# >= 0.5 = REAL
# < 0.5 = FAKE

predicted_labels = (
    raw_predictions >= 0.5
).astype(int)


# =========================
# METRICS
# =========================

accuracy = accuracy_score(
    labels,
    predicted_labels
)

precision = precision_score(
    labels,
    predicted_labels,
    pos_label=1
)

recall = recall_score(
    labels,
    predicted_labels,
    pos_label=1
)

f1 = f1_score(
    labels,
    predicted_labels,
    pos_label=1
)

cm = confusion_matrix(
    labels,
    predicted_labels
)


# =========================
# RESULTS
# =========================

results = []

results.append("=" * 60)
results.append("rAIdar MODEL EVALUATION")
results.append("=" * 60)

results.append(f"Total images : {len(labels)}")
results.append(f"Fake images  : {len(fake_files)}")
results.append(f"Real images  : {len(real_files)}")
results.append("")

results.append(f"Accuracy     : {accuracy * 100:.2f}%")
results.append(f"Precision    : {precision * 100:.2f}%")
results.append(f"Recall       : {recall * 100:.2f}%")
results.append(f"F1-score     : {f1 * 100:.2f}%")

results.append("")
results.append("Confusion Matrix")
results.append("----------------")
results.append("                 Predicted")
results.append("              FAKE       REAL")
results.append(
    f"Actual FAKE   {cm[0][0]:5d}      {cm[0][1]:5d}"
)
results.append(
    f"Actual REAL   {cm[1][0]:5d}      {cm[1][1]:5d}"
)

results.append("")
results.append("Classification Report")
results.append("----------------------")

report = classification_report(
    labels,
    predicted_labels,
    target_names=["FAKE", "REAL"],
    digits=4
)

results.append(report)
results.append("=" * 60)

final_results = "\n".join(results)

print()
print(final_results)

with open(
    "evaluation_results.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(final_results)

print()
print("Results saved to evaluation_results.txt")