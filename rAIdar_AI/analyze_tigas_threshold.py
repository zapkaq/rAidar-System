import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)

MODEL_PATH = "deepfake_detector_tigas_balanced.keras"
VALID_PATH = "datasets/diverse_balanced/valid"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

print("Loading TIGAS model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading validation images...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

if val_ds.class_names != ["fake", "real"]:
    raise ValueError("Unexpected class order. Stopping.")

y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in val_ds
]).astype(int)

prob_real = model.predict(val_ds, verbose=1).reshape(-1)

print(f"\nValidation images: {len(y_true)}")
print(f"ROC AUC: {roc_auc_score(y_true, prob_real):.4f}")
print("\nThreshold comparison:")
print("Threshold | Accuracy | Balanced accuracy | Fake recall | Real recall")

results = []

for threshold in np.arange(0.10, 0.91, 0.05):
    y_pred = (prob_real >= threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fake_correct = matrix[0, 0]
    fake_total = matrix[0].sum()
    real_correct = matrix[1, 1]
    real_total = matrix[1].sum()

    fake_recall = fake_correct / fake_total if fake_total else 0
    real_recall = real_correct / real_total if real_total else 0

    accuracy = accuracy_score(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)

    results.append((balanced, threshold, accuracy, fake_recall, real_recall))

    print(
        f"{threshold:9.2f} | {accuracy:8.4f} | "
        f"{balanced:17.4f} | {fake_recall:11.4f} | "
        f"{real_recall:11.4f}"
    )

best = max(results, key=lambda row: row[0])

print("\nBest validation threshold by balanced accuracy:")
print(f"Threshold: {best[1]:.2f}")
print(f"Accuracy: {best[2]:.4f}")
print(f"Balanced accuracy: {best[0]:.4f}")
print(f"Fake recall: {best[3]:.4f}")
print(f"Real recall: {best[4]:.4f}")

print("\nNo model weights or website files were changed.")
