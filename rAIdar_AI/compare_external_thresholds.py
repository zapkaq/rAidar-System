import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score
)

MODEL_PATH = "deepfake_detector_tigas_balanced.keras"
TEST_PATH = "external_test"

model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=(224, 224),
    batch_size=8,
    label_mode="binary",
    shuffle=False
)

if test_ds.class_names != ["fake", "real"]:
    raise ValueError("Unexpected class order. Stopping.")

y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in test_ds
]).astype(int)

prob_real = model.predict(test_ds, verbose=1).reshape(-1)

print("\nEXTERNAL TEST: THRESHOLD COMPARISON")
print(f"Images: {len(y_true)}")
print(f"ROC AUC (threshold-independent): {roc_auc_score(y_true, prob_real):.4f}")

for threshold in [0.50, 0.60]:
    y_pred = (prob_real >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fake_recall = matrix[0, 0] / matrix[0].sum()
    real_recall = matrix[1, 1] / matrix[1].sum()

    print(f"\nThreshold: {threshold:.2f}")
    print("Confusion matrix (rows=actual, columns=predicted)")
    print("Order: FAKE, REAL")
    print(matrix)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"Fake recall: {fake_recall:.4f}")
    print(f"Real recall: {real_recall:.4f}")

print("\nEvaluation complete. No model or website files changed.")
