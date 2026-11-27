import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

MODEL_PATH = "deepfake_detector_tigas_balanced.keras"
TEST_PATH = "datasets/diverse_balanced/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("Loading new TIGAS model...")
model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Class order:", test_ds.class_names)

if test_ds.class_names != ["fake", "real"]:
    raise ValueError("Unexpected class order; stopping evaluation.")

y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in test_ds
]).astype(int)

prob_real = model.predict(test_ds, verbose=1).reshape(-1)
y_pred = (prob_real >= 0.5).astype(int)

print("\nConfusion matrix (rows=actual, columns=predicted)")
print("Order: fake, real")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["FAKE", "REAL"],
    digits=4,
    zero_division=0
))

print(f"ROC AUC: {roc_auc_score(y_true, prob_real):.4f}")
print(f"Predicted FAKE: {(y_pred == 0).sum()}")
print(f"Predicted REAL: {(y_pred == 1).sum()}")
