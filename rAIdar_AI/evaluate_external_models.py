import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

TEST_PATH = "external_test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

MODELS = {
    "Previous retrained model": "deepfake_detector_retrained.keras",
    "New TIGAS model": "deepfake_detector_tigas_balanced.keras",
}

print("Loading independent external test images...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Class order:", test_ds.class_names)

if test_ds.class_names != ["fake", "real"]:
    raise ValueError("Unexpected class order. Evaluation stopped.")

y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in test_ds
]).astype(int)

print(f"Total images: {len(y_true)}")
print(f"Fake images: {(y_true == 0).sum()}")
print(f"Real images: {(y_true == 1).sum()}")

for model_name, model_path in MODELS.items():
    print("\n" + "=" * 60)
    print(model_name)
    print("File:", model_path)
    print("=" * 60)

    model = tf.keras.models.load_model(model_path)

    prob_real = model.predict(test_ds, verbose=1).reshape(-1)
    y_pred = (prob_real >= 0.5).astype(int)

    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_true, prob_real):.4f}")

    print("\nConfusion matrix (rows=actual, columns=predicted)")
    print("Order: FAKE, REAL")
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

    print("Mean probability of REAL:")
    print(f"  Actual fake images: {prob_real[y_true == 0].mean():.4f}")
    print(f"  Actual real images: {prob_real[y_true == 1].mean():.4f}")

    print(f"Predicted FAKE: {(y_pred == 0).sum()}")
    print(f"Predicted REAL: {(y_pred == 1).sum()}")

    del model

print("\nExternal evaluation completed.")
print("No model was trained or modified.")
