
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "deepfake_detector_tigas_balanced.keras"
VALID_PATH = ROOT / "datasets" / "diverse_balanced" / "valid"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load original model without changing it
print("Loading original model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Load validation images
print("Loading validation dataset...")

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_PATH,
    labels="inferred",
    label_mode="int",
    class_names=["fake", "real"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Get actual labels
y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in valid_ds
]).astype(int)

# Get model scores
print("Predicting...")
prob_real = model.predict(valid_ds, verbose=1).reshape(-1)

# Test different thresholds
thresholds = np.arange(0.30, 0.701, 0.05)

results = []

print("\nTHRESHOLD ANALYSIS")
print("-" * 85)
print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'FAKE Recall':<14}"
    f"{'REAL Recall':<14}"
    f"{'FAKE F1':<12}"
    f"{'REAL F1':<12}"
)

for threshold in thresholds:
    y_pred = (prob_real >= threshold).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn_fake, fp_fake = cm[0]
    fn_real, tp_real = cm[1]

    accuracy = (tn_fake + tp_real) / len(y_true)

    fake_precision = (
        tn_fake / (tn_fake + fn_real)
        if tn_fake + fn_real else 0
    )

    fake_recall = (
        tn_fake / (tn_fake + fp_fake)
        if tn_fake + fp_fake else 0
    )

    real_precision = (
        tp_real / (tp_real + fp_fake)
        if tp_real + fp_fake else 0
    )

    real_recall = (
        tp_real / (tp_real + fn_real)
        if tp_real + fn_real else 0
    )

    fake_f1 = (
        2 * fake_precision * fake_recall /
        (fake_precision + fake_recall)
        if fake_precision + fake_recall else 0
    )

    real_f1 = (
        2 * real_precision * real_recall /
        (real_precision + real_recall)
        if real_precision + real_recall else 0
    )

    results.append([
        threshold,
        accuracy,
        fake_recall,
        real_recall,
        fake_f1,
        real_f1
    ])

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{fake_recall:<14.4f}"
        f"{real_recall:<14.4f}"
        f"{fake_f1:<12.4f}"
        f"{real_f1:<12.4f}"
    )

# Save results
output_path = ROOT / "threshold_analysis.csv"

np.savetxt(
    output_path,
    np.array(results),
    delimiter=",",
    header="threshold,accuracy,fake_recall,real_recall,fake_f1,real_f1",
    comments="",
    fmt="%.6f"
)

print("\nResults saved to:", output_path)
print("Original model was not modified.")
