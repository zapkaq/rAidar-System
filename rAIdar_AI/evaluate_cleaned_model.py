
import os
import csv
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = "deepfake_detector_tigas_cleaned_v2.keras"

DATASET_PATH = os.path.join(
    "datasets",
    "diverse_balanced_cleaned"
)

TEST_PATH = os.path.join(DATASET_PATH, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
THRESHOLD = 0.50

OUTPUT_CSV = "cleaned_test_predictions_v2.csv"
OUTPUT_REPORT = "cleaned_model_evaluation_v2.txt"

# =====================================================
# CHECK FILES
# =====================================================

print("\n========== FILE VERIFICATION ==========")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.isdir(TEST_PATH):
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_PATH}"
    )

print("Model:", MODEL_PATH)
print("Test dataset:", TEST_PATH)

# =====================================================
# LOAD MODEL
# =====================================================

print("\n========== LOADING MODEL ==========")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

# =====================================================
# LOAD CLEANED TEST DATASET
# =====================================================

print("\n========== LOADING TEST DATASET ==========")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

expected_classes = ["fake", "real"]

print("Classes:", test_ds.class_names)

if test_ds.class_names != expected_classes:
    raise ValueError(
        f"Unexpected class order: {test_ds.class_names}"
    )

# =====================================================
# GET ACTUAL LABELS
# =====================================================

y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in test_ds
]).astype(int)

# =====================================================
# MODEL PREDICTIONS
# =====================================================

print("\n========== RUNNING PREDICTIONS ==========")

probabilities = model.predict(
    test_ds,
    verbose=1
).reshape(-1)

# Model output represents probability of REAL
y_pred = (
    probabilities >= THRESHOLD
).astype(int)

# =====================================================
# CALCULATE PERFORMANCE METRICS
# =====================================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred, zero_division=0
)

recall = recall_score(
    y_true, y_pred, zero_division=0
)

f1 = f1_score(
    y_true, y_pred, zero_division=0
)

auc = roc_auc_score(
    y_true, probabilities
)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["FAKE", "REAL"],
    digits=4,
    zero_division=0
)

# =====================================================
# DISPLAY RESULTS
# =====================================================

results = f"""
=====================================================
       CLEANED DATASET - MODEL V2 EVALUATION
=====================================================

Model: {MODEL_PATH}
Dataset: {TEST_PATH}

Total test images: {len(y_true)}
Threshold: {THRESHOLD}

---------------- OVERALL METRICS ----------------

Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1-score:  {f1:.4f}
ROC-AUC:   {auc:.4f}

---------------- CONFUSION MATRIX ----------------

Rows = Actual
Columns = Predicted

Labels: FAKE (0), REAL (1)

{matrix}

---------------- CLASSIFICATION REPORT ----------------

{report}

=====================================================
                 EVALUATION COMPLETE
=====================================================
"""

print(results)

# =====================================================
# SAVE PREDICTIONS TO CSV
# =====================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "actual_label",
        "predicted_label",
        "probability_real",
        "correct"
    ])

    for actual, predicted, probability in zip(
        y_true,
        y_pred,
        probabilities
    ):

        writer.writerow([
            "REAL" if actual == 1 else "FAKE",
            "REAL" if predicted == 1 else "FAKE",
            f"{probability:.6f}",
            actual == predicted
        ])

# =====================================================
# SAVE EVALUATION REPORT
# =====================================================

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as file:

    file.write(results)

print("Predictions saved:", OUTPUT_CSV)
print("Evaluation report saved:", OUTPUT_REPORT)
