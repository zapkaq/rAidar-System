
import csv
import os
from pathlib import Path
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path("datasets/diverse_balanced_cleaned")
TEST_PATH = DATASET_PATH / "test"
MANIFEST_PATH = DATASET_PATH / "manifest.csv"

PREDICTIONS_CSV = Path("cleaned_test_predictions_v2.csv")

OUTPUT_CSV = Path("source_evaluation_v2.csv")
OUTPUT_REPORT = Path("source_evaluation_v2.txt")

CLASS_NAMES = ["fake", "real"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# ============================================================
# VERIFY FILES
# ============================================================

print("\n========== FILE VERIFICATION ==========")

for path in [TEST_PATH, MANIFEST_PATH, PREDICTIONS_CSV]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

print("Test folder:", TEST_PATH)
print("Manifest:", MANIFEST_PATH)
print("Predictions:", PREDICTIONS_CSV)

# ============================================================
# LOAD MANIFEST
# ============================================================

print("\n========== LOADING MANIFEST ==========")

with open(MANIFEST_PATH, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    manifest_rows = [
        row for row in reader
        if row["local_split"].strip().lower() == "test"
    ]

manifest_by_path = {}

for row in manifest_rows:
    full_path = Path(row["local_path"]).resolve()
    key = os.path.normcase(os.path.normpath(str(full_path)))

    if key in manifest_by_path:
        raise ValueError(f"Duplicate manifest path: {full_path}")

    manifest_by_path[key] = row

print("Test manifest records:", len(manifest_rows))

# ============================================================
# RECONSTRUCT KERAS IMAGE ORDER
# ============================================================

print("\n========== RECONSTRUCTING IMAGE ORDER ==========")

ordered_images = []

for class_name in CLASS_NAMES:

    class_folder = TEST_PATH / class_name

    if not class_folder.is_dir():
        raise FileNotFoundError(
            f"Missing class folder: {class_folder}"
        )

    # Keras loads class folders alphabetically.
    # Filenames here are zero-padded, so alphabetical
    # and natural filename order are consistent.
    files = sorted(
        [
            path for path in class_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower()
    )

    for image_path in files:
        ordered_images.append(image_path)

print("Images found:", len(ordered_images))

# ============================================================
# LOAD EXISTING PREDICTIONS
# ============================================================

print("\n========== LOADING PREDICTIONS ==========")

with open(
    PREDICTIONS_CSV,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    predictions = list(csv.DictReader(file))

print("Prediction records:", len(predictions))

# ============================================================
# VERIFY ALIGNMENT
# ============================================================

print("\n========== VERIFYING ALIGNMENT ==========")

if not (
    len(ordered_images)
    == len(predictions)
    == len(manifest_rows)
):
    raise ValueError(
        "Record count mismatch!\n"
        f"Images: {len(ordered_images)}\n"
        f"Predictions: {len(predictions)}\n"
        f"Manifest: {len(manifest_rows)}"
    )

matched_records = []

for index, (image_path, prediction) in enumerate(
    zip(ordered_images, predictions)
):

    key = os.path.normcase(
        os.path.normpath(str(image_path.resolve()))
    )

    manifest = manifest_by_path.get(key)

    if manifest is None:
        raise ValueError(
            f"Image missing from manifest: {image_path}"
        )

    actual_from_manifest = manifest["label"].strip().upper()
    actual_from_prediction = prediction["actual_label"].strip().upper()

    if actual_from_manifest != actual_from_prediction:
        raise ValueError(
            f"Label mismatch at row {index + 1}:\n"
            f"Image: {image_path}\n"
            f"Manifest label: {actual_from_manifest}\n"
            f"Prediction CSV label: {actual_from_prediction}"
        )

    source = manifest["source"].strip()

    if not source:
        raise ValueError(
            f"Missing source category: {image_path}"
        )

    matched_records.append({
        "image": str(image_path),
        "source": source,
        "actual": 0 if actual_from_manifest == "FAKE" else 1,
        "predicted": (
            0 if prediction["predicted_label"].strip().upper() == "FAKE"
            else 1
        ),
        "probability_real": float(prediction["probability_real"])
    })

print("Matched records:", len(matched_records))
print("All image paths verified.")
print("All actual labels verified.")
print("Prediction alignment verified.")

# ============================================================
# GROUP BY SOURCE
# ============================================================

print("\n========== SOURCE-LEVEL EVALUATION ==========")

groups = defaultdict(list)

for record in matched_records:
    groups[record["source"]].append(record)

results = []

for source, records in groups.items():

    y_true = [record["actual"] for record in records]
    y_pred = [record["predicted"] for record in records]
    probabilities = [
        record["probability_real"] for record in records
    ]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    fake_count = y_true.count(0)
    real_count = y_true.count(1)

    fake_precision = precision_score(
        y_true, y_pred, pos_label=0, zero_division=0
    )

    fake_recall = recall_score(
        y_true, y_pred, pos_label=0, zero_division=0
    )

    fake_f1 = f1_score(
        y_true, y_pred, pos_label=0, zero_division=0
    )

    real_precision = precision_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    real_recall = recall_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    real_f1 = f1_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    auc = (
        roc_auc_score(y_true, probabilities)
        if len(set(y_true)) == 2
        else None
    )

    result = {
        "source": source,
        "total_images": len(records),
        "fake_images": fake_count,
        "real_images": real_count,
        "correct": int(sum(
            actual == predicted
            for actual, predicted in zip(y_true, y_pred)
        )),
        "accuracy": accuracy_score(y_true, y_pred),
        "fake_precision": fake_precision,
        "fake_recall": fake_recall,
        "fake_f1": fake_f1,
        "real_precision": real_precision,
        "real_recall": real_recall,
        "real_f1": real_f1,
        "roc_auc": auc,
        "fake_correct": int(matrix[0, 0]),
        "fake_predicted_real": int(matrix[0, 1]),
        "real_predicted_fake": int(matrix[1, 0]),
        "real_correct": int(matrix[1, 1])
    }

    results.append(result)

# Lowest accuracy first to make weak sources easier to identify.
results.sort(key=lambda item: item["accuracy"])

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nSOURCE PERFORMANCE")
print("-" * 100)

for result in results:
    print(
        f"{result['source']:15} "
        f"Total: {result['total_images']:3} | "
        f"Fake: {result['fake_images']:3} | "
        f"Real: {result['real_images']:3} | "
        f"Accuracy: {result['accuracy'] * 100:6.2f}% | "
        f"Fake Recall: {result['fake_recall'] * 100:6.2f}% | "
        f"Real Recall: {result['real_recall'] * 100:6.2f}%"
    )

# ============================================================
# SAVE CSV REPORT
# ============================================================

fieldnames = list(results[0].keys()) if results else []

with open(
    OUTPUT_CSV,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for result in results:
        writer.writerow(result)

# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_lines = [
    "============================================================",
    "       rAIdar - SOURCE-BY-SOURCE MODEL EVALUATION",
    "============================================================",
    "",
    f"Model: deepfake_detector_tigas_cleaned_v2.keras",
    f"Dataset: {DATASET_PATH}",
    f"Total images: {len(matched_records)}",
    f"Number of sources: {len(results)}",
    "Decision threshold: 0.50",
    "",
    "CONFUSION MATRIX LEGEND:",
    "Fake correct = Actual FAKE predicted FAKE",
    "Fake predicted real = Actual FAKE predicted REAL",
    "Real predicted fake = Actual REAL predicted FAKE",
    "Real correct = Actual REAL predicted REAL",
    "",
    "SOURCE RESULTS",
    "-" * 100
]

for result in results:

    auc_text = (
        f"{result['roc_auc']:.4f}"
        if result["roc_auc"] is not None
        else "N/A"
    )

    report_lines.extend([
        "",
        f"SOURCE: {result['source']}",
        f"Total images: {result['total_images']}",
        f"Fake images: {result['fake_images']}",
        f"Real images: {result['real_images']}",
        f"Correct predictions: {result['correct']}",
        f"Accuracy: {result['accuracy']:.4f}",
        f"Fake precision: {result['fake_precision']:.4f}",
        f"Fake recall: {result['fake_recall']:.4f}",
        f"Fake F1: {result['fake_f1']:.4f}",
        f"Real precision: {result['real_precision']:.4f}",
        f"Real recall: {result['real_recall']:.4f}",
        f"Real F1: {result['real_f1']:.4f}",
        f"ROC-AUC: {auc_text}",
        "",
        "Confusion matrix:",
        (
            f"[[{result['fake_correct']}, "
            f"{result['fake_predicted_real']}],"
        ),
        (
            f" [{result['real_predicted_fake']}, "
            f"{result['real_correct']}]]"
        ),
        "-" * 70
    ])

if results:
    best = max(results, key=lambda item: item["accuracy"])
    worst = min(results, key=lambda item: item["accuracy"])

    report_lines.extend([
        "",
        "BEST-PERFORMING SOURCE",
        f"{best['source']}: {best['accuracy'] * 100:.2f}%",
        "",
        "LOWEST-PERFORMING SOURCE",
        f"{worst['source']}: {worst['accuracy'] * 100:.2f}%",
    ])

report_lines.extend([
    "",
    "============================================================",
    "                 EVALUATION COMPLETE",
    "============================================================"
])

with open(OUTPUT_REPORT, "w", encoding="utf-8") as file:
    file.write("\n".join(report_lines))

print("\n========== EVALUATION COMPLETE ==========")
print("Sources evaluated:", len(results))
print("CSV report:", OUTPUT_CSV)
print("Text report:", OUTPUT_REPORT)
