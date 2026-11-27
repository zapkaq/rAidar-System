
import csv
from pathlib import Path

DATASET_PATH = Path("datasets/diverse_balanced_cleaned")
MANIFEST_PATH = DATASET_PATH / "manifest.csv"
PREDICTIONS_PATH = Path("cleaned_test_predictions_v2.csv")

OUTPUT_CSV = Path("misclassified_images_v2.csv")

# Load test manifest
with open(MANIFEST_PATH, "r", encoding="utf-8-sig", newline="") as f:
    manifest = list(csv.DictReader(f))

manifest = [
    row for row in manifest
    if row["local_split"].strip().lower() == "test"
]

# Index manifest using local image path
manifest_by_path = {
    Path(row["local_path"]).as_posix(): row
    for row in manifest
}

# Reconstruct Keras test image order
image_paths = []

for class_name in ["fake", "real"]:
    folder = DATASET_PATH / "test" / class_name

    image_paths.extend(sorted(
        [
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".gif"
            }
        ],
        key=lambda p: p.name.lower()
    ))

# Load predictions
with open(PREDICTIONS_PATH, "r", encoding="utf-8-sig", newline="") as f:
    predictions = list(csv.DictReader(f))

if len(image_paths) != len(predictions):
    raise ValueError("Image and prediction counts do not match.")

misclassified = []

for image_path, prediction in zip(image_paths, predictions):

    relative_path = image_path.as_posix()
    manifest_row = manifest_by_path.get(relative_path)

    if manifest_row is None:
        raise ValueError(f"Missing manifest record: {relative_path}")

    actual = prediction["actual_label"].upper()
    predicted = prediction["predicted_label"].upper()

    if actual != predicted:

        misclassified.append({
            "image": str(image_path),
            "filename": image_path.name,
            "source": manifest_row["source"],
            "actual_label": actual,
            "predicted_label": predicted,
            "probability_real": prediction["probability_real"],
            "error_type": (
                "FALSE_NEGATIVE"
                if actual == "FAKE" and predicted == "REAL"
                else "FALSE_POSITIVE"
            )
        })

# Save report
with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:

    fields = [
        "image",
        "filename",
        "source",
        "actual_label",
        "predicted_label",
        "probability_real",
        "error_type"
    ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(misclassified)

print("\n========== MISCLASSIFICATION ANALYSIS ==========")

print("Total test images:", len(predictions))
print("Correct predictions:", len(predictions) - len(misclassified))
print("Misclassified images:", len(misclassified))

print("\nERROR BREAKDOWN:")

false_negatives = [
    x for x in misclassified
    if x["error_type"] == "FALSE_NEGATIVE"
]

false_positives = [
    x for x in misclassified
    if x["error_type"] == "FALSE_POSITIVE"
]

print("Fake classified as REAL:", len(false_negatives))
print("Real classified as FAKE:", len(false_positives))

print("\nERRORS BY SOURCE:")

source_counts = {}

for item in misclassified:
    source_counts[item["source"]] = (
        source_counts.get(item["source"], 0) + 1
    )

for source, count in sorted(
    source_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{source:15} {count} errors")

print("\nReport saved:", OUTPUT_CSV)
