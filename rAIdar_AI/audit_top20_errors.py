
import csv
from pathlib import Path

INPUT_FILE = Path("misclassified_images_v2.csv")
OUTPUT_FILE = Path("audit_top20_errors.csv")

with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
    errors = list(csv.DictReader(f))

false_negatives = [
    row for row in errors
    if row["error_type"] == "FALSE_NEGATIVE"
]

false_positives = [
    row for row in errors
    if row["error_type"] == "FALSE_POSITIVE"
]

# Fake predicted as REAL:
# Highest REAL score means strongest incorrect REAL prediction.
false_negatives.sort(
    key=lambda row: float(row["probability_real"]),
    reverse=True
)

# Real predicted as FAKE:
# Lowest REAL score means strongest incorrect FAKE prediction.
false_positives.sort(
    key=lambda row: float(row["probability_real"])
)

selected = (
    false_negatives[:10] +
    false_positives[:10]
)

fields = [
    "image",
    "filename",
    "source",
    "actual_label",
    "predicted_label",
    "probability_real",
    "error_type"
]

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(selected)

print("\n========== TOP 20 ERROR AUDIT ==========")

print("\nTOP 10 FAKE IMAGES PREDICTED AS REAL")

for row in false_negatives[:10]:
    print(
        row["filename"],
        "|", row["source"],
        "| REAL score:", row["probability_real"]
    )

print("\nTOP 10 REAL IMAGES PREDICTED AS FAKE")

for row in false_positives[:10]:
    print(
        row["filename"],
        "|", row["source"],
        "| REAL score:", row["probability_real"]
    )

print("\nAudit file saved:", OUTPUT_FILE)
print("Total selected:", len(selected))
