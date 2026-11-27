
import csv
from pathlib import Path
from PIL import Image

DATASET = Path("datasets/diverse_balanced_cleaned")
MANIFEST = DATASET / "manifest.csv"
AUDIT = Path("audit_top20_errors.csv")
OUTPUT = Path("top20_label_verification.csv")

def normalize(path):
    return Path(path).as_posix().replace("\\", "/").lstrip("./")

# Load manifest
with open(MANIFEST, "r", encoding="utf-8-sig", newline="") as f:
    manifest = list(csv.DictReader(f))

manifest_map = {
    normalize(row["local_path"]): row
    for row in manifest
}

# Load selected errors
with open(AUDIT, "r", encoding="utf-8-sig", newline="") as f:
    audit_rows = list(csv.DictReader(f))

results = []

for item in audit_rows:

    image_path = Path(item["image"])
    key = normalize(image_path)

    record = manifest_map.get(key)

    if record is None:
        print("Manifest record missing:", image_path)
        continue

    if not image_path.exists():
        print("Image missing:", image_path)
        continue

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            image_format = img.format
            mode = img.mode
            img.verify()

        results.append({
            "filename": item["filename"],
            "source": item["source"],
            "prediction_actual_label": item["actual_label"],
            "manifest_label": record["label"],
            "labels_match": (
                item["actual_label"].upper()
                == record["label"].upper()
            ),
            "original_path": record["original_path"],
            "local_path": record["local_path"],
            "width": width,
            "height": height,
            "format": image_format,
            "mode": mode,
            "error_type": item["error_type"],
            "probability_real": item["probability_real"]
        })

    except Exception as e:
        print("Image verification error:", image_path, e)

fields = [
    "filename",
    "source",
    "prediction_actual_label",
    "manifest_label",
    "labels_match",
    "original_path",
    "local_path",
    "width",
    "height",
    "format",
    "mode",
    "error_type",
    "probability_real"
]

with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print("\n========== LABEL VERIFICATION ==========")
print("Selected images:", len(audit_rows))
print("Successfully inspected:", len(results))
print(
    "Matching labels:",
    sum(x["labels_match"] for x in results)
)
print(
    "Mismatching labels:",
    sum(not x["labels_match"] for x in results)
)
print("Report saved:", OUTPUT)
