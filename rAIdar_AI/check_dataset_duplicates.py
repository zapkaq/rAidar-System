
from pathlib import Path
from collections import defaultdict
import hashlib
import csv

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "diverse_balanced"

SPLITS = ["train", "valid", "test"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

hash_groups = defaultdict(list)

print("Scanning dataset for exact duplicate images...")

for split in SPLITS:
    for class_name in ["fake", "real"]:
        folder = DATASET / split / class_name

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        for file in folder.rglob("*"):
            if not file.is_file():
                continue

            if file.suffix.lower() not in EXTENSIONS:
                continue

            digest = hashlib.sha256(file.read_bytes()).hexdigest()

            hash_groups[digest].append({
                "split": split,
                "class": class_name,
                "path": str(file.relative_to(ROOT))
            })

duplicates = [
    files for files in hash_groups.values()
    if len(files) > 1
]

cross_split = []
same_split = []

for group in duplicates:
    splits = {item["split"] for item in group}

    if len(splits) > 1:
        cross_split.append(group)
    else:
        same_split.append(group)

print("\n========== RESULTS ==========")
print("Unique image contents:", len(hash_groups))
print("Duplicate groups:", len(duplicates))
print("Duplicates across splits:", len(cross_split))
print("Duplicates within same split:", len(same_split))

output = ROOT / "duplicate_report.csv"

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "class", "path", "duplicate_group"])

    for group_number, group in enumerate(duplicates, start=1):
        for item in group:
            writer.writerow([
                item["split"],
                item["class"],
                item["path"],
                group_number
            ])

print("\nReport saved to:", output)
print("Original model and dataset were not modified.")
