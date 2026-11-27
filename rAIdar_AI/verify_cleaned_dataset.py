
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import csv

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "diverse_balanced_cleaned"

SPLITS = ["train", "valid", "test"]
CLASSES = ["fake", "real"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

counts = Counter()
hash_groups = defaultdict(list)
missing_files = []
total = 0

print("Verifying cleaned dataset...")

manifest = DATASET / "manifest.csv"

with manifest.open("r", encoding="utf-8-sig", newline="") as f:
    records = list(csv.DictReader(f))

for row in records:
    path = ROOT / row["local_path"]

    if not path.is_file():
        missing_files.append(row["local_path"])

print("Manifest records:", len(records))
print("Missing manifest files:", len(missing_files))

for split in SPLITS:
    for class_name in CLASSES:
        folder = DATASET / split / class_name

        if not folder.exists():
            print("MISSING FOLDER:", folder)
            continue

        files = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        counts[(split, class_name)] = len(files)
        total += len(files)

        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            hash_groups[digest].append({
                "split": split,
                "class": class_name,
                "path": str(path.relative_to(ROOT))
            })

print("\n========== IMAGE COUNTS ==========")

for split in SPLITS:
    fake = counts[(split, "fake")]
    real = counts[(split, "real")]

    print(
        f"{split.upper()}: "
        f"FAKE={fake}, REAL={real}, TOTAL={fake + real}"
    )

duplicates = [
    group for group in hash_groups.values()
    if len(group) > 1
]

cross_split = [
    group for group in duplicates
    if len({item["split"] for item in group}) > 1
]

print("\n========== INTEGRITY RESULTS ==========")
print("Total images:", total)
print("Unique image contents:", len(hash_groups))
print("Duplicate groups:", len(duplicates))
print("Cross-split duplicate groups:", len(cross_split))
print("Missing files:", len(missing_files))

if missing_files:
    print("\nMissing file examples:")
    for path in missing_files[:10]:
        print(path)

if cross_split:
    print("\nCross-split duplicate examples:")
    for group in cross_split[:10]:
        for item in group:
            print(item)

expected = {
    ("train", "fake"): 1596,
    ("train", "real"): 1600,
    ("valid", "fake"): 200,
    ("valid", "real"): 200,
    ("test", "fake"): 200,
    ("test", "real"): 200
}

counts_correct = all(
    counts[key] == value for key, value in expected.items()
)

manifest_correct = len(records) == total

print("\n========== FINAL CHECK ==========")
print("Expected class counts correct:", counts_correct)
print("Manifest matches image count:", manifest_correct)
print(
    "CLEAN DATASET VERIFIED:",
    counts_correct
    and manifest_correct
    and not missing_files
    and not cross_split
    and total == 3996
)
