
import csv
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent

MANIFEST = ROOT / "datasets" / "diverse_balanced_cleaned" / "manifest.csv"

HF_CACHE = Path(
    r"C:\Users\Adrian\.cache\huggingface\hub"
    r"\datasets--H1merka--TIGAS_dataset\snapshots"
)

OUTPUT = ROOT / "all_test_original_verification.csv"

# Find original TIGAS snapshot
snapshot = next(
    (
        folder for folder in HF_CACHE.iterdir()
        if (folder / "test" / "images").is_dir()
    ),
    None
)

if snapshot is None:
    raise FileNotFoundError("Original TIGAS test/images folder not found.")

ORIGINAL_ROOT = snapshot / "test" / "images"


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# Read test manifest
with open(MANIFEST, encoding="utf-8-sig", newline="") as file:
    manifest = list(csv.DictReader(file))

test_records = [
    row for row in manifest
    if row["local_split"].lower() == "test"
]

print("Original dataset:", ORIGINAL_ROOT)
print("Test manifest records:", len(test_records))
print("\nVerifying original files...\n")

results = []
original_hashes = Counter()
original_paths = Counter()

for index, record in enumerate(test_records, start=1):

    filename = Path(record["local_path"]).name
    label = record["label"].lower()
    source = record["source"]

    local_path = ROOT / record["local_path"]

    original_parts = Path(record["original_path"]).parts

    try:
        images_index = original_parts.index("images")
    except ValueError:
        print(f"Invalid original path: {filename}")
        continue

    relative_path = Path(*original_parts[images_index + 1:])
    original_path = ORIGINAL_ROOT / relative_path

    result = {
        "filename": filename,
        "source": source,
        "manifest_label": label,
        "original_folder_label": "",
        "original_path": str(original_path),
        "local_path": str(local_path),
        "original_exists": original_path.is_file(),
        "local_exists": local_path.is_file(),
        "exact_hash_match": False,
        "label_match": False,
        "original_sha256": "",
        "error": ""
    }

    if not original_path.is_file():
        result["error"] = "Original file missing"

    elif not local_path.is_file():
        result["error"] = "Cleaned file missing"

    else:
        folder_label = relative_path.parent.name.lower()

        if folder_label == "1_fake":
            original_label = "fake"
        elif folder_label == "0_real":
            original_label = "real"
        else:
            original_label = "unknown"

        result["original_folder_label"] = original_label
        result["label_match"] = original_label == label

        original_digest = sha256(original_path)
        local_digest = sha256(local_path)

        result["original_sha256"] = original_digest
        result["exact_hash_match"] = (
            original_digest == local_digest
        )

        original_hashes[original_digest] += 1
        original_paths[str(original_path).lower()] += 1

    results.append(result)

    if index % 50 == 0:
        print(f"Verified {index}/{len(test_records)}")

# Save detailed results
columns = list(results[0].keys()) if results else []

with open(OUTPUT, "w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(results)

# Summary
print("\n========== FINAL VERIFICATION ==========")

print("Test records:", len(test_records))
print("Processed records:", len(results))

print(
    "Original files found:",
    sum(row["original_exists"] for row in results)
)

print(
    "Cleaned files found:",
    sum(row["local_exists"] for row in results)
)

print(
    "Exact hash matches:",
    sum(row["exact_hash_match"] for row in results)
)

print(
    "Label matches:",
    sum(row["label_match"] for row in results)
)

print(
    "Label disagreements:",
    sum(
        row["original_exists"] and not row["label_match"]
        for row in results
    )
)

print(
    "Missing original files:",
    sum(not row["original_exists"] for row in results)
)

print(
    "Missing cleaned files:",
    sum(not row["local_exists"] for row in results)
)

print(
    "Duplicate original paths:",
    sum(count > 1 for count in original_paths.values())
)

print(
    "Duplicate original image contents:",
    sum(count > 1 for count in original_hashes.values())
)

print("\nResults saved to:", OUTPUT)
