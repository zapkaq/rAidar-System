
import csv
import hashlib
from pathlib import Path

from PIL import Image
import imagehash

ROOT = Path(__file__).resolve().parent

AUDIT_CSV = ROOT / "audit_top20_errors.csv"
MANIFEST_CSV = ROOT / "datasets" / "diverse_balanced_cleaned" / "manifest.csv"

HF_CACHE = Path(
    r"C:\Users\Adrian\.cache\huggingface\hub"
    r"\datasets--H1merka--TIGAS_dataset\snapshots"
)

OUTPUT_CSV = ROOT / "top20_original_verification.csv"

# Find the snapshot containing the original test images.
snapshots = sorted(HF_CACHE.glob("*"))

SNAPSHOT_ROOT = next(
    (
        folder for folder in snapshots
        if (folder / "test" / "images").is_dir()
    ),
    None
)

if SNAPSHOT_ROOT is None:
    raise FileNotFoundError(
        "Could not find the original TIGAS test/images folder."
    )

ORIGINAL_ROOT = SNAPSHOT_ROOT / "test" / "images"

print("Original dataset:", ORIGINAL_ROOT)


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def phash(path):
    with Image.open(path) as image:
        return imagehash.phash(image.convert("RGB"))


# Load audit records.
with open(AUDIT_CSV, encoding="utf-8-sig", newline="") as file:
    audit_rows = list(csv.DictReader(file))

# Load manifest records.
with open(MANIFEST_CSV, encoding="utf-8-sig", newline="") as file:
    manifest_rows = list(csv.DictReader(file))

results = []

for index, audit in enumerate(audit_rows, start=1):

    filename = audit["filename"].strip()
    actual_label = audit["actual_label"].strip().lower()

    matching_records = [
        record for record in manifest_rows
        if record["local_split"].lower() == "test"
        and record["label"].lower() == actual_label
        and Path(record["local_path"]).name == filename
    ]

    if len(matching_records) != 1:
        print(
            f"[{index}] {filename}: "
            f"Expected one manifest record, found {len(matching_records)}"
        )
        continue

    record = matching_records[0]

    original_relative = Path(record["original_path"])

    # Original paths begin with images/<source>/<class>/<filename>.
    original_parts = original_relative.parts

    try:
        images_index = original_parts.index("images")
    except ValueError:
        print(f"Invalid original path: {record['original_path']}")
        continue

    original_relative = Path(*original_parts[images_index + 1:])

    original_path = ORIGINAL_ROOT / original_relative

    cleaned_path = ROOT / record["local_path"]

    if not original_path.is_file():
        print(f"[{index}] Original not found: {original_path}")
        continue

    if not cleaned_path.is_file():
        print(f"[{index}] Cleaned copy not found: {cleaned_path}")
        continue

    try:
        original_hash = sha256(original_path)
        cleaned_hash = sha256(cleaned_path)

        original_phash = phash(original_path)
        cleaned_phash = phash(cleaned_path)

        with Image.open(original_path) as image:
            original_dimensions = f"{image.width}x{image.height}"

        with Image.open(cleaned_path) as image:
            cleaned_dimensions = f"{image.width}x{image.height}"

        hash_distance = original_phash - cleaned_phash

        exact_match = original_hash == cleaned_hash

        folder_label = original_relative.parent.name.lower()

        if folder_label == "1_fake":
            original_folder_label = "fake"
        elif folder_label == "0_real":
            original_folder_label = "real"
        else:
            original_folder_label = "unknown"

        result = {
            "filename": filename,
            "source": record["source"],
            "manifest_label": actual_label,
            "original_folder_label": original_folder_label,
            "original_path": str(original_path),
            "cleaned_path": str(cleaned_path),
            "original_sha256": original_hash,
            "cleaned_sha256": cleaned_hash,
            "exact_file_match": exact_match,
            "phash_distance": hash_distance,
            "original_dimensions": original_dimensions,
            "cleaned_dimensions": cleaned_dimensions,
        }

        results.append(result)

        print(
            f"[{index}/{len(audit_rows)}] {filename} | "
            f"Exact match: {exact_match} | "
            f"pHash distance: {hash_distance} | "
            f"Original folder label: {original_folder_label}"
        )

    except Exception as error:
        print(f"Error processing {filename}: {error}")


columns = [
    "filename",
    "source",
    "manifest_label",
    "original_folder_label",
    "original_path",
    "cleaned_path",
    "original_sha256",
    "cleaned_sha256",
    "exact_file_match",
    "phash_distance",
    "original_dimensions",
    "cleaned_dimensions",
]

with open(
    OUTPUT_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as file:

    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(results)


print("\n========== VERIFICATION SUMMARY ==========")
print(f"Audit images: {len(audit_rows)}")
print(f"Successfully verified: {len(results)}")

print(
    "Exact file matches:",
    sum(row["exact_file_match"] for row in results)
)

print(
    "Original folder label disagreements:",
    sum(
        row["original_folder_label"] != row["manifest_label"]
        for row in results
    )
)

print(f"\nSaved results: {OUTPUT_CSV}")
