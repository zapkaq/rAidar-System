
import csv
from pathlib import Path
from PIL import Image
import imagehash

# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

AUDIT_CSV = ROOT / "audit_top20_errors.csv"

TARGET_ROOT = (
    ROOT / "datasets" / "diverse_balanced_cleaned" / "test"
)

SEARCH_ROOT = ROOT / "datasets" / "diverse"

OUTPUT_CSV = ROOT / "top20_phash_candidates.csv"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

MAX_DISTANCE = 6
TOP_MATCHES = 5


# ==========================================
# IMAGE HASH FUNCTION
# ==========================================

def calculate_hash(image_path):
    try:
        with Image.open(image_path) as img:
            return imagehash.phash(img.convert("RGB"))
    except Exception as error:
        print(f"Warning: Cannot read {image_path}: {error}")
        return None


# ==========================================
# CHECK REQUIRED FILES
# ==========================================

if not AUDIT_CSV.exists():
    raise FileNotFoundError(
        f"Audit CSV not found: {AUDIT_CSV}"
    )

if not TARGET_ROOT.exists():
    raise FileNotFoundError(
        f"Test folder not found: {TARGET_ROOT}"
    )

if not SEARCH_ROOT.exists():
    raise FileNotFoundError(
        f"Search folder not found: {SEARCH_ROOT}"
    )


# ==========================================
# READ TOP 20 MISCLASSIFIED IMAGES
# ==========================================

print("Loading top 20 misclassified images...")

with open(
    AUDIT_CSV,
    "r",
    encoding="utf-8-sig",
    newline=""
) as file:

    reader = csv.DictReader(file)

    audit_rows = list(reader)

    columns = reader.fieldnames or []

print("CSV columns:", columns)

required_columns = {
    "filename",
    "actual_label"
}

if not required_columns.issubset(set(columns)):
    raise ValueError(
        "Missing required columns. "
        "Expected filename and actual_label."
    )

print(f"Loaded {len(audit_rows)} audit records.")


# ==========================================
# SCAN DATASETS/DIVERSE
# ==========================================

print("\nScanning datasets/diverse...")

candidate_paths = sorted(
    path
    for path in SEARCH_ROOT.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

print(f"Found {len(candidate_paths)} image files.")

candidate_hashes = []

for index, path in enumerate(candidate_paths, start=1):

    image_hash = calculate_hash(path)

    if image_hash is not None:
        candidate_hashes.append(
            (path, image_hash)
        )

    if index % 500 == 0:
        print(
            f"Processed {index}/{len(candidate_paths)} images..."
        )

print(
    f"Successfully hashed {len(candidate_hashes)} images."
)


# ==========================================
# COMPARE TARGET IMAGES
# ==========================================

print("\nComparing top 20 images...")

results = []

for index, row in enumerate(audit_rows, start=1):

    filename = row["filename"].strip()

    actual_label = (
        row["actual_label"].strip().lower()
    )

    target_path = (
        TARGET_ROOT / actual_label / filename
    )

    print(
        f"\n[{index}/{len(audit_rows)}] "
        f"Checking {filename}"
    )

    if not target_path.exists():

        print(f"Target image not found: {target_path}")

        results.append({
            "target_filename": filename,
            "actual_label": actual_label,
            "target_path": str(target_path),
            "rank": "",
            "candidate_path": "",
            "candidate_split": "",
            "candidate_label": "",
            "phash_distance": "",
            "assessment": "Target image not found"
        })

        continue

    target_hash = calculate_hash(target_path)

    if target_hash is None:
        continue

    matches = []

    for candidate_path, candidate_hash in candidate_hashes:

        distance = target_hash - candidate_hash

        matches.append(
            (distance, candidate_path)
        )

    matches.sort(key=lambda item: item[0])

    if not matches:
        print("No readable candidate images found.")
        continue

    for rank, (distance, candidate_path) in enumerate(
        matches[:TOP_MATCHES],
        start=1
    ):

        if distance == 0:
            assessment = "Very close visual hash match"

        elif distance <= MAX_DISTANCE:
            assessment = "Possible near match - inspect manually"

        else:
            assessment = "No strong match"

        results.append({

            "target_filename": filename,

            "actual_label": actual_label,

            "target_path": str(
                target_path.relative_to(ROOT)
            ),

            "rank": rank,

            "candidate_path": str(
                candidate_path.relative_to(ROOT)
            ),

            "candidate_split": (
                candidate_path.parent.parent.name
            ),

            "candidate_label": (
                candidate_path.parent.name
            ),

            "phash_distance": distance,

            "assessment": assessment
        })

    print(
        f"Closest match distance: {matches[0][0]}"
    )


# ==========================================
# SAVE RESULTS
# ==========================================

print("\nSaving comparison results...")

output_columns = [
    "target_filename",
    "actual_label",
    "target_path",
    "rank",
    "candidate_path",
    "candidate_split",
    "candidate_label",
    "phash_distance",
    "assessment"
]

with open(
    OUTPUT_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=output_columns
    )

    writer.writeheader()

    writer.writerows(results)


# ==========================================
# FINAL SUMMARY
# ==========================================

possible_matches = sum(
    1
    for result in results
    if result["assessment"]
    == "Possible near match - inspect manually"
    or result["assessment"]
    == "Very close visual hash match"
)

print("\n==========================================")
print("COMPARISON COMPLETED")
print("==========================================")

print(f"Audit images: {len(audit_rows)}")
print(f"Candidate images scanned: {len(candidate_hashes)}")
print(f"Comparison records: {len(results)}")
print(f"Possible matches: {possible_matches}")

print(f"\nOutput file: {OUTPUT_CSV}")

print("\nRemember:")
print("- Distance 0: Very close visual hash match.")
print("- Distance 1-6: Possible near match.")
print("- Distance above 6: No strong match.")
print("- Matches require manual visual verification.")
