
from pathlib import Path
from PIL import Image, ImageOps
import imagehash
import csv

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "diverse_balanced"

SPLITS = ["train", "valid", "test"]
CLASSES = ["fake", "real"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Lower distance means more visually similar.
# 6 is a conservative screening threshold.
MAX_DISTANCE = 6

images = []

print("Calculating perceptual hashes...")

for split in SPLITS:
    for class_name in CLASSES:
        folder = DATASET / split / class_name

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        for path in folder.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in EXTENSIONS:
                continue

            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    perceptual_hash = imagehash.phash(img)

                images.append({
                    "split": split,
                    "class": class_name,
                    "path": str(path.relative_to(ROOT)),
                    "hash": perceptual_hash
                })

            except Exception as error:
                print(f"Could not process {path}: {error}")

print(f"Images processed: {len(images)}")
print("Comparing image hashes...")

results = []

for i in range(len(images)):
    for j in range(i + 1, len(images)):
        first = images[i]
        second = images[j]

        distance = first["hash"] - second["hash"]

        # Exclude exact perceptual matches already investigated.
        if 0 < distance <= MAX_DISTANCE:
            results.append({
                "distance": distance,
                "first": first,
                "second": second
            })

results.sort(key=lambda item: item["distance"])

output = ROOT / "near_duplicate_report.csv"

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "distance",
        "first_split",
        "first_class",
        "first_path",
        "second_split",
        "second_class",
        "second_path"
    ])

    for result in results:
        first = result["first"]
        second = result["second"]

        writer.writerow([
            result["distance"],
            first["split"],
            first["class"],
            first["path"],
            second["split"],
            second["class"],
            second["path"]
        ])

print("\n========== RESULTS ==========")
print("Near-duplicate pairs:", len(results))
print("Maximum hash distance:", MAX_DISTANCE)
print("Report saved to:", output)
print("No dataset images or models were modified.")
