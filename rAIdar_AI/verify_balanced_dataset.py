from pathlib import Path
from PIL import Image
import csv

root = Path("datasets/diverse_balanced")
expected = {
    ("train", "real"): 1600,
    ("train", "fake"): 1600,
    ("valid", "real"): 200,
    ("valid", "fake"): 200,
    ("test", "real"): 200,
    ("test", "fake"): 200,
}

total_valid = 0
total_invalid = 0

for (split, label), expected_count in expected.items():
    folder = root / split / label
    files = [p for p in folder.iterdir() if p.is_file()]
    valid = 0
    invalid = []

    for path in files:
        try:
            with Image.open(path) as img:
                img.verify()
            valid += 1
        except Exception as exc:
            invalid.append((path.name, str(exc)))

    total_valid += valid
    total_invalid += len(invalid)

    print(
        f"{split}/{label}: expected={expected_count}, "
        f"found={len(files)}, valid={valid}, invalid={len(invalid)}"
    )

    for name, error in invalid[:5]:
        print(f"  INVALID: {name} — {error}")

manifest = root / "manifest.csv"
with open(manifest, "r", encoding="utf-8", newline="") as f:
    manifest_count = sum(1 for _ in csv.DictReader(f))

print("\n===== FINAL VALIDATION =====")
print(f"Valid images: {total_valid}")
print(f"Invalid images: {total_invalid}")
print(f"Manifest records: {manifest_count}")
