
import hashlib
from pathlib import Path

DATASETS = {
    "diverse": Path("datasets/diverse"),
    "diverse_balanced": Path("datasets/diverse_balanced"),
    "diverse_balanced_cleaned": Path("datasets/diverse_balanced_cleaned")
}

def get_hashes(folder):
    results = {}

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }:
            continue

        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            results.setdefault(digest, []).append(path)
        except OSError as e:
            print("Could not read:", path, e)

    return results

hash_data = {}

for name, folder in DATASETS.items():
    print(f"\nScanning {name}...")
    hash_data[name] = get_hashes(folder)
    print("Unique image contents:", len(hash_data[name]))

print("\n========== DATASET COMPARISON ==========")

for first, second in [
    ("diverse", "diverse_balanced"),
    ("diverse_balanced", "diverse_balanced_cleaned"),
    ("diverse", "diverse_balanced_cleaned")
]:
    shared = (
        set(hash_data[first].keys())
        & set(hash_data[second].keys())
    )

    print(f"\n{first} vs {second}")
    print("Matching image contents:", len(shared))

    for digest in list(shared)[:3]:
        print("Example:")
        print(" ", hash_data[first][digest][0])
        print(" ", hash_data[second][digest][0])

print("\nComparison completed.")
