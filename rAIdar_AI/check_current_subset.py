from pathlib import Path
from collections import Counter

root = Path("datasets/diverse")

for split in ["train", "valid", "test"]:
    print(f"\n===== {split.upper()} CURRENT SUBSET =====")

    for label in ["real", "fake"]:
        folder = root / split / label
        files = [p for p in folder.iterdir() if p.is_file()]
        print(f"{label}: {len(files)} images")
