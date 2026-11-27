from pathlib import Path
from PIL import Image

root = Path("datasets/pilot/train")
valid = 0
invalid = 0

for category in ["real", "fake"]:
    folder = root / category
    print(f"\n--- {category.upper()} ---")

    for path in sorted(folder.iterdir()):
        try:
            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                print(f"OK: {path.name} | Format: {img.format} | Size: {img.size}")
            valid += 1

        except Exception as error:
            print(f"INVALID: {path.name} | {error}")
            invalid += 1

print("\n--- SUMMARY ---")
print("Valid images:", valid)
print("Invalid images:", invalid)
