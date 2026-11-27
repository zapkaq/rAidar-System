
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent

files = [
    ROOT / "datasets/diverse_balanced/train/real/01507.png",
    ROOT / "datasets/diverse_balanced/valid/real/00185.png"
]

labels = [
    "TRAIN - REAL - 01507.png",
    "VALIDATION - REAL - 00185.png"
]

width = 600
height = 500

sheet = Image.new("RGB", (width * 2, height + 60), "white")
draw = ImageDraw.Draw(sheet)

for index, path in enumerate(files):
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        original_size = img.size
        img.thumbnail((width - 30, height - 30))

        x = index * width
        y = 10

        sheet.paste(img, (x + 15, y + 15))

        draw.text(
            (x + 15, height + 10),
            labels[index],
            fill="black"
        )

        print(f"{path.name}: {original_size}")

output = ROOT / "near_duplicate_comparison.jpg"
sheet.save(output, quality=95)

print("\nComparison saved:", output)
