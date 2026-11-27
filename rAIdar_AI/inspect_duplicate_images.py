
from pathlib import Path
from PIL import Image, ImageDraw
import csv

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "duplicate_report.csv"

items = []

with REPORT.open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        items.append(row)

thumb_width = 240
thumb_height = 220
label_height = 55
columns = 2
rows = len(items)

sheet = Image.new(
    "RGB",
    (columns * thumb_width, rows * (thumb_height + label_height)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for index, item in enumerate(items):
    path = ROOT / item["path"]

    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((thumb_width - 20, thumb_height - 20))

    x = (index % columns) * thumb_width
    y = (index // columns) * (thumb_height + label_height)

    sheet.paste(image, (x + 10, y + 10))

    label = (
        f"Group {item['duplicate_group']} | "
        f"{item['split']} | {item['class']}"
    )

    draw.text((x + 10, y + thumb_height), label, fill="black")
    draw.text(
        (x + 10, y + thumb_height + 20),
        path.name,
        fill="black"
    )

output = ROOT / "duplicate_contact_sheet.jpg"
sheet.save(output, quality=95)

print("Contact sheet saved:", output)
