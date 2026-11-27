
import csv
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CSV_FILE = Path("audit_top20_errors.csv")
OUTPUT_DIR = Path("top20_audit_images")

OUTPUT_DIR.mkdir(exist_ok=True)

with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as f:
    data = list(csv.DictReader(f))

false_negatives = [
    x for x in data if x["error_type"] == "FALSE_NEGATIVE"
]

false_positives = [
    x for x in data if x["error_type"] == "FALSE_POSITIVE"
]

def make_sheet(images, filename, title):

    columns = 2
    cell_width = 520
    cell_height = 310
    rows = math.ceil(len(images) / columns)

    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height + 60),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 19)
        title_font = ImageFont.truetype("arial.ttf", 25)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    draw.text((15, 15), title, fill="black", font=title_font)

    for i, item in enumerate(images):

        path = Path(item["image"])

        if not path.exists():
            print("Missing image:", path)
            continue

        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((480, 220))

            x = (i % columns) * cell_width
            y = 60 + (i // columns) * cell_height

            image_x = x + (cell_width - image.width) // 2

            sheet.paste(image, (image_x, y))

            details = [
                f"Filename: {item['filename']}",
                f"Source: {item['source']}",
                f"Actual: {item['actual_label']}",
                f"Predicted: {item['predicted_label']}",
                f"REAL score: {float(item['probability_real']):.2%}"
            ]

            for j, line in enumerate(details):
                draw.text(
                    (x + 12, y + 225 + j * 17),
                    line,
                    fill="black",
                    font=font
                )

        except Exception as e:
            print("Error:", path, e)

    output = OUTPUT_DIR / filename
    sheet.save(output, quality=95)

    print("Saved:", output)


make_sheet(
    false_negatives,
    "top10_false_negatives.jpg",
    "TOP 10 FAKE IMAGES PREDICTED AS REAL"
)

make_sheet(
    false_positives,
    "top10_false_positives.jpg",
    "TOP 10 REAL IMAGES PREDICTED AS FAKE"
)

print("\nContact sheets completed.")
