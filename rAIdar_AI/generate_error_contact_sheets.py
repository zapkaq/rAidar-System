
import csv
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CSV_PATH = Path("misclassified_images_v2.csv")
OUTPUT_DIR = Path("misclassified_contact_sheets")

THUMB_WIDTH = 180
THUMB_HEIGHT = 160
CELL_WIDTH = 220
CELL_HEIGHT = 220
COLUMNS = 4

OUTPUT_DIR.mkdir(exist_ok=True)

with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

groups = {
    "FALSE_NEGATIVE": [],
    "FALSE_POSITIVE": []
}

for row in rows:
    groups[row["error_type"]].append(row)

try:
    font = ImageFont.truetype("arial.ttf", 12)
except OSError:
    font = ImageFont.load_default()


def create_contact_sheet(error_type, images):

    if not images:
        print(f"No images found for {error_type}")
        return

    rows_needed = math.ceil(len(images) / COLUMNS)

    sheet = Image.new(
        "RGB",
        (COLUMNS * CELL_WIDTH, rows_needed * CELL_HEIGHT),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, item in enumerate(images):

        image_path = Path(item["image"])

        if not image_path.exists():
            print("Missing:", image_path)
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((THUMB_WIDTH, THUMB_HEIGHT))

            x = (index % COLUMNS) * CELL_WIDTH
            y = (index // COLUMNS) * CELL_HEIGHT

            image_x = x + (CELL_WIDTH - image.width) // 2

            sheet.paste(image, (image_x, y))

            probability = float(item["probability_real"])

            information = [
                item["filename"],
                f"Source: {item['source']}",
                f"Actual: {item['actual_label']}",
                f"Predicted: {item['predicted_label']}",
                f"REAL score: {probability:.2%}"
            ]

            for line_number, line in enumerate(information):
                draw.text(
                    (x + 5, y + THUMB_HEIGHT + 5 + line_number * 14),
                    line,
                    fill="black",
                    font=font
                )

        except Exception as error:
            print(f"Could not process {image_path}: {error}")

    output_path = OUTPUT_DIR / f"{error_type.lower()}.jpg"

    sheet.save(output_path, quality=95)

    print(f"Saved: {output_path}")
    print(f"Images in group: {len(images)}")


create_contact_sheet(
    "FALSE_NEGATIVE",
    groups["FALSE_NEGATIVE"]
)

create_contact_sheet(
    "FALSE_POSITIVE",
    groups["FALSE_POSITIVE"]
)

print("\nContact sheet generation completed.")
