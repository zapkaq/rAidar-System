from PIL import Image, ImageOps, ImageDraw
from pathlib import Path
import random

original_dir = Path(
    r"C:\Users\Adrian\OneDrive\ドキュメント\rAIdar_Dataset"
    r"\real_vs_fake\real-vs-fake\test\fake"
)

external_dir = Path(
    r"C:\Users\Adrian\source\repos\rAIdar_AI\external_test\fake"
)

output = Path("actual_fake_comparison.jpg")

def get_images(folder):
    return sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ])

original_files = get_images(original_dir)
external_files = get_images(external_dir)

random.seed(42)

original_sample = random.sample(
    original_files,
    min(20, len(original_files))
)

external_sample = external_files[:20]

thumb_size = (220, 180)
cols = 4
rows = 5
label_height = 25
margin = 10
header_height = 50

section_width = cols * thumb_size[0] + (cols + 1) * margin
section_height = (
    header_height +
    rows * (thumb_size[1] + label_height + margin) +
    margin
)

canvas = Image.new(
    "RGB",
    (
        section_width * 2 + margin * 3,
        section_height + margin * 2
    ),
    "white"
)

draw = ImageDraw.Draw(canvas)


def add_section(x_offset, title, files):

    draw.text(
        (x_offset + section_width // 2, margin + 20),
        title,
        fill="black",
        anchor="mm"
    )

    start_y = margin + header_height

    for i, file in enumerate(files):

        row = i // cols
        col = i % cols

        x = (
            x_offset +
            margin +
            col * (thumb_size[0] + margin)
        )

        y = (
            start_y +
            row * (
                thumb_size[1] +
                label_height +
                margin
            )
        )

        image = Image.open(file).convert("RGB")

        image = ImageOps.contain(
            image,
            thumb_size
        )

        tile = Image.new(
            "RGB",
            thumb_size,
            "white"
        )

        tile.paste(
            image,
            (
                (thumb_size[0] - image.width) // 2,
                (thumb_size[1] - image.height) // 2
            )
        )

        canvas.paste(tile, (x, y))

        draw.rectangle(
            [
                x,
                y,
                x + thumb_size[0],
                y + thumb_size[1]
            ],
            outline="gray"
        )

        name = file.name

        if len(name) > 25:
            name = name[:22] + "..."

        draw.text(
            (x + thumb_size[0] // 2,
             y + thumb_size[1] + 5),
            name,
            fill="black",
            anchor="ma"
        )


add_section(
    margin,
    "ORIGINAL TEST FAKE",
    original_sample
)

add_section(
    section_width + margin * 2,
    "EXTERNAL FAKE",
    external_sample
)

canvas.save(
    output,
    quality=95
)

print("Created:", output)
print("Original sample:", len(original_sample))
print("External sample:", len(external_sample))