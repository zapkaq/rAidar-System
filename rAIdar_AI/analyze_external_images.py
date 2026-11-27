from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
import csv

TEST_DIR = Path("external_test")
MODEL_PATH = "deepfake_detector_tigas_balanced.keras"
OUTPUT_FILE = "external_image_diagnostic.csv"
IMAGE_SIZE = (224, 224)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

results = []

for actual_label in ["fake", "real"]:
    folder = TEST_DIR / actual_label

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue

        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize(IMAGE_SIZE)
                image_array = np.asarray(img, dtype=np.float32)

            # Keep pixel values in the 0-255 range.
            score = float(
                model.predict(
                    np.expand_dims(image_array, axis=0),
                    verbose=0
                ).reshape(-1)[0]
            )

            predicted_label = "real" if score >= 0.5 else "fake"

            results.append({
                "filename": path.name,
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "real_score": round(score, 4),
                "width": Image.open(path).size[0],
                "height": Image.open(path).size[1],
                "file_size_kb": round(path.stat().st_size / 1024, 2),
            })

        except Exception as exc:
            print(f"Could not process {path.name}: {exc}")

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "filename",
        "actual_label",
        "predicted_label",
        "real_score",
        "width",
        "height",
        "file_size_kb",
    ])
    writer.writeheader()
    writer.writerows(results)

print(f"\nReport saved to: {OUTPUT_FILE}")
print(f"Images analyzed: {len(results)}")

false_positives = [
    row for row in results
    if row["actual_label"] == "real"
    and row["predicted_label"] == "fake"
]

print(f"Real images predicted as fake: {len(false_positives)}")

for row in false_positives:
    print(
        f'{row["filename"]}: '
        f'REAL score={row["real_score"]}, '
        f'size={row["width"]}x{row["height"]}, '
        f'{row["file_size_kb"]} KB'
    )

print("\nNo model was trained or modified.")