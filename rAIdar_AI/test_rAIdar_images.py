"""
rAIdar image dataset evaluator
--------------------------------
Place this file in the same folder as:
  - deepfake_detector_tigas_balanced.keras
  - test_dataset/images/real/
  - test_dataset/images/fake/

This script only loads the existing model and reads test images.
It does not modify the model, Flask API, or website.
"""

import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "deepfake_detector_tigas_balanced.keras"
REAL_DIR = BASE_DIR / "test_dataset" / "images" / "real"
FAKE_DIR = BASE_DIR / "test_dataset" / "images" / "fake"
REPORT_PATH = BASE_DIR / "image_test_results.csv"
IMAGE_SIZE = (224, 224)
THRESHOLD = 0.5
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder):
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def prepare_image(path):
    # Match app.py: RGB, 224x224, float32, no extra normalization.
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        return np.asarray(image, dtype=np.float32)


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find model:\n{MODEL_PATH}\n"
            "Put this script in the rAIdar_AI folder containing the .keras model."
        )

    real_images = get_images(REAL_DIR)
    fake_images = get_images(FAKE_DIR)

    if not REAL_DIR.exists() or not FAKE_DIR.exists():
        raise FileNotFoundError(
            "Dataset folders not found. Expected:\n"
            f"{REAL_DIR}\n{FAKE_DIR}"
        )

    if not real_images or not fake_images:
        raise ValueError(
            "Both folders need at least one image.\n"
            f"Real images found: {len(real_images)}\n"
            f"Fake images found: {len(fake_images)}"
        )

    print("Loading existing model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded. Real images: {len(real_images)} | Fake images: {len(fake_images)}")
    print("Running image tests. This may take a while...\n")

    rows = []
    skipped = []

    # Ground-truth class comes from the folder containing each image.
    for true_label, paths in (("REAL", real_images), ("FAKE", fake_images)):
        for index, path in enumerate(paths, start=1):
            try:
                batch = np.expand_dims(prepare_image(path), axis=0)
                raw_score = float(np.asarray(model.predict(batch, verbose=0)).reshape(-1)[0])
                predicted_label = "REAL" if raw_score >= THRESHOLD else "FAKE"
                confidence = (raw_score if predicted_label == "REAL" else 1.0 - raw_score) * 100.0

                rows.append({
                    "file_name": path.name,
                    "file_path": str(path.relative_to(BASE_DIR)),
                    "actual_label": true_label,
                    "predicted_label": predicted_label,
                    "raw_prediction": round(raw_score, 6),
                    "reported_confidence_percent": round(confidence, 2),
                    "correct": predicted_label == true_label,
                })
                print(
                    f"[{len(rows):>3}] Actual: {true_label:<4} | "
                    f"Predicted: {predicted_label:<4} | "
                    f"Score: {confidence:6.2f}% | {path.name}"
                )
            except Exception as exc:
                skipped.append((str(path), str(exc)))
                print(f"[SKIPPED] {path.name}: {exc}")

    if not rows:
        raise RuntimeError("No images could be evaluated. Check the files and model input format.")

    # Confusion matrix: actual REAL/FAKE vs predicted REAL/FAKE.
    actual = [r["actual_label"] for r in rows]
    predicted = [r["predicted_label"] for r in rows]
    tp_real = sum(a == "REAL" and p == "REAL" for a, p in zip(actual, predicted))
    fn_real = sum(a == "REAL" and p == "FAKE" for a, p in zip(actual, predicted))
    fp_real = sum(a == "FAKE" and p == "REAL" for a, p in zip(actual, predicted))
    tn_fake = sum(a == "FAKE" and p == "FAKE" for a, p in zip(actual, predicted))

    correct_count = sum(r["correct"] for r in rows)
    accuracy = safe_divide(correct_count, len(rows))
    precision_fake = safe_divide(tn_fake, tn_fake + fn_real)  # predicted FAKE that were actually FAKE
    recall_fake = safe_divide(tn_fake, tn_fake + fp_real)     # actual FAKE correctly found
    f1_fake = safe_divide(2 * precision_fake * recall_fake, precision_fake + recall_fake)

    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = [
            "file_name", "file_path", "actual_label", "predicted_label",
            "raw_prediction", "reported_confidence_percent", "correct"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("IMAGE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Images evaluated: {len(rows)}")
    print(f"Images skipped:   {len(skipped)}")
    print(f"Correct:          {correct_count}")
    print(f"Accuracy:         {accuracy * 100:.2f}%")
    print(f"FAKE precision:   {precision_fake * 100:.2f}%")
    print(f"FAKE recall:      {recall_fake * 100:.2f}%")
    print(f"FAKE F1-score:    {f1_fake * 100:.2f}%")
    print("\nConfusion matrix (rows = actual, columns = predicted)")
    print("                 Predicted REAL   Predicted FAKE")
    print(f"Actual REAL      {tp_real:>14}   {fn_real:>14}")
    print(f"Actual FAKE      {fp_real:>14}   {tn_fake:>14}")
    print(f"\nDetailed results saved to:\n{REPORT_PATH}")
    print("\nNote: the API's confidence field is a model score, not a guarantee of authenticity.")

    if skipped:
        print("\nSkipped files:")
        for file_path, error in skipped:
            print(f"- {file_path}: {error}")


if __name__ == "__main__":
    main()
