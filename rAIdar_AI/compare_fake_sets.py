import os
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "deepfake_detector_retrained.keras"

ORIGINAL_FAKE = r"C:\Users\Adrian\OneDrive\ドキュメント\rAIdar_Dataset\real_vs_fake\real-vs-fake\test\fake"

EXTERNAL_FAKE = r"C:\Users\Adrian\source\repos\rAIdar_AI\external_test\fake"

IMG_SIZE = (224, 224)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)


def predict(path):
    image = Image.open(path).convert("RGB")
    image = image.resize(IMG_SIZE)

    array = np.array(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    return float(model.predict(array, verbose=0)[0][0])


def get_images(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])


def analyze(name, folder, limit=20):

    files = get_images(folder)[:limit]

    predictions = []

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    for path in files:

        p = predict(path)

        predictions.append(p)

        if p >= 0.5:
            label = "REAL"
            confidence = p * 100
        else:
            label = "FAKE"
            confidence = (1 - p) * 100

        print(
            f"{os.path.basename(path):35} "
            f"{label:5} "
            f"{confidence:7.2f}% "
            f"raw={p:.6f}"
        )

    predictions = np.array(predictions)

    print("\nSUMMARY")
    print("Images:", len(predictions))
    print("Mean raw prediction:", predictions.mean())
    print("Median:", np.median(predictions))
    print("Minimum:", predictions.min())
    print("Maximum:", predictions.max())
    print("Predicted FAKE:", np.sum(predictions < 0.5))
    print("Predicted REAL:", np.sum(predictions >= 0.5))

    return predictions


original = analyze(
    "ORIGINAL TEST FAKE",
    ORIGINAL_FAKE
)

external = analyze(
    "EXTERNAL FAKE",
    EXTERNAL_FAKE
)

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(
    f"Original fake mean: {original.mean():.6f}"
)

print(
    f"External fake mean: {external.mean():.6f}"
)

print(
    f"Difference: {external.mean() - original.mean():.6f}"
)