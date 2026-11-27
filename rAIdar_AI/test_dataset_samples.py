import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = "deepfake_detector_retrained.keras"

DATASET_PATH = r"C:\Users\Adrian\OneDrive\ドキュメント\rAIdar_Dataset\real_vs_fake\real-vs-fake"
FAKE_PATH = os.path.join(DATASET_PATH, "test", "fake")
REAL_PATH = os.path.join(DATASET_PATH, "test", "real")

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(filepath):
    image = Image.open(filepath).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = float(model.predict(image_array, verbose=0)[0][0])

    if prediction >= 0.5:
        label = "REAL"
        confidence = prediction * 100
    else:
        label = "FAKE"
        confidence = (1 - prediction) * 100

    return label, confidence, prediction


print("=" * 70)
print("FAKE DATASET SAMPLES")
print("=" * 70)

fake_files = sorted(os.listdir(FAKE_PATH))[:10]

fake_correct = 0

for filename in fake_files:
    filepath = os.path.join(FAKE_PATH, filename)

    label, confidence, raw = predict_image(filepath)

    correct = label == "FAKE"

    if correct:
        fake_correct += 1

    print(
        f"{filename:25} "
        f"Prediction: {label:4} "
        f"Confidence: {confidence:7.2f}% "
        f"Raw: {raw:.6f} "
        f"{'CORRECT' if correct else 'WRONG'}"
    )


print("\n" + "=" * 70)
print("REAL DATASET SAMPLES")
print("=" * 70)

real_files = sorted(os.listdir(REAL_PATH))[:10]

real_correct = 0

for filename in real_files:
    filepath = os.path.join(REAL_PATH, filename)

    label, confidence, raw = predict_image(filepath)

    correct = label == "REAL"

    if correct:
        real_correct += 1

    print(
        f"{filename:25} "
        f"Prediction: {label:4} "
        f"Confidence: {confidence:7.2f}% "
        f"Raw: {raw:.6f} "
        f"{'CORRECT' if correct else 'WRONG'}"
    )


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Fake samples correct: {fake_correct}/10")
print(f"Real samples correct: {real_correct}/10")
print(f"Total correct: {fake_correct + real_correct}/20")
print(f"Sample accuracy: {(fake_correct + real_correct) / 20 * 100:.2f}%")