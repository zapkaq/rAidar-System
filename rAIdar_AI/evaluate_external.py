import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import os

MODEL_PATH = "deepfake_detector_retrained.keras"

EXTERNAL_PATH = r"C:\Users\Adrian\source\repos\rAIdar_AI\external_test"

FAKE_PATH = os.path.join(EXTERNAL_PATH, "fake")
REAL_PATH = os.path.join(EXTERNAL_PATH, "real")

model = tf.keras.models.load_model(MODEL_PATH)

y_true = []
y_prob = []

print("=" * 70)
print("EXTERNAL TEST SET")
print("=" * 70)

def process_folder(folder_path, expected_label, folder_name):

    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])

    correct = 0

    print("\n" + "=" * 70)
    print(folder_name)
    print("=" * 70)

    for filename in files:

        filepath = os.path.join(folder_path, filename)

        image = Image.open(filepath).convert("RGB")
        image = image.resize((224, 224))

        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        prediction = float(
            model.predict(image_array, verbose=0)[0][0]
        )

        predicted_label = 1 if prediction >= 0.5 else 0

        if predicted_label == expected_label:
            result = "CORRECT"
            correct += 1
        else:
            result = "WRONG"

        confidence = (
            prediction * 100
            if predicted_label == 1
            else (1 - prediction) * 100
        )

        print(
            f"{filename:30} "
            f"{'REAL' if predicted_label == 1 else 'FAKE':4} "
            f"{confidence:7.2f}% "
            f"Raw: {prediction:.6f} "
            f"{result}"
        )

        y_true.append(expected_label)
        y_prob.append(prediction)

    print(
        f"\n{folder_name} accuracy: "
        f"{correct}/{len(files)} "
        f"({correct / len(files) * 100:.2f}%)"
    )


process_folder(
    FAKE_PATH,
    expected_label=0,
    folder_name="FAKE IMAGES"
)

process_folder(
    REAL_PATH,
    expected_label=1,
    folder_name="REAL IMAGES"
)

y_true_np = np.array(y_true)
y_prob_np = np.array(y_prob)
y_pred_np = (y_prob_np >= 0.5).astype(int)

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

accuracy = np.mean(y_pred_np == y_true_np)

print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(
    classification_report(
        y_true_np,
        y_pred_np,
        target_names=["FAKE", "REAL"],
        digits=4
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_true_np, y_pred_np))

try:
    auc = roc_auc_score(y_true_np, y_prob_np)
    print(f"\nAUC: {auc:.4f}")
except Exception as e:
    print("AUC could not be calculated:", e)

print("\nPrediction Statistics:")
print(
    "Fake mean prediction:",
    y_prob_np[y_true_np == 0].mean()
)

print(
    "Real mean prediction:",
    y_prob_np[y_true_np == 1].mean()
)