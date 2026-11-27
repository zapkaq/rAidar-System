from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

MODEL_PATH = "deepfake_detector_tigas_balanced.keras"
TEST_DIR = Path("datasets/diverse_balanced/test")
IMG_SIZE = (224, 224)
THRESHOLD = 0.5

print("Loading model...")
model = keras.models.load_model(MODEL_PATH)

class_names = sorted(
    [p.name for p in TEST_DIR.iterdir() if p.is_dir()]
)

if class_names != ["fake", "real"]:
    raise ValueError(f"Unexpected class order: {class_names}")

image_paths = sorted(
    p for p in TEST_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp"
    }
)

print(f"Found {len(image_paths)} test images.")
print("Class order:", class_names)

if len(image_paths) == 0:
    raise RuntimeError("No test images found.")

labels = []
scores = {"TensorFlow": [], "PIL": [], "OpenCV": []}


def predict_score(array):
    array = np.asarray(array, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    return float(model.predict(array, verbose=0).reshape(-1)[0])


for index, path in enumerate(image_paths, start=1):
    actual_class = path.relative_to(TEST_DIR).parts[0]

    if actual_class not in class_names:
        raise ValueError(f"Unexpected class folder: {path}")

    # Class order: fake = 0, real = 1
    labels.append(class_names.index(actual_class))

    try:
        with Image.open(path) as image:
            original = np.asarray(image.convert("RGB"))

        # TensorFlow bilinear resizing, matching the training loader
        tf_image = tf.image.resize(
            original, IMG_SIZE, method="bilinear"
        ).numpy()

        # PIL resizing, matching the Flask image endpoint
        with Image.open(path) as image:
            pil_image = image.convert("RGB").resize(
                (IMG_SIZE[1], IMG_SIZE[0])
            )
            pil_image = np.asarray(pil_image)

        # OpenCV resizing, matching the video endpoint's resize style
        cv_image = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        cv_image = cv2.resize(
            cv_image, (IMG_SIZE[1], IMG_SIZE[0])
        )
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        scores["TensorFlow"].append(predict_score(tf_image))
        scores["PIL"].append(predict_score(pil_image))
        scores["OpenCV"].append(predict_score(cv_image))

    except Exception as exc:
        raise RuntimeError(f"Failed to process {path}: {exc}") from exc

    if index % 50 == 0 or index == len(image_paths):
        print(f"Processed {index}/{len(image_paths)} images")

y_true = np.asarray(labels)

print("\n" + "=" * 65)
print("PREPROCESSING COMPARISON")
print("Threshold: score >= 0.5 predicts REAL; score < 0.5 predicts FAKE")
print("=" * 65)

for method, method_scores in scores.items():
    y_score = np.asarray(method_scores)
    y_pred = (y_score >= THRESHOLD).astype(int)

    print(f"\n--- {method} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["FAKE", "REAL"],
            digits=4,
            zero_division=0,
        )
    )
    print(f"ROC AUC: {roc_auc_score(y_true, y_score):.4f}")

print("\nFinished. No model was trained or modified.")