import os
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

MODEL_PATH = "deepfake_detector_tigas_balanced.keras"
TEST_FOLDER = os.path.join("datasets", "diverse_balanced", "test")
IMAGE_SIZE = (224, 224)
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

print("Loading TIGAS model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

if not os.path.isdir(TEST_FOLDER):
    raise FileNotFoundError(f"Test folder not found: {TEST_FOLDER}")

y_true = []
y_scores = []
errors = 0

for class_name, label in [("fake", 0), ("real", 1)]:
    class_folder = os.path.join(TEST_FOLDER, class_name)

    if not os.path.isdir(class_folder):
        raise FileNotFoundError(f"Class folder not found: {class_folder}")

    files = sorted(
        name for name in os.listdir(class_folder)
        if name.lower().endswith(EXTENSIONS)
    )

    print(f"{class_name}: {len(files)} images")

    for filename in files:
        filepath = os.path.join(class_folder, filename)

        try:
            with Image.open(filepath) as image:
                image = image.convert("RGB").resize(IMAGE_SIZE)
                array = np.asarray(image, dtype=np.float32)

            # The saved model already includes MobileNetV2 preprocessing.
            array = np.expand_dims(array, axis=0)
            score_real = float(model.predict(array, verbose=0)[0][0])

            y_true.append(label)
            y_scores.append(score_real)

        except Exception as error:
            errors += 1
            print(f"Could not process {filepath}: {error}")

if not y_true:
    raise RuntimeError("No images were successfully evaluated.")

y_pred = [1 if score >= 0.5 else 0 for score in y_scores]

print("\n" + "=" * 55)
print("TIGAS MODEL TEST RESULTS")
print("=" * 55)
print("Images evaluated:", len(y_true))
print("Images skipped due to errors:", errors)
print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
print("\nConfusion matrix (rows=actual fake, real; columns=predicted fake, real):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["FAKE", "REAL"],
    zero_division=0
))

if len(set(y_true)) == 2:
    print(f"ROC-AUC: {roc_auc_score(y_true, y_scores):.4f}")

print("Threshold: 0.50")
print("=" * 55)
