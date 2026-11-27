import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "deepfake_detector_retrained.keras"
EXTERNAL_PATH = r"C:\Users\Adrian\source\repos\rAIdar_AI\external_test"

IMG_SIZE = (224, 224)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# OpenCV built-in face detector
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)


def predict_image(pil_image):
    """
    Uses the EXACT preprocessing used by the retrained MobileNetV2 model:
    RGB -> 224x224 -> float32
    No manual /255 because preprocessing is inside the model.
    """
    image = pil_image.convert("RGB")
    image = image.resize(IMG_SIZE)

    array = np.array(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    prediction = float(model.predict(array, verbose=0)[0][0])

    if prediction >= 0.5:
        label = "REAL"
        confidence = prediction * 100
    else:
        label = "FAKE"
        confidence = (1 - prediction) * 100

    return label, confidence, prediction


def center_crop(image):
    """
    Square center crop.
    """
    width, height = image.size
    size = min(width, height)

    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size

    return image.crop((left, top, right, bottom))


def detect_largest_face(image):
    """
    Detect faces and return the largest detected face crop.
    Returns None if no face is detected.
    """

    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return None

    # Choose largest face
    largest = max(
        faces,
        key=lambda box: box[2] * box[3]
    )

    x, y, w, h = largest

    # Add 20% padding around face
    pad_x = int(w * 0.20)
    pad_y = int(h * 0.20)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(rgb.shape[1], x + w + pad_x)
    y2 = min(rgb.shape[0], y + h + pad_y)

    face = rgb[y1:y2, x1:x2]

    return Image.fromarray(face)


def process_file(filepath, actual):
    filename = os.path.basename(filepath)

    try:
        original = Image.open(filepath).convert("RGB")

        # ---------------------------------------------------------
        # 1. FULL IMAGE
        # ---------------------------------------------------------
        full_label, full_conf, full_raw = predict_image(original)

        # ---------------------------------------------------------
        # 2. CENTER CROP
        # ---------------------------------------------------------
        cropped = center_crop(original)
        center_label, center_conf, center_raw = predict_image(cropped)

        # ---------------------------------------------------------
        # 3. FACE CROP
        # ---------------------------------------------------------
        face = detect_largest_face(original)

        if face is not None:
            face_label, face_conf, face_raw = predict_image(face)
            face_result = f"{face_label} {face_conf:.2f}%"
        else:
            face_label = "NO_FACE"
            face_conf = None
            face_raw = None
            face_result = "NO_FACE"

        print(
            f"{filename[:30]:30} "
            f"{actual:5} | "
            f"FULL: {full_label:4} {full_conf:6.2f}% | "
            f"CENTER: {center_label:4} {center_conf:6.2f}% | "
            f"FACE: {face_result}"
        )

        return {
            "filename": filename,
            "actual": actual,
            "full_label": full_label,
            "full_conf": full_conf,
            "full_raw": full_raw,
            "center_label": center_label,
            "center_conf": center_conf,
            "center_raw": center_raw,
            "face_label": face_label,
            "face_conf": face_conf,
            "face_raw": face_raw,
        }

    except Exception as e:
        print(f"{filename}: ERROR - {e}")
        return None


all_results = []

print("\n" + "=" * 130)
print("CROP / FACE DIAGNOSTIC")
print("=" * 130)

for actual in ["fake", "real"]:

    folder = os.path.join(EXTERNAL_PATH, actual)

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])

    for filename in files:
        filepath = os.path.join(folder, filename)

        result = process_file(filepath, actual.upper())

        if result:
            all_results.append(result)


# ================================================================
# SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)


def accuracy(results, label_key):
    valid = [
        r for r in results
        if r[label_key] in ["FAKE", "REAL"]
    ]

    if not valid:
        return None

    correct = sum(
        r[label_key] == r["actual"]
        for r in valid
    )

    return correct / len(valid) * 100


for group in ["FAKE", "REAL", "ALL"]:

    if group == "ALL":
        subset = all_results
    else:
        subset = [
            r for r in all_results
            if r["actual"] == group
        ]

    print(f"\n{group}")

    full_acc = accuracy(subset, "full_label")
    center_acc = accuracy(subset, "center_label")
    face_acc = accuracy(subset, "face_label")

    print(
        "Full-image accuracy:",
        f"{full_acc:.2f}%" if full_acc is not None else "N/A"
    )

    print(
        "Center-crop accuracy:",
        f"{center_acc:.2f}%" if center_acc is not None else "N/A"
    )

    print(
        "Face-crop accuracy:",
        f"{face_acc:.2f}%" if face_acc is not None else "N/A"
    )


# Face detection rate
face_detected = sum(
    r["face_label"] != "NO_FACE"
    for r in all_results
)

print("\nFace detection:")
print(
    f"{face_detected}/{len(all_results)} images "
    f"({face_detected / len(all_results) * 100:.2f}%) "
    "had a detected face."
)

print("\nDiagnostic complete.")