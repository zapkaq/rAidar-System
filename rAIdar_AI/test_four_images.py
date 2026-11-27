import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = "deepfake_detector_retrained.keras"

IMAGE_FILES = [
    "real_image.jpg",
    "real_filtered_image.jpg",
    "ai_image.jpg",
    "ai_filtered_image.jpg"
]

model = tf.keras.models.load_model(MODEL_PATH)

for filename in IMAGE_FILES:

    if not os.path.exists(filename):
        print(f"\nNOT FOUND: {filename}")
        continue

    image = Image.open(filename).convert("RGB")
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

    print("=" * 50)
    print("Image:", filename)
    print("Prediction:", label)
    print("Confidence:", f"{confidence:.2f}%")
    print("Raw prediction:", f"{prediction:.6f}")