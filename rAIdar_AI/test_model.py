import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = "deepfake_detector.keras"

model = tf.keras.models.load_model(MODEL_PATH)

images = [
    "real_image.jpg",
    "real_filtered_image.jpg",
    "ai_image.jpg",
    "ai_filtered_image.jpg"
]

for filename in images:

    if not os.path.exists(filename):
        print(f"NOT FOUND: {filename}")
        continue

    image = Image.open(filename).convert("RGB")
    image = image.resize((224, 224))

    array = np.array(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    prediction = float(model.predict(array, verbose=0)[0][0])

    print("--------------------------------")
    print("Image:", filename)
    print("Raw prediction:", prediction)

    if prediction >= 0.5:
        print("REAL")
    else:
        print("FAKE")

    print("--------------------------------")