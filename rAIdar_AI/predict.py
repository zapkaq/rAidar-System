import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("deepfake_detector.keras")

# Image path from Laravel
img_path = sys.argv[1]

# Load image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array, verbose=0)

confidence = float(prediction[0][0])

if confidence > 0.5:
    label = "REAL"
    confidence = confidence * 100
else:
    label = "FAKE"
    confidence = (1 - confidence) * 100

print(f"{label},{confidence:.2f}")