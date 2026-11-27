import tensorflow as tf
import numpy as np

MODEL_PATH = "deepfake_detector_tigas_balanced.keras"

print("Loading TIGAS model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("\nInput shape:", model.input_shape)
print("Output shape:", model.output_shape)

print("\nRunning a compatibility test...")
test_image = np.zeros((1, 224, 224, 3), dtype=np.float32)
prediction = model.predict(test_image, verbose=0)

print("Prediction shape:", prediction.shape)
print("Prediction value:", prediction.tolist())

assert model.input_shape[-3:] == (224, 224, 3), "Unexpected input shape"
assert prediction.shape == (1, 1), "Unexpected output shape"
assert np.isfinite(prediction).all(), "Prediction is not finite"
assert 0.0 <= float(prediction[0, 0]) <= 1.0, "Output is not a probability"

print("\nCompatibility checks passed.")
print("No files were changed.")
