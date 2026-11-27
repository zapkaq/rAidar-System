
import numpy as np
import tensorflow as tf

OLD_MODEL = "deepfake_detector_tigas_cleaned.keras"
NEW_MODEL = "deepfake_detector_tigas_cleaned_v2.keras"

print("Loading original model...")
old_model = tf.keras.models.load_model(OLD_MODEL)

print("Loading v2 model...")
new_model = tf.keras.models.load_model(NEW_MODEL)

old_weights = old_model.get_weights()
new_weights = new_model.get_weights()

print("\n========== MODEL WEIGHT COMPARISON ==========")

print("Original weight arrays:", len(old_weights))
print("V2 weight arrays:", len(new_weights))

if len(old_weights) != len(new_weights):
    print("Different number of weight arrays.")
else:
    different_arrays = 0
    max_difference = 0.0

    for i, (old_w, new_w) in enumerate(zip(old_weights, new_weights)):
        if old_w.shape != new_w.shape:
            print(f"Array {i}: SHAPE MISMATCH")
            different_arrays += 1
            continue

        difference = np.max(np.abs(old_w - new_w))
        max_difference = max(max_difference, float(difference))

        if not np.array_equal(old_w, new_w):
            different_arrays += 1

    print("Different weight arrays:", different_arrays)
    print("Maximum weight difference:", max_difference)

    if different_arrays == 0:
        print("\nRESULT: Model weights are identical.")
    else:
        print("\nRESULT: Model weights are different.")
