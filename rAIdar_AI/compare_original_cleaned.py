
import numpy as np
import tensorflow as tf

ORIGINAL = "deepfake_detector_tigas_balanced.keras"
CLEANED = "deepfake_detector_tigas_cleaned.keras"

print("Loading original model...")
original = tf.keras.models.load_model(ORIGINAL)

print("Loading cleaned model...")
cleaned = tf.keras.models.load_model(CLEANED)

original_weights = original.get_weights()
cleaned_weights = cleaned.get_weights()

print("\n========== ORIGINAL VS CLEANED ==========")

print("Original arrays:", len(original_weights))
print("Cleaned arrays:", len(cleaned_weights))

if len(original_weights) != len(cleaned_weights):
    print("Different number of weight arrays.")
else:
    different_arrays = 0
    max_difference = 0.0

    for i, (a, b) in enumerate(
        zip(original_weights, cleaned_weights)
    ):
        if a.shape != b.shape:
            print(f"Array {i}: SHAPE MISMATCH")
            different_arrays += 1
            continue

        difference = float(np.max(np.abs(a - b)))
        max_difference = max(max_difference, difference)

        if not np.array_equal(a, b):
            different_arrays += 1

    print("Different weight arrays:", different_arrays)
    print("Maximum weight difference:", max_difference)

    if different_arrays == 0:
        print("\nRESULT: Original and cleaned weights are identical.")
    else:
        print("\nRESULT: Original and cleaned weights are different.")
