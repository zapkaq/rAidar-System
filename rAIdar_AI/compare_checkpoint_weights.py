
import numpy as np
import tensorflow as tf

CHECKPOINT = "best_model_tigas_balanced_cleaned_v2.keras"
FINAL_MODEL = "deepfake_detector_tigas_cleaned_v2.keras"

print("Loading checkpoint...")
checkpoint = tf.keras.models.load_model(CHECKPOINT)

print("Loading final model...")
final_model = tf.keras.models.load_model(FINAL_MODEL)

checkpoint_weights = checkpoint.get_weights()
final_weights = final_model.get_weights()

print("\n========== CHECKPOINT COMPARISON ==========")

print("Checkpoint arrays:", len(checkpoint_weights))
print("Final model arrays:", len(final_weights))

if len(checkpoint_weights) != len(final_weights):
    print("Different number of weight arrays.")
else:
    different_arrays = 0
    max_difference = 0.0

    for i, (a, b) in enumerate(
        zip(checkpoint_weights, final_weights)
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
        print("RESULT: Checkpoint and final model are identical.")
    else:
        print("RESULT: Checkpoint and final model are different.")
