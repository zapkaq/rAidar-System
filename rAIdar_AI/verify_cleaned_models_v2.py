
import os
import tensorflow as tf

models = [
    "best_model_tigas_balanced_cleaned_v2.keras",
    "deepfake_detector_tigas_cleaned_v2.keras"
]

for path in models:
    print("\nChecking:", path)

    if not os.path.exists(path):
        print("FILE NOT FOUND")
        continue

    try:
        model = tf.keras.models.load_model(path)

        print("STATUS: SUCCESS")
        print("Input:", model.input_shape)
        print("Output:", model.output_shape)
        print("Parameters:", model.count_params())

    except Exception as e:
        print("STATUS: FAILED")
        print("ERROR:", str(e))
