
import os
import tensorflow as tf

models = [
    "best_model_tigas_balanced_cleaned.keras",
    "deepfake_detector_tigas_cleaned.keras"
]

for model_path in models:
    print("\nChecking:", model_path)

    if not os.path.exists(model_path):
        print("STATUS: FILE NOT FOUND")
        continue

    try:
        model = tf.keras.models.load_model(model_path)

        print("STATUS: MODEL LOADED SUCCESSFULLY")
        print("Input shape:", model.input_shape)
        print("Output shape:", model.output_shape)
        print("Parameters:", model.count_params())

    except Exception as e:
        print("STATUS: MODEL FAILED TO LOAD")
        print("ERROR:", str(e))
