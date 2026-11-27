
import numpy as np
import tensorflow as tf
from tensorflow import keras

DATASET_PATH = "datasets/diverse_balanced_cleaned/valid"

OLD_MODEL = "deepfake_detector_tigas_cleaned.keras"
NEW_MODEL = "deepfake_detector_tigas_cleaned_v2.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("Loading validation dataset...")

valid_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Loading original model...")
old_model = keras.models.load_model(OLD_MODEL)

print("Loading v2 model...")
new_model = keras.models.load_model(NEW_MODEL)

print("\nGenerating predictions...")

old_predictions = old_model.predict(valid_ds, verbose=1).flatten()
new_predictions = new_model.predict(valid_ds, verbose=1).flatten()

differences = np.abs(old_predictions - new_predictions)

print("\n========== VALIDATION COMPARISON ==========")
print("Validation images:", len(old_predictions))
print("Different predictions:", np.count_nonzero(differences))
print("Maximum probability difference:", np.max(differences))
print("Mean probability difference:", np.mean(differences))

if np.array_equal(old_predictions, new_predictions):
    print("\nRESULT: Validation predictions are exactly identical.")
else:
    print("\nRESULT: Validation predictions are different.")
