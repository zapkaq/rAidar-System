
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = os.path.join(
    "datasets", "diverse_balanced_cleaned"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

BEST_MODEL = "best_model_tigas_balanced_cleaned.keras"
FINAL_MODEL = "deepfake_detector_tigas_cleaned.keras"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VALID_PATH = os.path.join(DATASET_PATH, "valid")

tf.keras.utils.set_random_seed(SEED)

# Prevent accidental overwriting of existing models.
for filename in [BEST_MODEL, FINAL_MODEL]:
    if Path(filename).exists():
        raise FileExistsError(
            f"{filename} already exists. "
            "Rename or back up the existing file before training."
        )

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING CLEANED TIGAS DATASET")
print("=" * 70)

for path in [TRAIN_PATH, VALID_PATH]:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Dataset folder not found: {path}"
        )

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)

valid_ds = keras.utils.image_dataset_from_directory(
    VALID_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

expected_classes = ["fake", "real"]

for name, dataset in [
    ("train", train_ds),
    ("valid", valid_ds)
]:
    print(f"{name} classes: {dataset.class_names}")

    if dataset.class_names != expected_classes:
        raise ValueError(
            f"Unexpected class order in {name}: "
            f"{dataset.class_names}"
        )

print("Label mapping: fake=0, real=1")
print("Model output represents the probability of REAL.")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)

# ============================================================
# BUILD MODEL
# ============================================================

print("=" * 70)
print("BUILDING MOBILENETV2")
print("=" * 70)

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

# Images are loaded in the 0-255 range.
# Preprocessing is performed inside the model.
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)


def compile_model(learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc")
        ]
    )


model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    keras.callbacks.ModelCheckpoint(
        BEST_MODEL,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    )
]

# ============================================================
# PHASE 1: TRAIN CLASSIFIER
# ============================================================

print("=" * 70)
print("PHASE 1: TRAINING CLASSIFIER")
print("=" * 70)

compile_model(1e-4)

model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5,
    callbacks=callbacks
)

# ============================================================
# PHASE 2: FINE-TUNING
# ============================================================

print("=" * 70)
print("PHASE 2: FINE-TUNING MOBILE NET V2")
print("=" * 70)

base_model.trainable = True

# Freeze all but the last 30 layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

compile_model(1e-5)

model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5,
    callbacks=callbacks
)

# ============================================================
# SAVE BEST MODEL
# ============================================================

print("=" * 70)
print("LOADING BEST VALIDATION CHECKPOINT")
print("=" * 70)

if not Path(BEST_MODEL).exists():
    raise FileNotFoundError(
        "Best checkpoint was not created. "
        "Training may not have completed successfully."
    )

best_model = keras.models.load_model(BEST_MODEL)

best_model.save(FINAL_MODEL)

print("=" * 70)
print("CLEANED DATASET TRAINING COMPLETE")
print("=" * 70)

print(f"Best checkpoint: {BEST_MODEL}")
print(f"Final model: {FINAL_MODEL}")
print("Test dataset was not used during training.")
