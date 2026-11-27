
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# CLEANED DATASET
DATASET_PATH = os.path.join(
    "datasets", "diverse_balanced_cleaned"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# NEW MODEL FILENAMES
BEST_MODEL = "best_model_tigas_balanced_cleaned_v2.keras"
FINAL_MODEL = "deepfake_detector_tigas_cleaned_v2.keras"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VALID_PATH = os.path.join(DATASET_PATH, "valid")

tf.keras.utils.set_random_seed(SEED)

# Prevent accidental overwriting
for path in [BEST_MODEL, FINAL_MODEL]:
    if os.path.exists(path):
        raise FileExistsError(
            f"{path} already exists. Use another filename."
        )

# Verify dataset folders
for path in [TRAIN_PATH, VALID_PATH]:
    if not os.path.isdir(path):
        raise FileNotFoundError(path)

# Load training data
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
            f"Unexpected class order in {name}"
        )

print("Training uses cleaned dataset.")
print("Label mapping: fake=0, real=1")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)

# MobileNetV2
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

x = keras.applications.mobilenet_v2.preprocess_input(
    inputs
)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    1, activation="sigmoid"
)(x)

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

def create_callbacks():
    return [
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

# PHASE 1: Train classifier
print("\n========== PHASE 1 ==========")

compile_model(1e-4)

model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5,
    callbacks=create_callbacks()
)

# PHASE 2: Fine-tuning
print("\n========== PHASE 2 ==========")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

compile_model(1e-5)

model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5,
    callbacks=create_callbacks()
)

# Load best validation checkpoint
print("\nLoading best checkpoint...")

best_model = keras.models.load_model(BEST_MODEL)

# Save final model
best_model.save(FINAL_MODEL)

print("\n========== TRAINING COMPLETE ==========")
print("Best checkpoint:", BEST_MODEL)
print("Final model:", FINAL_MODEL)
print("Test dataset was NOT used during training.")
