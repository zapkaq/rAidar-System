import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\Adrian\OneDrive\ドキュメント\rAIdar_Dataset\real_vs_fake\real-vs-fake"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VALID_PATH = os.path.join(DATASET_PATH, "valid")
TEST_PATH = os.path.join(DATASET_PATH, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ============================================================
# DATASET
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Classes:", train_ds.class_names)

# Expected:
# ['fake', 'real']

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("BUILDING MOBILENETV2 MODEL")
print("=" * 70)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "best_model_retrained.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# ============================================================
# PHASE 1 - TRAIN CLASSIFIER
# ============================================================

print("=" * 70)
print("PHASE 1: TRAINING CLASSIFIER")
print("=" * 70)

history1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5,
    callbacks=callbacks
)

# ============================================================
# PHASE 2 - FINE TUNING
# ============================================================

print("=" * 70)
print("PHASE 2: FINE-TUNING LAST 30 MOBILENETV2 LAYERS")
print("=" * 70)

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

history2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5,
    callbacks=callbacks
)

# ============================================================
# TEST
# ============================================================

print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

results = model.evaluate(
    test_ds,
    verbose=1
)

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.6f}")

# ============================================================
# SAVE
# ============================================================

FINAL_MODEL = "deepfake_detector_retrained.keras"

model.save(FINAL_MODEL)

print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(FINAL_MODEL)
print("Parameters:", model.count_params())