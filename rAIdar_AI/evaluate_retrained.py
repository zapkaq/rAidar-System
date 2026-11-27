import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = r"C:\Users\Adrian\OneDrive\ドキュメント\rAIdar_Dataset\real_vs_fake\real-vs-fake"
MODEL_PATH = "deepfake_detector_retrained.keras"

test_path = DATASET_PATH + r"\test"

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(224, 224),
    batch_size=32,
    label_mode="binary",
    shuffle=False
)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = []
y_prob = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0).ravel()
    y_true.extend(labels.numpy().ravel())
    y_prob.extend(predictions)

y_true = np.array(y_true).astype(int)
y_prob = np.array(y_prob)
y_pred = (y_prob >= 0.5).astype(int)

print("Accuracy:", np.mean(y_pred == y_true))
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["FAKE", "REAL"],
    digits=4
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nPrediction Statistics:")
print("Fake mean:", y_prob[y_true == 0].mean())
print("Real mean:", y_prob[y_true == 1].mean())
print("Minimum:", y_prob.min())
print("Maximum:", y_prob.max())