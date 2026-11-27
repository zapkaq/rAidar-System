import tensorflow as tf

MODEL_PATH = "deepfake_detector.keras"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("\n========== BASIC MODEL INFORMATION ==========")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

print("\n========== MODEL INPUT ==========")
print(model.inputs)

print("\n========== MODEL OUTPUT ==========")
print(model.outputs)

print("\n========== FINAL LAYER ==========")

final_layer = model.layers[-1]

print("Layer name:", final_layer.name)
print("Layer type:", final_layer.__class__.__name__)
print("Layer configuration:")
print(final_layer.get_config())

print("\n========== ALL MODEL LAYERS ==========")

for index, layer in enumerate(model.layers):
    print(
        index,
        "|",
        layer.name,
        "|",
        layer.__class__.__name__
    )

print("\n========== MODEL CONFIGURATION ==========")

config = model.get_config()

for layer_config in config.get("layers", []):
    layer_name = layer_config.get("config", {}).get("name")
    layer_class = layer_config.get("class_name")

    print("\nLayer:", layer_name)
    print("Type:", layer_class)

    if layer_class in [
        "InputLayer",
        "Rescaling",
        "Normalization",
        "TrueDivide",
        "Subtract",
        "Dense"
    ]:
        print("Configuration:", layer_config.get("config"))