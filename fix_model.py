import tensorflow as tf

# Load your old model
model = tf.keras.models.load_model("brain_tumor_model.h5", compile=False)

# Save in new format
model.save("brain_tumor_model.keras")

print("Model converted successfully!")