import tensorflow as tf
import numpy as np
import os

# Load the model - adjust path to parent directory
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'flood_prediction_model.h5')
print(f"Looking for model at: {model_path}")
model = tf.keras.models.load_model(model_path)

# Print model summary
model.summary()

# Create a test input with shape (1, 24, 11)
test_input = np.random.random((1, 24, 11)).astype(np.float32)
print(f"Input shape: {test_input.shape}")

# Make a prediction
prediction = model.predict(test_input)
print(f"Output shape: {prediction.shape}")
print(f"Prediction example: {prediction[0][:5]}")  # Show first 5 values of prediction
