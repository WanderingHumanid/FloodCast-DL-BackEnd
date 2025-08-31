import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def test_model_input():
    try:
        # Load the model
        model = tf.keras.models.load_model("../flood_prediction_model.h5")
        print("Model loaded successfully")
        print(f"Model input shape: {model.input_shape}")
        
        # Create some test data with shape (1, 24, 11)
        key_features = [
            'water_level', 'rain', 'temperature', 
            'wind_speed', 'sea_level_pressure', 'hour_sin', 
            'hour_cos', 'dayofyear', 'tide_level', 
            'tide_velocity', 'moon_illumination_fraction'
        ]
        
        # Create a simple time series dataset
        dates = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
        df = pd.DataFrame({
            'water_level': np.random.random(24) * 10,
            'rain': np.random.random(24) * 5,
            'temperature': np.random.random(24) * 30 + 10,
            'wind_speed': np.random.random(24) * 20,
            'sea_level_pressure': np.random.random(24) * 10 + 1000,
            'hour_sin': np.sin(2 * np.pi * np.array([d.hour for d in dates]) / 24),
            'hour_cos': np.cos(2 * np.pi * np.array([d.hour for d in dates]) / 24),
            'dayofyear': [d.timetuple().tm_yday for d in dates],
            'tide_level': np.random.random(24) * 100 + 100,
            'tide_velocity': np.random.random(24) * 5 - 2.5,
            'moon_illumination_fraction': np.random.random(24)
        }, index=dates)
        
        # Extract the values and reshape
        input_sequence = df[key_features].values.astype(np.float32)
        input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension
        
        print(f"Input sequence shape: {input_sequence.shape}")
        
        # Make a prediction
        prediction = model.predict(input_sequence, verbose=0)
        print(f"Prediction result: {prediction[0][0]}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_model_input()
