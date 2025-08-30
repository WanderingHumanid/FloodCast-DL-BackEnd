"""
model_improved.py

The improved FloodCast model that achieves a positive R² score on 2025 data.
This is a production-ready version that can be used in the application.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

class ImprovedFloodModel:
    """
    An improved flood prediction model that handles time-series data better
    and includes better feature normalization and drift handling.
    """
    
    def __init__(self):
        """Initialize the model, loading the trained model and scaler if available."""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_loaded = False
        
        # Try to load the pre-trained model
        try:
            model_path = os.path.join(os.getcwd(), "models", "floodsense_xgb_model_improved.pkl")
            scaler_path = os.path.join(os.getcwd(), "models", "feature_scaler.pkl")
            features_path = os.path.join(os.getcwd(), "models", "model_features.txt")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                with open(features_path, "r") as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
                
                self.is_loaded = True
                print("✅ Improved flood model loaded successfully")
            else:
                print("⚠️ Pre-trained model files not found - model needs to be trained first")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def prepare_features(self, df):
        """
        Prepare features for prediction using the same transformations as during training.
        
        Args:
            df (pandas.DataFrame): Input data with datetime index and required weather/tide columns
            
        Returns:
            pandas.DataFrame: Processed features ready for prediction
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                raise ValueError("DataFrame index must be convertible to DatetimeIndex")
        
        # Create a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()
        
        # Check required columns
        required_columns = ['water_level', 'rain']
        missing_columns = [col for col in required_columns if col not in df_copy.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Time-based features
        df_copy['hour'] = df_copy.index.hour
        df_copy['dayofweek'] = df_copy.index.dayofweek
        df_copy['month'] = df_copy.index.month
        df_copy['dayofyear'] = df_copy.index.dayofyear
        df_copy['year'] = df_copy.index.year
        df_copy['quarter'] = df_copy.index.quarter
        df_copy['is_weekend'] = df_copy.index.dayofweek >= 5
        
        # Cyclical encoding of time features
        for col in ['hour', 'dayofweek', 'month', 'dayofyear']:
            max_val = {'hour': 23, 'dayofweek': 6, 'month': 12, 'dayofyear': 366}[col]
            df_copy[f'{col}_sin'] = np.sin(2 * np.pi * df_copy[col] / max_val)
            df_copy[f'{col}_cos'] = np.cos(2 * np.pi * df_copy[col] / max_val)
        
        # Define the core numerical features for creating lags
        base_features = [
            'water_level', 'wind_speed', 'temperature', 'dew_point',
            'sea_level_pressure', 'rain', 'moon_illumination_fraction'
        ]
        
        # Add tide_level if it exists in the dataset
        if 'tide_level' in df_copy.columns:
            base_features.append('tide_level')
        
        # Create time-lagged features
        lags = [1, 2, 3, 6, 12, 24]
        for lag in lags:
            for feature in base_features:
                if feature in df_copy.columns:
                    df_copy[f"{feature}_lag{lag}"] = df_copy[feature].shift(lag)
        
        # Add rolling window statistics
        windows = [3, 6, 12, 24]
        for window in windows:
            for feature in base_features:
                if feature in df_copy.columns:
                    df_copy[f"{feature}_rolling_mean_{window}h"] = df_copy[feature].rolling(window=window).mean()
                    df_copy[f"{feature}_rolling_std_{window}h"] = df_copy[feature].rolling(window=window).std()
                    df_copy[f"{feature}_rolling_min_{window}h"] = df_copy[feature].rolling(window=window).min()
                    df_copy[f"{feature}_rolling_max_{window}h"] = df_copy[feature].rolling(window=window).max()
        
        # Create interaction features
        if 'rain' in df_copy.columns and 'tide_level' in df_copy.columns:
            df_copy['rain_tide_interaction'] = df_copy['rain'] * df_copy['tide_level']
        
        if 'wind_speed' in df_copy.columns and 'rain' in df_copy.columns:
            df_copy['wind_rain_interaction'] = df_copy['wind_speed'] * df_copy['rain']
        
        # One-hot encode moon phase if present
        if 'moon_phase' in df_copy.columns:
            df_copy = pd.get_dummies(df_copy, columns=['moon_phase'], prefix='phase')
        
        # Handle missing features required by the model
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(df_copy.columns)
            for feature in missing_features:
                df_copy[feature] = 0  # Fill with zeros
            
            # Keep only the features needed by the model
            df_copy = df_copy[self.feature_names]
        
        return df_copy
    
    def predict_next_hour(self, current_data):
        """
        Predict the water level for the next hour based on current data.
        
        Args:
            current_data (pandas.DataFrame): Current weather and water level data
                                            with datetime index
        
        Returns:
            float: Predicted water level for the next hour
            dict: Additional prediction metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Please train or load the model first.")
        
        try:
            # Prepare features
            features_df = self.prepare_features(current_data)
            
            # Apply the same scaling as during training
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Calculate prediction time
            prediction_time = current_data.index[-1] + pd.Timedelta(hours=1)
            
            # Return the prediction with metadata
            return prediction, {
                "prediction_time": prediction_time,
                "current_water_level": current_data["water_level"].iloc[-1],
                "predicted_change": prediction - current_data["water_level"].iloc[-1],
                "model_version": "improved_v1.0"
            }
        
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None, {"error": str(e)}
    
    def predict_hours_ahead(self, current_data, hours=24):
        """
        Predict water levels for multiple hours ahead.
        
        Args:
            current_data (pandas.DataFrame): Current weather and water level data
            hours (int): Number of hours to predict ahead
        
        Returns:
            pandas.DataFrame: DataFrame with timestamps and predicted water levels
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Please train or load the model first.")
        
        # Create a copy of the data to avoid modifying the original
        working_data = current_data.copy()
        
        # Create a DataFrame to store predictions
        start_time = working_data.index[-1] + pd.Timedelta(hours=1)
        prediction_times = pd.date_range(start=start_time, periods=hours, freq='H')
        predictions_df = pd.DataFrame(index=prediction_times, columns=['predicted_water_level'])
        
        # Make sequential predictions
        for i in range(hours):
            # Prepare features for the current state
            features_df = self.prepare_features(working_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction for the next hour
            next_water_level = self.model.predict(features_scaled)[-1]
            
            # Store the prediction
            predictions_df.iloc[i, 0] = next_water_level
            
            # Create a new row for the next hour with the predicted water level
            next_hour = working_data.index[-1] + pd.Timedelta(hours=1)
            new_row_data = working_data.iloc[-1].copy()
            new_row_data['water_level'] = next_water_level
            
            # Add the new row to the working data
            new_row = pd.DataFrame([new_row_data], index=[next_hour])
            working_data = pd.concat([working_data, new_row])
        
        return predictions_df

# Example usage when running this file directly
if __name__ == "__main__":
    print("Initializing improved flood model...")
    model = ImprovedFloodModel()
    
    if model.is_loaded:
        print("Model loaded successfully. Ready for predictions!")
    else:
        print("Model not loaded. Please train the model first using improved_model.py")
