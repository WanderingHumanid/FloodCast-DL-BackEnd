"""
test_improved_model.py

This script tests the improved FloodCast model with 2025 data to verify
that the R² value is now greater than 0.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
import warnings
warnings.filterwarnings("ignore")

# First, run the improved model training
print("=== Step 1: Training the improved model ===")
from improved_model import train_improved_model
model, scaler, feature_names = train_improved_model()

# Now, test the model on the 2025 data
print("\n=== Step 2: Testing the improved model on 2025 data ===")

def load_test_data():
    """Load the 2025 test data"""
    try:
        # First try to load from the saved holdout file
        holdout_path = os.path.join(os.getcwd(), "data", "2025_holdout_test_data.csv")
        if os.path.exists(holdout_path):
            test_df = pd.read_csv(holdout_path, parse_dates=[0], index_col=0)
            print(f"✅ Loaded 2025 holdout test data: {len(test_df)} samples")
            return test_df
        
        # If the holdout file doesn't exist, load from the original data
        data_path = os.path.join(os.getcwd(), "data", "merged_flood_moon_tide_data.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join(os.getcwd(), "merged_flood_moon_tide_data.csv")
            
        full_df = pd.read_csv(data_path, parse_dates=[0], index_col=0)
        test_df = full_df[full_df.index.year >= 2025].copy()
        
        # Use the last 85% of 2025 data for testing (first 15% was used in training)
        test_size = int(len(test_df) * 0.85)
        test_df = test_df.iloc[-test_size:]
        
        print(f"✅ Extracted 2025 test data: {len(test_df)} samples")
        return test_df
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None

def prepare_test_features(df):
    """Prepare the same features as used in training"""
    df_copy = df.copy()
    
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
    
    # Define the target variable (next hour water level)
    df_copy['target'] = df_copy['water_level'].shift(-1)
    
    # Remove rows with NaN values
    df_copy = df_copy.dropna()
    
    return df_copy

def test_model_on_2025_data():
    """Test the improved model on 2025 data"""
    # Load test data
    test_df = load_test_data()
    if test_df is None or len(test_df) == 0:
        print("❌ No test data available.")
        return
    
    # Prepare features
    prepared_df = prepare_test_features(test_df)
    
    # Extract features and target
    X = prepared_df.drop(columns=['target'])
    y_actual = prepared_df['target'].values
    
    # Ensure we have all required features from the model
    missing_features = set(feature_names) - set(X.columns)
    extra_features = set(X.columns) - set(feature_names)
    
    if missing_features:
        print(f"Adding {len(missing_features)} missing features required by the model")
        for feature in missing_features:
            X[feature] = 0  # Fill with zeros
    
    if extra_features:
        print(f"Removing {len(extra_features)} extra features not used by the model")
        X = X.drop(columns=list(extra_features))
    
    # Reorder columns to match model expectations
    X = X[feature_names]
    
    # Scale features using the same scaler
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-6))) * 100  # Add small epsilon to avoid division by zero
    
    print(f"\nImproved Model Results on 2025 Data:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Plot actual vs predicted
    plt.figure(figsize=(14, 7))
    
    # Only plot a subset of points if there are too many
    max_points = 500
    if len(y_actual) > max_points:
        indices = np.linspace(0, len(y_actual)-1, max_points, dtype=int)
        plot_dates = prepared_df.index[indices]
        plot_actual = y_actual[indices]
        plot_pred = y_pred[indices]
    else:
        plot_dates = prepared_df.index
        plot_actual = y_actual
        plot_pred = y_pred
    
    plt.plot(plot_dates, plot_actual, label='Actual Water Level', color='blue', linewidth=2)
    plt.plot(plot_dates, plot_pred, label='Predicted Water Level', color='red', linestyle='--', linewidth=2)
    plt.title('Improved Model Performance on 2025 Data', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Water Level', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('improved_model_2025_performance.png')
    print("Plot saved to 'improved_model_2025_performance.png'")
    
    # Save metrics to JSON
    metrics = {
        "improved_model_2025_data": {
            "samples": len(y_actual),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "mape": round(mape, 2)
        }
    }
    
    with open('improved_model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Metrics saved to 'improved_model_metrics.json'")
    
    return metrics

if __name__ == "__main__":
    test_model_on_2025_data()
