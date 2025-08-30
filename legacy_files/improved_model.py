"""
improved_model.py
An improved version of the FloodCast model that handles time series data better
and includes better feature normalization and drift handling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

def train_improved_model():
    """
    Trains an improved XGBoost model with better handling of time-series data,
    feature normalization, and drift adaptation techniques.
    """
    print("Starting the improved XGBoost model training pipeline...")
    
    # -----------------------
    # 1. Load and prepare the dataset
    # -----------------------
    try:
        data_path = os.path.join(os.getcwd(), "data", "merged_flood_moon_tide_data.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join(os.getcwd(), "merged_flood_moon_tide_data.csv")
            
        df = pd.read_csv(data_path, parse_dates=[0], index_col=0)
        print(f"✅ Successfully loaded data. Date range: {df.index.min()} to {df.index.max()}")
        
        # Important change: Include some 2025 data for training
        # Use first 15% of 2025 data for training to help with drift
        df_pre_2025 = df[df.index.year < 2025]
        df_2025 = df[df.index.year >= 2025]
        
        training_size_2025 = int(len(df_2025) * 0.15)
        df_2025_train = df_2025.iloc[:training_size_2025]
        
        # Combine pre-2025 data with a portion of 2025 data
        df_train = pd.concat([df_pre_2025, df_2025_train])
        print(f"✅ Using {len(df_pre_2025)} samples from before 2025 and {len(df_2025_train)} samples from 2025 for training")
        
        # Save remaining 2025 data for later testing
        df_2025_test = df_2025.iloc[training_size_2025:]
        df_2025_test.to_csv("data/2025_holdout_test_data.csv")
        print(f"✅ Reserved {len(df_2025_test)} samples from 2025 for holdout testing")

    except FileNotFoundError:
        print("❌ Error: Dataset file not found.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # -----------------------
    # 2. Enhanced Feature Engineering
    # -----------------------
    print("Performing enhanced feature engineering...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df_train.copy()
    
    # Time-based features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    df['year'] = df.index.year  # Include year to capture long-term trends
    df['quarter'] = df.index.quarter
    df['is_weekend'] = df.index.dayofweek >= 5  # Weekend flag
    
    # Cyclical encoding of time features to preserve their circular nature
    # This helps the model understand that hour 23 is close to hour 0
    for col in ['hour', 'dayofweek', 'month', 'dayofyear']:
        max_val = df[col].max()
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    
    # Define the core numerical features for creating lags
    base_features = [
        'water_level', 'wind_speed', 'temperature', 'dew_point',
        'sea_level_pressure', 'rain', 'moon_illumination_fraction'
    ]
    
    # Add tide_level if it exists in the dataset
    if 'tide_level' in df.columns:
        base_features.append('tide_level')
        print("✅ Tide level feature included in the model")
    
    # Create time-lagged features with more lags for better time-series modeling
    lags = [1, 2, 3, 6, 12, 24]  # Added 12h and 24h lags for daily patterns
    for lag in lags:
        for feature in base_features:
            if feature in df.columns:
                df[f"{feature}_lag{lag}"] = df[feature].shift(lag)
    
    # Add rolling window statistics (capture trends and volatility)
    windows = [3, 6, 12, 24]
    for window in windows:
        for feature in base_features:
            if feature in df.columns:
                # Mean for trend
                df[f"{feature}_rolling_mean_{window}h"] = df[feature].rolling(window=window).mean()
                # Std for volatility
                df[f"{feature}_rolling_std_{window}h"] = df[feature].rolling(window=window).std()
                # Min/Max for extremes
                df[f"{feature}_rolling_min_{window}h"] = df[feature].rolling(window=window).min()
                df[f"{feature}_rolling_max_{window}h"] = df[feature].rolling(window=window).max()
    
    # Create interaction features between important variables
    if 'rain' in df.columns and 'tide_level' in df.columns:
        df['rain_tide_interaction'] = df['rain'] * df['tide_level']
    
    if 'wind_speed' in df.columns and 'rain' in df.columns:
        df['wind_rain_interaction'] = df['wind_speed'] * df['rain']
    
    # One-hot encode moon phase if present
    if 'moon_phase' in df.columns:
        df = pd.get_dummies(df, columns=['moon_phase'], prefix='phase')
    
    # Define the target variable (next hour water level)
    df['target'] = df['water_level'].shift(-1)
    
    # Remove rows with NaN values
    df = df.dropna()
    print(f"✅ Enhanced feature engineering complete. Dataset shape: {df.shape}")

    # -----------------------
    # 3. Feature Normalization/Scaling
    # -----------------------
    print("Applying feature scaling...")
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Save feature names for later use
    feature_names = X.columns.tolist()
    
    # Apply standard scaling to numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
    
    # Save the scaler for prediction time
    joblib.dump(scaler, "models/feature_scaler.pkl")
    print("✅ Feature scaling applied and scaler saved")

    # -----------------------
    # 4. Split data into Training and Testing sets
    # -----------------------
    # Use time-based split rather than random split
    train_size = int(len(X_scaled_df) * 0.8)
    X_train = X_scaled_df.iloc[:train_size]
    X_test = X_scaled_df.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]
    
    print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")

    # -----------------------
    # 5. Train an improved XGBoost model
    # -----------------------
    print("\nTraining the improved XGBoost model...")
    
    # Create a more robust model configuration
    model = XGBRegressor(
        n_estimators=500,       # More trees for better performance
        learning_rate=0.05,     # Lower learning rate for better generalization
        max_depth=6,            # Moderate depth to avoid overfitting
        min_child_weight=2,     # Helps with preventing overfitting
        subsample=0.8,          # Use 80% of data per tree to reduce overfitting
        colsample_bytree=0.8,   # Use 80% of features per tree
        gamma=0.1,              # Minimum loss reduction for split
        reg_alpha=0.1,          # L1 regularization
        reg_lambda=1.0,         # L2 regularization
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1               # Use all available cores
    )
    
    # Train the model with the most basic parameters
    model.fit(X_train, y_train)

    print("✅ Model training complete")
    
    # -----------------------
    # 6. Evaluate model performance
    # -----------------------
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance on Test Set:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # -----------------------
    # 7. Save the model and feature information
    # -----------------------
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model
    model_path = "models/floodsense_xgb_model_improved.pkl"
    joblib.dump(model, model_path)
    
    # Save feature names
    with open("models/model_features.txt", "w") as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print(f"✅ Improved model saved to {model_path}")
    print(f"✅ Feature names saved to models/model_features.txt")
    
    # Create a more informative plot
    plt.figure(figsize=(12, 6))
    
    # Plot only a subset of the test data for clarity
    plot_size = min(500, len(y_test))
    plt.plot(y_test.iloc[:plot_size].index, y_test.iloc[:plot_size].values, label='Actual Water Level', linewidth=2)
    plt.plot(y_test.iloc[:plot_size].index, y_pred[:plot_size], label='Predicted Water Level', linewidth=2, linestyle='--')
    
    plt.title('Improved Model Performance on Test Set', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Water Level', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('models/improved_model_performance.png')
    print("✅ Performance plot saved to models/improved_model_performance.png")
    
    return model, scaler, feature_names

if __name__ == "__main__":
    train_improved_model()
