# floodsense_xgb_trainer.py
# This script trains an XGBoost model using the pre-merged flood and moon data.
# It includes hyperparameter tuning and time-series cross-validation for improved accuracy.
# It assumes 'merged_flood_moon_data.csv' is in the same directory.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import joblib

def train_model_from_merged_data():
    """
    Loads the merged dataset, creates lag and time-based features, uses GridSearchCV
    with TimeSeriesSplit to find the best hyperparameters, trains the final model,
    evaluates its performance, and plots the results.
    """
    print("Starting the XGBoost model training pipeline...")
    
    # -----------------------
    # 1. Load and Filter the pre-merged dataset
    # -----------------------
    try:
        df = pd.read_csv("merged_flood_moon_data.csv", parse_dates=[0], index_col=0)
        print("✅ Successfully loaded 'merged_flood_moon_data.csv'.")

        # -- NEW: Filter out data from 2025 and later --
        df = df[df.index.year < 2025]
        print(f"✅ Data filtered to include only years before 2025. New date range: {df.index.min()} to {df.index.max()}")

    except FileNotFoundError:
        print("❌ Error: 'merged_flood_moon_data.csv' not found.")
        print("Please ensure the merged data file is in the same directory as this script.")
        return

    # -----------------------
    # 2. Feature Engineering
    # -----------------------
    print("Performing feature engineering (creating lag and time-based features)...")
    
    # Add time-based features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear

    # Define the core numerical features for creating lags
    base_features = [
        'water_level', 'wind_speed', 'temperature', 'dew_point',
        'sea_level_pressure', 'rain', 'moon_illumination_fraction'
    ]

    # Automatically find the one-hot encoded moon phase columns
    phase_columns = [col for col in df.columns if col.startswith('phase_')]
    
    # Combine all features to check for their existence
    all_features = base_features + phase_columns
    for feature in all_features:
        if feature not in df.columns:
            raise ValueError(f"Missing expected feature column in the CSV: {feature}")

    # Create time-lagged features
    lags = [1, 2, 3, 6]
    for lag in lags:
        for feature in base_features:
            df[f"{feature}_lag{lag}"] = df[feature].shift(lag)

    # Define the target variable
    df['target'] = df['water_level'].shift(-1)

    # Remove rows with NaN values
    df = df.dropna()
    print(f"✅ Feature engineering complete. Dataset shape for training: {df.shape}")

    # -----------------------
    # 3. Split data into Training and Testing sets
    # -----------------------
    X = df.drop(columns=['target'])
    y = df['target']

    # We still do a final hold-out test set split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Data split into training ({len(X_train)} rows) and final testing ({len(X_test)} rows) sets.")

    # -----------------------
    # 4. Hyperparameter Tuning with TimeSeriesSplit
    # -----------------------
    print("\nSetting up hyperparameter tuning with GridSearchCV...")
    
    # Define the cross-validation strategy for time series data
    tscv = TimeSeriesSplit(n_splits=5)

    # Define the grid of hyperparameters to search
    param_grid = {
        'max_depth': [5, 7],
        'n_estimators': [300, 500],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }

    # Initialize the XGBoost model
    model = XGBRegressor(random_state=42, objective='reg:squarederror')

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='r2',
        verbose=2,
        n_jobs=-1
    )

    # Run the search
    print("Starting grid search... This may take some time.")
    grid_search.fit(X_train, y_train)

    print(f"✅ Grid search complete. Best R² score found: {grid_search.best_score_:.4f}")
    print(f"Best parameters found: {grid_search.best_params_}")

    # Get the best model from the search
    best_model = grid_search.best_estimator_

    # Save the best model
    model_filename = "floodsense_xgb_model_tuned.pkl"
    joblib.dump(best_model, model_filename)
    print(f"✅ Best model saved as '{model_filename}'")

    # -----------------------
    # 5. Make Predictions and Evaluate Metrics on the Hold-out Test Set
    # -----------------------
    print("\nEvaluating the best model on the unseen test set...")
    y_pred = best_model.predict(X_test)
    
    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Final Model Performance ---")
    print(f"RMSE: {rmse:.3f} meters")
    print(f"R² Score: {r2:.4f}")
    print("-----------------------------\n")

    # -----------------------
    # 6. Plot Actual vs. Predicted Results
    # -----------------------
    print("Generating plot of actual vs. predicted water levels...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))
    
    plt.plot(y_test.index, y_test.values, label="Actual Water Level", color="dodgerblue", linewidth=2)
    plt.plot(y_test.index, y_pred, label="Predicted Water Level (Tuned)", color="red", linestyle='--', alpha=0.8)
    
    plt.title("Water Level Prediction (Tuned XGBoost with Moon Data)", fontsize=16, fontweight='bold')
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Water Level (m)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.text(0.02, 0.95, f"R² = {r2:.4f}", 
             transform=plt.gca().transAxes, 
             fontsize=12,
             verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    train_model_from_merged_data()
