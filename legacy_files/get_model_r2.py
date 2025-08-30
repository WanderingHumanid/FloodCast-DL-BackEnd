import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def prepare_time_features(df):
    """Prepare time features for prediction"""
    df.index = pd.to_datetime(df.index)
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    df_copy['dayofweek'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    df_copy['dayofyear'] = df_copy.index.dayofyear
    
    # Create lag features
    base_features = ['water_level', 'wind_speed', 'temperature', 'dew_point', 'sea_level_pressure', 'rain', 'moon_illumination_fraction']
    lags = [1, 2, 3, 6]
    for lag in lags:
        for feature in base_features:
            df_copy[f"{feature}_lag{lag}"] = df_copy[feature].shift(lag)
    
    # One-hot encode moon phases if they exist as boolean columns
    moon_phase_cols = [col for col in df_copy.columns if col.startswith('phase_')]
    if not moon_phase_cols:
        # If no moon phase columns, try to create them
        if 'moon_phase' in df_copy.columns:
            df_copy = pd.get_dummies(df_copy, columns=['moon_phase'], prefix='phase')
    
    return df_copy

# Load the dataset and models
try:
    print("Loading data and models...")
    df = pd.read_csv('data/merged_flood_moon_data.csv', index_col=0, parse_dates=True)
    regression_model = joblib.load('models/floodsense_xgb_model_tuned.pkl')
    classifier_model = joblib.load('models/floodsense_spatio_temporal_classifier.pkl')
    
    print("\nPreparing features...")
    prepared_df = prepare_time_features(df).dropna()
    print(f"Features prepared. Shape: {prepared_df.shape}")
    
    # Get the features used by the regression model
    regression_features = regression_model.get_booster().feature_names
    print(f"\nRegression model features: {len(regression_features)}")
    
    # Check which features we have available
    available_features = [f for f in regression_features if f in prepared_df.columns]
    missing_features = [f for f in regression_features if f not in prepared_df.columns]
    
    print(f"Available features: {len(available_features)}/{len(regression_features)}")
    if missing_features:
        print(f"Missing features: {missing_features}")
    
    # Use last 20% of data for testing
    split_idx = int(len(prepared_df) * 0.8)
    train_df = prepared_df.iloc[:split_idx]
    test_df = prepared_df.iloc[split_idx:]
    
    print("\nEvaluating on test set...")
    
    # If there are missing features, we'll need to add them
    if missing_features:
        for feature in missing_features:
            prepared_df[feature] = 0
            test_df[feature] = 0
    
    # Ensure column order matches model expectation
    X_test = test_df[regression_features]
    y_test = test_df['water_level']
    
    # Predict and evaluate
    y_pred = regression_model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nRegression Model Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Check if correlation patterns make sense
    correlation = df[['rain', 'water_level']].corr()
    print("\nCorrelation between rain and water level:")
    print(correlation)
    
except Exception as e:
    print(f"Error: {e}")
