"""
FloodCast Model Improvement Script
- Enhances R² value through advanced feature engineering
- Improves lead time by implementing early warning techniques
- Uses a more sophisticated model architecture with time series components
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create output directories
plots_dir = 'data_analysis_plots'
models_dir = 'models'
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

def load_data():
    """Load and prepare the dataset with tidal information"""
    logging.info("Loading merged data with tide information...")
    
    # Load the complete dataset with tidal data
    df = pd.read_csv('data/merged_flood_moon_tide_data.csv', index_col=0, parse_dates=True)
    
    # Check and handle missing values
    logging.info(f"Missing values before cleaning:\n{df.isnull().sum()}")
    df = df.dropna()
    logging.info(f"Total records after removing missing values: {len(df)}")
    
    return df

def engineer_advanced_features(df):
    """
    Create advanced features to improve model performance:
    1. Time-based features (hourly, daily, seasonal patterns)
    2. Interaction features between tide, rain, and moon
    3. Lag features with longer windows
    4. Rolling statistics features
    5. Rate of change features
    """
    logging.info("Performing advanced feature engineering...")
    
    # Make a copy to avoid modifying original
    enhanced_df = df.copy()
    
    # 1. Enhanced time-based features
    enhanced_df['hour'] = enhanced_df.index.hour
    enhanced_df['day'] = enhanced_df.index.day
    enhanced_df['dayofweek'] = enhanced_df.index.dayofweek
    enhanced_df['month'] = enhanced_df.index.month
    enhanced_df['quarter'] = enhanced_df.index.quarter
    enhanced_df['dayofyear'] = enhanced_df.index.dayofyear
    enhanced_df['weekofyear'] = enhanced_df.index.isocalendar().week
    
    # Time of day encoding using sine and cosine for cyclical nature
    enhanced_df['hour_sin'] = np.sin(2 * np.pi * enhanced_df['hour']/24)
    enhanced_df['hour_cos'] = np.cos(2 * np.pi * enhanced_df['hour']/24)
    
    # Day of year encoding using sine and cosine for cyclical nature
    enhanced_df['day_of_year_sin'] = np.sin(2 * np.pi * enhanced_df['dayofyear']/365)
    enhanced_df['day_of_year_cos'] = np.cos(2 * np.pi * enhanced_df['dayofyear']/365)
    
    # 2. Interaction features
    enhanced_df['tide_rain_interaction'] = enhanced_df['tide_level'] * enhanced_df['rain']
    enhanced_df['tide_moon_interaction'] = enhanced_df['tide_level'] * enhanced_df['moon_illumination_fraction']
    enhanced_df['rain_moon_interaction'] = enhanced_df['rain'] * enhanced_df['moon_illumination_fraction']
    
    # 3. Enhanced lag features (more lags and different windows)
    base_features = ['water_level', 'rain', 'tide_level', 'moon_illumination_fraction', 
                   'temperature', 'wind_speed', 'sea_level_pressure']
    
    # Wider range of lags - hourly, daily, and weekly for tidal patterns
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]  # hours
    
    for lag in lags:
        for feature in base_features:
            enhanced_df[f"{feature}_lag{lag}"] = enhanced_df[feature].shift(lag)
    
    # 4. Rolling statistics features
    windows = [3, 6, 12, 24, 48]  # hours
    
    for window in windows:
        for feature in ['water_level', 'rain', 'tide_level']:
            # Rolling mean
            enhanced_df[f"{feature}_rolling_mean_{window}h"] = enhanced_df[feature].rolling(window=window).mean()
            
            # Rolling max - important for peak detection
            enhanced_df[f"{feature}_rolling_max_{window}h"] = enhanced_df[feature].rolling(window=window).max()
            
            # Rolling min
            enhanced_df[f"{feature}_rolling_min_{window}h"] = enhanced_df[feature].rolling(window=window).min()
            
            # Rolling standard deviation - captures volatility
            enhanced_df[f"{feature}_rolling_std_{window}h"] = enhanced_df[feature].rolling(window=window).std()
    
    # 5. Rate of change features (derivatives)
    for feature in ['water_level', 'rain', 'tide_level']:
        # First-order difference (velocity)
        enhanced_df[f"{feature}_velocity"] = enhanced_df[feature].diff()
        
        # Second-order difference (acceleration)
        enhanced_df[f"{feature}_acceleration"] = enhanced_df[feature].diff().diff()
    
    # 6. Tidal specific features
    # Calculate tidal range (difference between consecutive high and low tides)
    enhanced_df['tide_range_6h'] = enhanced_df['tide_level'].rolling(window=6).max() - enhanced_df['tide_level'].rolling(window=6).min()
    enhanced_df['tide_range_12h'] = enhanced_df['tide_level'].rolling(window=12).max() - enhanced_df['tide_level'].rolling(window=12).min()
    enhanced_df['tide_range_24h'] = enhanced_df['tide_level'].rolling(window=24).max() - enhanced_df['tide_level'].rolling(window=24).min()
    
    # Remove rows with NaN values from feature engineering
    enhanced_df = enhanced_df.dropna()
    logging.info(f"Dataset after feature engineering: {enhanced_df.shape} rows, {enhanced_df.shape[1]} features")
    
    return enhanced_df

def optimize_hyperparameters(X_train, y_train):
    """Perform hyperparameter optimization for the XGBoost model"""
    logging.info("Performing hyperparameter optimization...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Use a small grid for initial testing - expand for production
    small_param_grid = {
        'n_estimators': [200, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [6, 8],
        'subsample': [0.8, 0.9]
    }
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=small_param_grid,  # Use small_param_grid for faster results
        scoring='r2',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    logging.info(f"Best R² score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_lead_time(model, df, lead_times):
    """
    Evaluate model performance at different lead times
    
    Args:
        model: Trained XGBoost model
        df: DataFrame with features
        lead_times: List of lead times in hours to test
    
    Returns:
        DataFrame with performance metrics for each lead time
    """
    logging.info("Evaluating model performance across different lead times...")
    
    lead_time_results = []
    
    for lead_time in lead_times:
        logging.info(f"Testing lead time of {lead_time} hours")
        
        # Create shifted targets for this lead time
        target_column = f'water_level_lead_{lead_time}h'
        df[target_column] = df['water_level'].shift(-lead_time)
        
        # Drop rows with NaN in the target
        lead_time_df = df.dropna(subset=[target_column])
        
        # Use the same features as the main model
        X_lead = lead_time_df[model.feature_names_in_]
        y_lead = lead_time_df[target_column]
        
        # Split data
        X_train_lead, X_test_lead, y_train_lead, y_test_lead = train_test_split(
            X_lead, y_lead, test_size=0.2, random_state=42
        )
        
        # Train a new model for this lead time
        lead_model = xgb.XGBRegressor(
            n_estimators=model.n_estimators,
            learning_rate=model.learning_rate,
            max_depth=model.max_depth,
            subsample=model.subsample,
            colsample_bytree=model.colsample_bytree,
            random_state=42
        )
        
        lead_model.fit(X_train_lead, y_train_lead)
        
        # Make predictions
        y_pred_lead = lead_model.predict(X_test_lead)
        
        # Calculate metrics
        r2 = r2_score(y_test_lead, y_pred_lead)
        rmse = np.sqrt(mean_squared_error(y_test_lead, y_pred_lead))
        mae = mean_absolute_error(y_test_lead, y_pred_lead)
        
        lead_time_results.append({
            'lead_time_hours': lead_time,
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae
        })
        
        # Save the model for this lead time
        model_filename = f'models/floodsense_xgb_lead_time_{lead_time}h.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(lead_model, f)
        
        logging.info(f"Lead time {lead_time}h - R²: {r2:.4f}, RMSE: {rmse:.4f}")
    
    results_df = pd.DataFrame(lead_time_results)
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(results_df['lead_time_hours'], results_df['r2_score'], 'o-', linewidth=2)
    plt.title('R² Score by Lead Time')
    plt.xlabel('Lead Time (hours)')
    plt.ylabel('R² Score')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(results_df['lead_time_hours'], results_df['rmse'], 'o-', linewidth=2, color='red')
    plt.title('RMSE by Lead Time')
    plt.xlabel('Lead Time (hours)')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/lead_time_performance.png')
    
    return results_df

def main():
    """Main execution function"""
    # Load and prepare data
    df = load_data()
    
    # Engineer advanced features
    enhanced_df = engineer_advanced_features(df)
    
    # Define features and target
    exclude_cols = ['water_level']
    features = [col for col in enhanced_df.columns if col not in exclude_cols]
    
    X = enhanced_df[features]
    y = enhanced_df['water_level']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optimize hyperparameters
    best_model = optimize_hyperparameters(X_train, y_train)
    
    # Train the model with best parameters
    logging.info("Training final model with optimal parameters...")
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    logging.info(f"Enhanced model performance:")
    logging.info(f"R² score: {r2:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    
    # Save the enhanced model
    model_filename = 'models/floodsense_xgb_enhanced.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
    logging.info(f"Enhanced model saved to {model_filename}")
    
    # Save feature names for reference
    with open('models/enhanced_feature_names.txt', 'w') as f:
        f.write('\n'.join(features))
    
    # Feature importance
    plt.figure(figsize=(12, 8))
    feature_importance = best_model.feature_importances_
    
    # Get the top 25 features
    indices = np.argsort(feature_importance)[-25:]
    plt.barh(range(len(indices)), feature_importance[indices])
    plt.yticks(range(len(indices)), np.array(features)[indices])
    plt.title('Top 25 Feature Importance (Enhanced Model)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/enhanced_feature_importance.png')
    
    # Evaluate lead time performance
    lead_times = [1, 3, 6, 12, 18, 24, 36, 48]
    lead_time_results = evaluate_lead_time(best_model, enhanced_df, lead_times)
    
    # Save lead time results
    lead_time_results.to_csv('data/lead_time_performance.csv', index=False)
    logging.info("Lead time performance results saved to 'data/lead_time_performance.csv'")
    
    # Create an early warning threshold analysis
    logging.info("Analyzing early warning thresholds...")
    analyze_early_warning_thresholds(enhanced_df, best_model, X_test, y_test, y_pred)
    
    logging.info("Model enhancement completed.")

def analyze_early_warning_thresholds(df, model, X_test, y_test, y_pred):
    """
    Analyze different thresholds for early warning to optimize lead time
    while maintaining acceptable false alarm rates
    """
    # Calculate prediction errors
    errors = y_test - y_pred
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{plots_dir}/prediction_error_distribution.png')
    
    # Define different thresholds for early warning
    thresholds = [50, 100, 150, 200, 250, 300]
    
    # Calculate potential lead time improvements with adjusted thresholds
    threshold_results = []
    
    for threshold in thresholds:
        # Consider an alert if predicted value + threshold exceeds a critical level
        critical_level = 700  # Water level threshold for flooding
        
        # Count correct alerts (true positives)
        true_positives = sum((y_pred + threshold > critical_level) & (y_test > critical_level))
        
        # Count false alerts (false positives)
        false_positives = sum((y_pred + threshold > critical_level) & (y_test <= critical_level))
        
        # Count missed alerts (false negatives)
        false_negatives = sum((y_pred + threshold <= critical_level) & (y_test > critical_level))
        
        # Count correct non-alerts (true negatives)
        true_negatives = sum((y_pred + threshold <= critical_level) & (y_test <= critical_level))
        
        # Calculate metrics
        total_alerts = true_positives + false_positives
        if total_alerts > 0:
            precision = true_positives / total_alerts
        else:
            precision = 0
            
        if (true_positives + false_negatives) > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0
            
        if (precision + recall) > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0
        
        false_alarm_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        
        threshold_results.append({
            'threshold': threshold,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_alarm_rate': false_alarm_rate
        })
    
    # Convert to DataFrame
    threshold_df = pd.DataFrame(threshold_results)
    
    # Plot threshold analysis
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(threshold_df['threshold'], threshold_df['precision'], 'o-', label='Precision')
    plt.plot(threshold_df['threshold'], threshold_df['recall'], 's-', label='Recall')
    plt.plot(threshold_df['threshold'], threshold_df['f1_score'], '^-', label='F1 Score')
    plt.title('Alert Performance by Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(threshold_df['threshold'], threshold_df['false_alarm_rate'], 'o-', color='red')
    plt.title('False Alarm Rate by Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('False Alarm Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/early_warning_threshold_analysis.png')
    
    # Save threshold analysis results
    threshold_df.to_csv('data/early_warning_threshold_analysis.csv', index=False)
    logging.info("Early warning threshold analysis saved to 'data/early_warning_threshold_analysis.csv'")

if __name__ == "__main__":
    main()
