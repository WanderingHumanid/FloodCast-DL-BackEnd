import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# Create output directory for plots
plots_dir = 'data_analysis_plots'
os.makedirs(plots_dir, exist_ok=True)

print("Loading test data...")
# Load the complete dataset with tidal data
df = pd.read_csv('data/merged_flood_moon_tide_data.csv', index_col=0, parse_dates=True)
df = df.dropna()

# Create test set (last 20% of the data)
test_size = int(len(df) * 0.2)
test_df = df.iloc[-test_size:]

# Load the tidal model
print("Loading model with tidal data...")
try:
    with open('models/floodsense_xgb_with_tidal_data.pkl', 'rb') as f:
        tidal_model = pickle.load(f)
    
    # Features with tidal data
    tidal_features = [
        'rain', 'wind_speed', 'temperature', 
        'dew_point', 'sea_level_pressure',
        'moon_illumination_fraction',
        'tide_level'
    ]
    
    # Add moon phase columns if they exist in the dataset
    moon_phase_columns = [col for col in df.columns if col.startswith('phase_')]
    tidal_features.extend(moon_phase_columns)
    
    # Make predictions with tidal model
    X_test_tidal = test_df[tidal_features]
    y_test = test_df['water_level']
    y_pred_tidal = tidal_model.predict(X_test_tidal)
    
    # Calculate metrics for tidal model
    r2_tidal = r2_score(y_test, y_pred_tidal)
    mae_tidal = mean_absolute_error(y_test, y_pred_tidal)
    rmse_tidal = np.sqrt(mean_squared_error(y_test, y_pred_tidal))
    
    print(f"Tidal model performance:")
    print(f"R² score: {r2_tidal:.4f}")
    print(f"MAE: {mae_tidal:.4f}")
    print(f"RMSE: {rmse_tidal:.4f}")
    
    # Train a model without tidal data for comparison
    print("\nTraining model without tidal data for comparison...")
    import xgboost as xgb
    
    # Features without tidal data
    non_tidal_features = [f for f in tidal_features if f != 'tide_level']
    X_test_non_tidal = test_df[non_tidal_features]
    
    # Train a model without tidal data
    X_train_non_tidal = df.iloc[:-test_size][non_tidal_features]
    y_train = df.iloc[:-test_size]['water_level']
    
    non_tidal_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    non_tidal_model.fit(X_train_non_tidal, y_train)
    
    # Make predictions with non-tidal model
    y_pred_non_tidal = non_tidal_model.predict(X_test_non_tidal)
    
    # Calculate metrics for non-tidal model
    r2_non_tidal = r2_score(y_test, y_pred_non_tidal)
    mae_non_tidal = mean_absolute_error(y_test, y_pred_non_tidal)
    rmse_non_tidal = np.sqrt(mean_squared_error(y_test, y_pred_non_tidal))
    
    print(f"Non-tidal model performance:")
    print(f"R² score: {r2_non_tidal:.4f}")
    print(f"MAE: {mae_non_tidal:.4f}")
    print(f"RMSE: {rmse_non_tidal:.4f}")
    
    # Calculate improvement
    r2_improvement = ((r2_tidal - r2_non_tidal) / r2_non_tidal) * 100
    rmse_improvement = ((rmse_non_tidal - rmse_tidal) / rmse_non_tidal) * 100
    
    print(f"\nImprovement with tidal data:")
    print(f"R² improvement: {r2_improvement:.2f}%")
    print(f"RMSE improvement: {rmse_improvement:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Predicted for both models
    plt.subplot(2, 2, 1)
    
    # Sample 100 points for clarity
    sample_size = 100
    sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    plt.scatter(y_test.iloc[sample_indices], y_pred_non_tidal[sample_indices], 
                alpha=0.5, label=f'Without Tidal Data (R²={r2_non_tidal:.4f})', color='red')
    plt.scatter(y_test.iloc[sample_indices], y_pred_tidal[sample_indices], 
                alpha=0.5, label=f'With Tidal Data (R²={r2_tidal:.4f})', color='blue')
    
    # Perfect prediction line
    min_val = min(y_test.iloc[sample_indices].min(), 
                  y_pred_tidal[sample_indices].min(), 
                  y_pred_non_tidal[sample_indices].min())
    max_val = max(y_test.iloc[sample_indices].max(), 
                  y_pred_tidal[sample_indices].max(), 
                  y_pred_non_tidal[sample_indices].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.xlabel('Actual Water Level')
    plt.ylabel('Predicted Water Level')
    plt.title('Model Comparison: Actual vs Predicted')
    plt.legend()
    
    # Plot 2: Prediction Error Distribution
    plt.subplot(2, 2, 2)
    
    error_non_tidal = y_test.values - y_pred_non_tidal
    error_tidal = y_test.values - y_pred_tidal
    
    plt.hist(error_non_tidal, bins=50, alpha=0.5, label='Without Tidal Data', color='red')
    plt.hist(error_tidal, bins=50, alpha=0.5, label='With Tidal Data', color='blue')
    
    plt.xlabel('Prediction Error (cm)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution Comparison')
    plt.legend()
    
    # Plot 3: Time Series of Predictions
    plt.subplot(2, 1, 2)
    
    # Select a week of data for visualization
    week_data = test_df.iloc[:168]  # First week (7 days * 24 hours)
    week_predictions_tidal = tidal_model.predict(week_data[tidal_features])
    week_predictions_non_tidal = non_tidal_model.predict(week_data[non_tidal_features])
    
    plt.plot(week_data.index, week_data['water_level'], 'k-', label='Actual Water Level')
    plt.plot(week_data.index, week_predictions_tidal, 'b-', label='Predicted with Tidal Data')
    plt.plot(week_data.index, week_predictions_non_tidal, 'r-', label='Predicted without Tidal Data')
    
    # Add tide level on a second axis
    ax2 = plt.twinx()
    ax2.plot(week_data.index, week_data['tide_level'], 'g--', label='Tide Level', alpha=0.3)
    ax2.set_ylabel('Tide Level', color='g')
    
    plt.xlabel('Date')
    plt.ylabel('Water Level (cm)')
    plt.title('One Week Prediction Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/tidal_vs_non_tidal_model_comparison.png')
    
    # Save the non-tidal model for future use
    with open('models/floodsense_xgb_without_tidal_data.pkl', 'wb') as f:
        pickle.dump(non_tidal_model, f)
    print(f"Non-tidal model saved to models/floodsense_xgb_without_tidal_data.pkl")
    
    print("\nModel comparison complete.")
    
except Exception as e:
    print(f"Error in model comparison: {e}")
