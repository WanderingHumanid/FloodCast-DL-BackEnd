import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from datetime import datetime
import os

# Create output directory for plots
plots_dir = 'data_analysis_plots'
os.makedirs(plots_dir, exist_ok=True)

print("Loading merged data with tide information...")
# Load the complete dataset with tidal data
df = pd.read_csv('data/merged_flood_moon_tide_data.csv', index_col=0, parse_dates=True)

# Make sure we don't have any NaN values
print(f"Missing values before cleaning:\n{df.isnull().sum()}")
df = df.dropna()
print(f"Total records after removing missing values: {len(df)}")

# Define features and target variable
features = [
    'rain', 'wind_speed', 'temperature', 
    'dew_point', 'sea_level_pressure',
    'moon_illumination_fraction',
    'tide_level'  # Adding tidal data as a new feature
]

# Add moon phase columns if they exist in the dataset
moon_phase_columns = [col for col in df.columns if col.startswith('phase_')]
features.extend(moon_phase_columns)

print(f"Using these features: {features}")
X = df[features]
y = df['water_level']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining XGBoost model with tidal data...")
# Train the model
model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model performance with tidal data:")
print(f"R² score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Save the model
model_filename = 'models/floodsense_xgb_with_tidal_data.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_filename}")

# Feature importance
plt.figure(figsize=(12, 6))
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.title('Feature Importance (with Tidal Data)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(f'{plots_dir}/feature_importance_with_tidal_data.png')

# Compare predictions with and without tidal data
print("\nLoading original model for comparison...")
try:
    with open('models/floodsense_xgb_model_tuned.pkl', 'rb') as f:
        original_model = pickle.load(f)
    
    # Create features without tide_level for original model
    features_no_tide = [f for f in features if f != 'tide_level']
    X_test_no_tide = X_test[features_no_tide]
    
    # Predict with original model
    y_pred_original = original_model.predict(X_test_no_tide)
    
    # Calculate metrics for original model
    r2_original = r2_score(y_test, y_pred_original)
    rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))
    
    print(f"Original model performance (without tidal data):")
    print(f"R² score: {r2_original:.4f}")
    print(f"RMSE: {rmse_original:.4f}")
    
    # Plot comparison of predictions
    plt.figure(figsize=(12, 6))
    
    # Sample 100 random data points for clearer visualization
    sample_size = 100
    sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    plt.scatter(y_test.iloc[sample_indices], y_pred_original[sample_indices], 
                alpha=0.5, label=f'Original Model (R²={r2_original:.4f})')
    plt.scatter(y_test.iloc[sample_indices], y_pred[sample_indices], 
                alpha=0.5, label=f'Model with Tidal Data (R²={r2:.4f})')
    
    # Add perfect prediction line
    min_val = min(y_test.iloc[sample_indices].min(), y_pred.min(), y_pred_original.min())
    max_val = max(y_test.iloc[sample_indices].max(), y_pred.max(), y_pred_original.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.xlabel('Actual Water Level')
    plt.ylabel('Predicted Water Level')
    plt.title('Comparison of Model Predictions With and Without Tidal Data')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/model_comparison_with_tidal_data.png')
    
    # Calculate improvement
    r2_improvement = ((r2 - r2_original) / r2_original) * 100
    rmse_improvement = ((rmse_original - rmse) / rmse_original) * 100
    
    print(f"\nImprovement with tidal data:")
    print(f"R² improvement: {r2_improvement:.2f}%")
    print(f"RMSE improvement: {rmse_improvement:.2f}%")
    
except Exception as e:
    print(f"Could not load original model for comparison: {e}")

print("\nAnalysis and model training complete.")
