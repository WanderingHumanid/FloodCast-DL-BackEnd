import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def validate_tidal_data_integration():
    """
    This function checks if tidal data is correctly integrated with the model
    and validates the effect of tidal data on prediction accuracy.
    """
    # Set up output directory
    output_dir = 'data_analysis_plots'
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Validating Tidal Data Integration ===")
    
    # Load models
    enhanced_model = joblib.load("models/floodsense_xgb_enhanced.pkl")
    try:
        original_model = joblib.load("models/floodsense_xgb_model_tuned.pkl")
        both_models_available = True
        print("✅ Both enhanced and original models loaded successfully")
    except:
        both_models_available = False
        print("⚠️ Only enhanced model available, cannot compare with original model")
    
    # Load data
    try:
        tidal_data = pd.read_csv("data/merged_flood_moon_tide_data.csv", index_col=0, parse_dates=True)
        print(f"✅ Loaded tidal dataset with {len(tidal_data)} records")
    except Exception as e:
        print(f"❌ Error loading tidal data: {e}")
        return False
    
    # Check for missing values in key columns
    missing_values = {
        'water_level': tidal_data['water_level'].isna().sum(),
        'tide_level': tidal_data['tide_level'].isna().sum() if 'tide_level' in tidal_data.columns else "Column missing",
        'moon_illumination_fraction': tidal_data['moon_illumination_fraction'].isna().sum() if 'moon_illumination_fraction' in tidal_data.columns else "Column missing"
    }
    
    print("\nMissing values check:")
    for col, count in missing_values.items():
        print(f"  {col}: {count}")
    
    # Fill missing tidal values if needed
    if 'tide_level' in tidal_data.columns and tidal_data['tide_level'].isna().sum() > 0:
        tidal_data['tide_level'] = tidal_data['tide_level'].fillna(method='ffill').fillna(method='bfill')
        print("  ⚠️ Filled missing tidal values")
    
    # Prepare features
    from app import prepare_time_features
    prepared_data = prepare_time_features(tidal_data)
    
    # Handle NaN values after feature preparation
    nan_count_before = prepared_data.isna().sum().sum()
    if nan_count_before > 0:
        print(f"  ⚠️ Found {nan_count_before} NaN values after feature preparation")
        # For testing, drop rows with NaN
        prepared_data = prepared_data.dropna()
        print(f"  ℹ️ Dropped rows with NaN values, {len(prepared_data)} records remaining")
    
    # Extract features that enhanced model uses
    model_features = enhanced_model.get_booster().feature_names
    
    # Check if all model features are available
    missing_features = [feat for feat in model_features if feat not in prepared_data.columns]
    if missing_features:
        print("\n⚠️ Missing features required by the model:")
        for feat in missing_features:
            print(f"  - {feat}")
        # Create dummy columns for missing features
        for feat in missing_features:
            prepared_data[feat] = 0
        print("  ℹ️ Created dummy columns for missing features with value 0")
    
    # Split into train/test
    test_size = min(1000, int(len(prepared_data) * 0.2))  # Use at most 1000 samples for testing
    train_data = prepared_data.iloc[:-test_size]
    test_data = prepared_data.iloc[-test_size:]
    
    print(f"\nUsing {len(test_data)} records for testing model performance")
    
    # Make predictions with enhanced model
    X_test = test_data[model_features]
    y_test = test_data['water_level']
    
    y_pred_enhanced = enhanced_model.predict(X_test)
    
    # Calculate metrics
    rmse_enhanced = np.sqrt(mean_squared_error(y_test, y_pred_enhanced))
    r2_enhanced = r2_score(y_test, y_pred_enhanced)
    
    print("\nEnhanced Model Performance:")
    print(f"  RMSE: {rmse_enhanced:.4f}")
    print(f"  R²: {r2_enhanced:.4f}")
    
    # If original model is available, compare performance
    if both_models_available:
        # Get features the original model uses
        original_features = original_model.get_booster().feature_names
        
        # Check and create any missing features
        for feat in original_features:
            if feat not in prepared_data.columns:
                prepared_data[feat] = 0
        
        X_test_original = test_data[original_features]
        y_pred_original = original_model.predict(X_test_original)
        
        rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))
        r2_original = r2_score(y_test, y_pred_original)
        
        print("\nOriginal Model Performance:")
        print(f"  RMSE: {rmse_original:.4f}")
        print(f"  R²: {r2_original:.4f}")
        
        print("\nPerformance Improvement:")
        rmse_improvement = (rmse_original - rmse_enhanced) / rmse_original * 100
        r2_improvement = (r2_enhanced - r2_original) / max(0.001, r2_original) * 100
        
        print(f"  RMSE Reduction: {rmse_improvement:.2f}%")
        print(f"  R² Improvement: {r2_improvement:.2f}%")
        
        # Plot comparison
        plt.figure(figsize=(14, 8))
        
        # Actual vs Predicted - Enhanced Model
        plt.subplot(2, 2, 1)
        plt.scatter(y_test, y_pred_enhanced, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Water Level')
        plt.ylabel('Predicted Water Level')
        plt.title(f'Enhanced Model: R² = {r2_enhanced:.4f}')
        
        # Actual vs Predicted - Original Model
        plt.subplot(2, 2, 2)
        plt.scatter(y_test, y_pred_original, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Water Level')
        plt.ylabel('Predicted Water Level')
        plt.title(f'Original Model: R² = {r2_original:.4f}')
        
        # Prediction Error Distribution - Enhanced Model
        plt.subplot(2, 2, 3)
        error_enhanced = y_test - y_pred_enhanced
        plt.hist(error_enhanced, bins=50, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'Enhanced Model Error: RMSE = {rmse_enhanced:.4f}')
        
        # Prediction Error Distribution - Original Model
        plt.subplot(2, 2, 4)
        error_original = y_test - y_pred_original
        plt.hist(error_original, bins=50, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'Original Model Error: RMSE = {rmse_original:.4f}')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/tidal_vs_non_tidal_model_comparison.png')
        print(f"\n✅ Comparison plot saved to {output_dir}/tidal_vs_non_tidal_model_comparison.png")
    
    # Analyze the relationship between tidal data and prediction accuracy
    if 'tide_level' in tidal_data.columns:
        # Create bins of tide levels
        tide_bins = pd.qcut(test_data['tide_level'], 5, duplicates='drop')
        
        # Calculate error metrics per bin
        tide_bin_stats = []
        for tide_bin in tide_bins.unique():
            bin_mask = tide_bins == tide_bin
            if sum(bin_mask) > 0:  # Ensure we have samples in this bin
                bin_rmse = np.sqrt(mean_squared_error(
                    y_test[bin_mask], 
                    y_pred_enhanced[bin_mask]
                ))
                bin_data = {
                    'tide_range': tide_bin,
                    'min_tide': tide_bin.left,
                    'max_tide': tide_bin.right,
                    'sample_count': sum(bin_mask),
                    'rmse': bin_rmse
                }
                tide_bin_stats.append(bin_data)
        
        # Convert to DataFrame for easier analysis
        tide_stats_df = pd.DataFrame(tide_bin_stats)
        
        print("\nPrediction Error by Tide Level Range:")
        for _, row in tide_stats_df.iterrows():
            print(f"  Tide range {row['min_tide']:.1f}-{row['max_tide']:.1f}: RMSE = {row['rmse']:.4f} ({row['sample_count']} samples)")
        
        # Plot error by tide level
        plt.figure(figsize=(10, 6))
        plt.bar(
            range(len(tide_stats_df)), 
            tide_stats_df['rmse'],
            tick_label=[f"{r['min_tide']:.1f}-{r['max_tide']:.1f}" for _, r in tide_stats_df.iterrows()]
        )
        plt.xlabel('Tide Level Range')
        plt.ylabel('RMSE')
        plt.title('Prediction Error by Tide Level Range')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/prediction_error_by_tide_level.png')
        print(f"✅ Tide level error analysis saved to {output_dir}/prediction_error_by_tide_level.png")
    
    return True

if __name__ == "__main__":
    validate_tidal_data_integration()
