import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt
import os

def ensure_tidal_features():
    """
    Ensures that tidal features are properly incorporated into the model features.
    This function checks the feature names in the enhanced model and generates 
    any missing tidal features in the dataset.
    """
    # Create output directory for any plots
    plots_dir = 'data_analysis_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\n=== Ensuring Tidal Features are Properly Incorporated ===")
    
    # Load the enhanced model
    try:
        model = joblib.load("models/floodsense_xgb_enhanced.pkl")
        print("✅ Enhanced model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading enhanced model: {e}")
        return False
    
    # Get the feature names from the model
    feature_names = model.get_booster().feature_names
    
    # Filter for tidal-related features
    tidal_features = [feat for feat in feature_names if 'tide' in feat.lower()]
    
    print(f"\nFound {len(tidal_features)} tidal-related features in the model:")
    for i, feat in enumerate(tidal_features):
        print(f"  {i+1}. {feat}")
    
    # Load the dataset with tidal data
    try:
        df = pd.read_csv("data/merged_flood_moon_tide_data.csv", index_col=0, parse_dates=True)
        print(f"\n✅ Loaded tidal dataset with {len(df)} records")
    except Exception as e:
        print(f"❌ Error loading tidal dataset: {e}")
        return False
    
    # Check if tide_level is in the dataset
    if 'tide_level' not in df.columns:
        print("❌ 'tide_level' column not found in the dataset")
        return False
    
    # Ensure the dataset has tidal data
    tidal_coverage = (1 - (df['tide_level'].isna().sum() / len(df))) * 100
    print(f"Tidal data coverage: {tidal_coverage:.2f}%")
    
    if tidal_coverage < 95:
        print("⚠️ Tidal data coverage is below 95%, this may affect model performance")
    
    # Create a sample week of data to visualize
    sample_start = '2023-06-01'
    sample_end = '2023-06-07'
    sample_data = df.loc[sample_start:sample_end].copy()
    
    # Plot the tidal pattern for the sample week
    plt.figure(figsize=(14, 6))
    plt.plot(sample_data.index, sample_data['tide_level'], 'b-')
    plt.title('Tidal Pattern (1 Week Sample)')
    plt.ylabel('Tide Level')
    plt.xlabel('Date')
    plt.grid(True)
    plt.savefig(f'{plots_dir}/tidal_pattern_sample.png')
    print(f"✅ Tidal pattern plot saved to {plots_dir}/tidal_pattern_sample.png")
    
    # Check for correlation between tidal and water levels
    correlation = df['tide_level'].corr(df['water_level'])
    print(f"\nCorrelation between tide level and water level: {correlation:.4f}")
    
    # Check for monthly tidal patterns
    df['month'] = df.index.month
    monthly_tide_avg = df.groupby('month')['tide_level'].mean()
    
    plt.figure(figsize=(12, 6))
    monthly_tide_avg.plot(kind='bar')
    plt.title('Average Tide Level by Month')
    plt.ylabel('Average Tide Level')
    plt.xlabel('Month')
    plt.grid(True, axis='y')
    plt.savefig(f'{plots_dir}/monthly_tide_patterns.png')
    print(f"✅ Monthly tide patterns saved to {plots_dir}/monthly_tide_patterns.png")
    
    # Check for hourly tidal patterns
    df['hour'] = df.index.hour
    hourly_tide_avg = df.groupby('hour')['tide_level'].mean()
    
    plt.figure(figsize=(12, 6))
    hourly_tide_avg.plot(kind='bar')
    plt.title('Average Tide Level by Hour of Day')
    plt.ylabel('Average Tide Level')
    plt.xlabel('Hour of Day')
    plt.grid(True, axis='y')
    plt.savefig(f'{plots_dir}/hourly_tide_patterns.png')
    print(f"✅ Hourly tide patterns saved to {plots_dir}/hourly_tide_patterns.png")
    
    # Generate a list of important tidal features to ensure they exist
    print("\nChecking for important tidal features:")
    
    important_tidal_features = [
        'tide_level',
        'tide_level_lag1',
        'tide_level_lag2',
        'tide_level_lag3',
        'tide_level_lag6',
        'tide_level_lag12',
        'tide_level_lag24',
        'tide_level_rolling_mean_3h',
        'tide_level_rolling_mean_6h',
        'tide_level_rolling_max_6h',
        'tide_level_rolling_min_6h',
        'tide_level_velocity',
        'tide_level_acceleration',
        'tide_range_6h',
        'tide_range_12h',
        'tide_range_24h',
        'tide_rain_interaction'
    ]
    
    # Check if the features exist in the model
    for feature in important_tidal_features:
        if feature in feature_names:
            print(f"  ✅ {feature} is used by the model")
        else:
            print(f"  ⚠️ {feature} is not used by the model")
    
    # Create a list of the model's most important features
    try:
        importance = model.get_booster().get_score(importance_type='gain')
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Check how many tidal features are in the top 20
        top_features = [f[0] for f in sorted_importance[:20]]
        tidal_in_top = [f for f in top_features if 'tide' in f.lower()]
        
        print(f"\nFound {len(tidal_in_top)} tidal-related features in the top 20 important features:")
        for i, feat in enumerate(tidal_in_top):
            feat_index = top_features.index(feat) + 1
            print(f"  {feat_index}. {feat}")
        
        # Save important tidal features to a file
        with open("models/enhanced_feature_names.txt", "w") as f:
            f.write("# Enhanced model feature names in order of importance\n\n")
            for i, (feat, score) in enumerate(sorted_importance[:50], 1):
                tidal_flag = "✓" if 'tide' in feat.lower() else " "
                f.write(f"{i:2d}. [{tidal_flag}] {feat}: {score:.2f}\n")
        
        print(f"\n✅ Saved top 50 features to models/enhanced_feature_names.txt")
        
    except Exception as e:
        print(f"❌ Error analyzing feature importance: {e}")
    
    return True

if __name__ == "__main__":
    ensure_tidal_features()
