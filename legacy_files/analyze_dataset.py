import pandas as pd
import numpy as np
from scipy import stats

# Load the dataset
df = pd.read_csv('data/merged_flood_moon_data.csv', index_col=0, parse_dates=True)

# Basic dataset info
print(f'Time range: {df.index.min()} to {df.index.max()}')
print(f'Total records: {len(df)}')
print(f'Missing values:\n{df.isna().sum()}')
print(f'Date range frequency: {pd.infer_freq(df.index)}')

# Check for gaps in the time series
expected_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
missing_dates = expected_dates.difference(df.index)
print(f'\nNumber of missing hourly records: {len(missing_dates)}')
if len(missing_dates) > 0:
    print(f'First 10 missing dates: {missing_dates[:10]}')

# Check for duplicates
duplicates = df.index.duplicated()
print(f'\nNumber of duplicate timestamps: {duplicates.sum()}')
if duplicates.sum() > 0:
    dup_indices = df.index[duplicates]
    print(f'First 10 duplicate timestamps: {dup_indices[:10]}')

# Get distribution by year and month to check for data imbalance
print("\nData distribution by year:")
print(df.index.year.value_counts().sort_index())

print("\nData distribution by month:")
print(df.index.month.value_counts().sort_index())

# Check for repeating patterns (modified to handle index properly)
print("\nChecking for exact repeating patterns in the data...")
sample_cols = ['rain', 'water_level']
for col in sample_cols:
    # Compare day 1 with day 2
    day1 = df[col].iloc[0:24].values
    day2 = df[col].iloc[24:48].values
    is_identical = np.array_equal(day1, day2)
    print(f"First 24 hours of {col} identical to next 24 hours: {is_identical}")

# Descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# Check for extreme values or outliers
print("\nChecking for outliers (values > 3 standard deviations):")
for col in df.select_dtypes(include=[np.number]).columns:
    if col.startswith('phase_'):  # Skip binary columns
        continue
    mean = df[col].mean()
    std = df[col].std()
    threshold = 3 * std
    outliers = df[np.abs(df[col] - mean) > threshold]
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")

# Look for potential data quality issues
print("\nChecking for potential data quality issues:")
# Check for zero rainfall when water level increases significantly
water_level_increases = df['water_level'].diff() > 5  # Significant increase in water level
zero_rain = df['rain'] == 0
suspicious = water_level_increases & zero_rain
print(f"Times when water level increased by >5 units with zero rain: {suspicious.sum()} records")

# Load labeled_flood_data to check the model's R² if available
try:
    labeled_df = pd.read_csv('data/labeled_flood_data.csv', index_col=0, parse_dates=True)
    print("\nLabeled data statistics:")
    print(f"Total records: {len(labeled_df)}")
    flood_events = labeled_df['flood_event'].sum()
    print(f"Flood events: {flood_events} ({flood_events/len(labeled_df)*100:.2f}%)")
except Exception as e:
    print(f"\nCouldn't analyze labeled data: {e}")

# Try to find R² value from model validation files
try:
    # Load validation_metrics.py to see if R² is there
    with open('utils/validation_metrics.py', 'r') as f:
        validation_code = f.read()
        if 'r2_score' in validation_code:
            print("\nFound r2_score in validation_metrics.py")
except Exception as e:
    print(f"\nCouldn't check validation metrics: {e}")

# Try to run model validation to get R² value
try:
    import joblib
    from sklearn.metrics import r2_score, mean_squared_error
    
    model = joblib.load('models/floodsense_xgb_model_tuned.pkl')
    
    # Take a sample for validation (last 20% of data)
    test_size = int(len(df) * 0.2)
    test_df = df.iloc[-test_size:]
    
    # Prepare features (similar to what's in app.py)
    test_df_features = test_df.copy()
    test_df_features['hour'] = test_df_features.index.hour
    test_df_features['dayofweek'] = test_df_features.index.dayofweek
    test_df_features['month'] = test_df_features.index.month
    
    # Use relevant features only (need to match what model was trained on)
    try:
        model_features = model.get_booster().feature_names
        X_test = test_df_features[model_features]
        y_test = test_df_features['water_level']
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate R² and RMSE
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\nModel Performance on Test Set (Last 20% of data):")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
    except Exception as e:
        print(f"\nError calculating model metrics: {e}")
except Exception as e:
    print(f"\nCouldn't load model for validation: {e}")
