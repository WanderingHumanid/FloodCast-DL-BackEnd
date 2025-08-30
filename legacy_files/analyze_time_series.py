import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os

# Create directory for plots
plots_dir = 'data_analysis_plots'
os.makedirs(plots_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv('data/merged_flood_moon_data.csv', index_col=0, parse_dates=True)

# 1. Check for seasonal patterns
print("Analyzing seasonal patterns...")
monthly_avg = df.groupby(df.index.month).mean()
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
monthly_avg['rain'].plot(kind='bar')
plt.title('Average Rainfall by Month')
plt.ylabel('Rainfall (mm)')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.subplot(2, 1, 2)
monthly_avg['water_level'].plot(kind='bar')
plt.title('Average Water Level by Month')
plt.ylabel('Water Level (cm)')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.savefig(f'{plots_dir}/seasonal_patterns.png')

# 2. Check for autocorrelation in water level
print("Analyzing autocorrelation...")
acf_lags = 24  # Look at autocorrelation over 24 hours
acf_values = []
for lag in range(1, acf_lags + 1):
    acf_values.append(df['water_level'].autocorr(lag=lag))

plt.figure(figsize=(12, 6))
plt.bar(range(1, acf_lags + 1), acf_values)
plt.title('Autocorrelation of Water Level')
plt.xlabel('Lag (hours)')
plt.ylabel('Autocorrelation')
plt.axhline(y=0, color='r', linestyle='-')
plt.savefig(f'{plots_dir}/autocorrelation.png')

# 3. Check if rain leads to water level changes with lag
print("Analyzing rain-water level relationship...")
max_lag = 12  # Look at up to 12 hours of lag
cross_corr = []
for lag in range(max_lag + 1):
    shifted_rain = df['rain'].shift(lag)
    valid_data = ~shifted_rain.isna()
    corr = df.loc[valid_data, 'water_level'].corr(shifted_rain[valid_data])
    cross_corr.append(corr)

plt.figure(figsize=(12, 6))
plt.bar(range(max_lag + 1), cross_corr)
plt.title('Cross-Correlation: Water Level vs. Rainfall (with lag)')
plt.xlabel('Rainfall Lag (hours)')
plt.ylabel('Correlation')
plt.axhline(y=0, color='r', linestyle='-')
plt.savefig(f'{plots_dir}/rain_water_cross_correlation.png')

# 4. Plot sample time periods to inspect data
print("Generating sample time series plots...")
# Get a rainy week
rain_sum = df['rain'].rolling(window=7*24).sum()
rainiest_week_start = df.index[rain_sum.argmax() - 7*24]
rainiest_week = df.loc[rainiest_week_start:rainiest_week_start + timedelta(days=7)]

plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
rainiest_week['rain'].plot()
plt.title(f'Rainfall during Rainiest Week (starting {rainiest_week_start.date()})')
plt.ylabel('Rainfall (mm)')

plt.subplot(2, 1, 2)
rainiest_week['water_level'].plot()
plt.title('Water Level during Same Period')
plt.ylabel('Water Level (cm)')
plt.tight_layout()
plt.savefig(f'{plots_dir}/rainiest_week.png')

# 5. Check for anomalies in the relationship
print("Checking for anomalies...")
# Calculate water level change
df['water_level_change'] = df['water_level'].diff()

# Look for significant water level increases with no rain
threshold = df['water_level_change'].quantile(0.95)  # 95th percentile of water level increases
anomalies = df[(df['water_level_change'] > threshold) & (df['rain'] < 0.1)]
print(f"Found {len(anomalies)} anomalies (large water level increases with minimal rain)")
if len(anomalies) > 0:
    print("Sample anomalies:")
    print(anomalies.head())

    # Plot a sample anomaly
    if len(anomalies) > 0:
        anomaly_time = anomalies.index[0]
        start_time = anomaly_time - timedelta(hours=24)
        end_time = anomaly_time + timedelta(hours=24)
        anomaly_period = df.loc[start_time:end_time]
        
        plt.figure(figsize=(14, 8))
        plt.subplot(2, 1, 1)
        anomaly_period['rain'].plot()
        plt.axvline(x=anomaly_time, color='r', linestyle='--', label='Anomaly')
        plt.title(f'Rainfall Around Anomaly ({anomaly_time})')
        plt.ylabel('Rainfall (mm)')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        anomaly_period['water_level'].plot()
        plt.axvline(x=anomaly_time, color='r', linestyle='--', label='Anomaly')
        plt.title('Water Level Around Anomaly')
        plt.ylabel('Water Level (cm)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/anomaly_example.png')

# 6. Data continuity check
print("Checking data continuity...")
# Check if there are any gaps larger than 1 hour
df_sorted = df.sort_index()
time_diffs = df_sorted.index.to_series().diff().dt.total_seconds() / 3600
large_gaps = time_diffs[time_diffs > 1]
if len(large_gaps) > 0:
    print(f"Found {len(large_gaps)} gaps in the data larger than 1 hour")
    print("Sample gaps:")
    print(large_gaps.head())
else:
    print("No gaps found in the data")

# 7. Data quality summary
print("\nData Quality Summary:")
print(f"Time period: {df.index.min()} to {df.index.max()}")
print(f"Total hours: {len(df)}")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Hourly data continuity: {'Perfect' if len(large_gaps) == 0 else f'{len(large_gaps)} gaps'}")
print(f"R² value from previous analysis: 0.8922")

print(f"\nAll plots saved to {plots_dir}/ directory")
print("Analysis complete!")
