import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

# Create output directory for plots
plots_dir = 'data_analysis_plots'
os.makedirs(plots_dir, exist_ok=True)

print("Loading tidal dataset...")
# Load the tidal data (format appears to be: year, month, day, hour, tide_level)
tidal_df = pd.read_csv('data/h174.csv', header=None, 
                        names=['year', 'month', 'day', 'hour', 'tide_level'])

# Convert to datetime and set as index
tidal_df['datetime'] = pd.to_datetime(tidal_df[['year', 'month', 'day', 'hour']])
tidal_df = tidal_df.set_index('datetime')

# Print basic information about the dataset
print(f"Tidal dataset time range: {tidal_df.index.min()} to {tidal_df.index.max()}")
print(f"Total records: {len(tidal_df)}")
print(f"Missing values: {tidal_df['tide_level'].isna().sum()}")

# Extract only the data we need (2023 to 2025)
start_date = '2023-01-01'
end_date = '2025-08-23 18:00:00'  # Matching the end of our existing dataset
filtered_tidal_df = tidal_df.loc[start_date:end_date].copy()
print(f"\nExtracted data from {start_date} to {end_date}")
print(f"Extracted records: {len(filtered_tidal_df)}")

# Save the extracted data
filtered_tidal_df[['tide_level']].to_csv('data/kochi_tidal_data_2023_2025.csv')
print("Saved extracted tidal data to 'data/kochi_tidal_data_2023_2025.csv'")

# Load the merged flood moon data to check correlation
print("\nLoading merged flood moon data...")
merged_df = pd.read_csv('data/merged_flood_moon_data.csv', index_col=0, parse_dates=True)
print(f"Merged dataset time range: {merged_df.index.min()} to {merged_df.index.max()}")

# Merge the tidal data with the existing dataset for correlation analysis
print("\nMerging datasets for correlation analysis...")
merged_with_tidal = pd.merge(merged_df, 
                            filtered_tidal_df[['tide_level']], 
                            left_index=True, 
                            right_index=True, 
                            how='left')

print(f"Merged records: {len(merged_with_tidal)}")
print(f"Missing tidal values after merge: {merged_with_tidal['tide_level'].isna().sum()}")

# Fill any missing values using forward fill (should be minimal if any)
if merged_with_tidal['tide_level'].isna().sum() > 0:
    merged_with_tidal['tide_level'] = merged_with_tidal['tide_level'].fillna(method='ffill')
    print("Filled missing tidal values using forward fill")

# Calculate correlations with moon illumination and water level
print("\nCalculating correlations:")
moon_corr, moon_p = pearsonr(merged_with_tidal['moon_illumination_fraction'], 
                            merged_with_tidal['tide_level'])
water_corr, water_p = pearsonr(merged_with_tidal['water_level'], 
                             merged_with_tidal['tide_level'])
rain_corr, rain_p = pearsonr(merged_with_tidal['rain'], 
                           merged_with_tidal['tide_level'])

print(f"Correlation between tide level and moon illumination: {moon_corr:.4f} (p={moon_p:.4f})")
print(f"Correlation between tide level and water level: {water_corr:.4f} (p={water_p:.4f})")
print(f"Correlation between tide level and rainfall: {rain_corr:.4f} (p={rain_p:.4f})")

# Plot the correlation between tide level and moon illumination
plt.figure(figsize=(10, 6))
plt.scatter(merged_with_tidal['moon_illumination_fraction'], 
           merged_with_tidal['tide_level'], 
           alpha=0.1)
plt.title(f'Tide Level vs. Moon Illumination (Correlation: {moon_corr:.4f})')
plt.xlabel('Moon Illumination Fraction')
plt.ylabel('Tide Level')
plt.savefig(f'{plots_dir}/tide_moon_correlation.png')

# Plot the correlation between tide level and water level
plt.figure(figsize=(10, 6))
plt.scatter(merged_with_tidal['tide_level'], 
           merged_with_tidal['water_level'], 
           alpha=0.1)
plt.title(f'Water Level vs. Tide Level (Correlation: {water_corr:.4f})')
plt.xlabel('Tide Level')
plt.ylabel('Water Level')
plt.savefig(f'{plots_dir}/tide_water_correlation.png')

# Plot a time series of tide level and water level for a sample period (1 week)
sample_start = '2023-06-01'
sample_end = '2023-06-07'
sample_data = merged_with_tidal.loc[sample_start:sample_end]

plt.figure(figsize=(14, 8))
plt.subplot(3, 1, 1)
plt.plot(sample_data.index, sample_data['tide_level'], 'r-')
plt.title('Tide Level (1 Week Sample)')
plt.ylabel('Tide Level')

plt.subplot(3, 1, 2)
plt.plot(sample_data.index, sample_data['water_level'], 'b-')
plt.title('Water Level (Same Period)')
plt.ylabel('Water Level')

plt.subplot(3, 1, 3)
plt.plot(sample_data.index, sample_data['moon_illumination_fraction'], 'g-')
plt.title('Moon Illumination (Same Period)')
plt.ylabel('Moon Illumination')

plt.tight_layout()
plt.savefig(f'{plots_dir}/tide_water_time_series.png')

# Save the merged dataset with tidal data
merged_with_tidal.to_csv('data/merged_flood_moon_tide_data.csv')
print("\nSaved complete merged dataset to 'data/merged_flood_moon_tide_data.csv'")

# Create a version that includes only the primary variables for easier viewing
primary_vars = ['water_level', 'rain', 'tide_level', 'moon_illumination_fraction']
merged_with_tidal[primary_vars].head(20).to_csv('data/sample_merged_data.csv')
print("Saved sample of merged data to 'data/sample_merged_data.csv'")

print("\nAnalysis complete. Check the data_analysis_plots directory for visualization results.")
