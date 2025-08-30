import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import pearsonr
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Create output directory for plots
plots_dir = 'data_analysis_plots'
os.makedirs(plots_dir, exist_ok=True)

print("Loading the merged dataset with tidal data...")
df = pd.read_csv('data/merged_flood_moon_tide_data.csv', index_col=0, parse_dates=True)

# Basic statistics
print(f"Dataset time range: {df.index.min()} to {df.index.max()}")
print(f"Total records: {len(df)}")

# Create a monthly view of tidal patterns
print("\nAnalyzing monthly tidal patterns...")
monthly_tides = df.resample('M').agg({
    'tide_level': ['mean', 'min', 'max', 'std'],
    'water_level': ['mean', 'min', 'max', 'std']
})

# Flatten the MultiIndex columns
monthly_tides.columns = ['_'.join(col).strip() for col in monthly_tides.columns.values]

print(monthly_tides.head())

# Plot monthly tidal statistics
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(monthly_tides.index, monthly_tides['tide_level_mean'], 'b-', label='Mean Tide Level')
plt.fill_between(monthly_tides.index, 
                monthly_tides['tide_level_min'], 
                monthly_tides['tide_level_max'], 
                alpha=0.2)
plt.title('Monthly Tide Level Statistics')
plt.ylabel('Tide Level')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(monthly_tides.index, monthly_tides['water_level_mean'], 'r-', label='Mean Water Level')
plt.fill_between(monthly_tides.index, 
                monthly_tides['water_level_min'], 
                monthly_tides['water_level_max'], 
                alpha=0.2)
plt.title('Monthly Water Level Statistics')
plt.ylabel('Water Level')
plt.legend()

plt.tight_layout()
plt.savefig(f'{plots_dir}/monthly_tide_water_patterns.png')

# Analyze tidal and water level patterns in rainy vs. dry periods
print("\nAnalyzing tidal influence during rainy vs. dry periods...")
# Define rainy periods (above 75th percentile of rainfall)
rain_threshold = df['rain'].quantile(0.75)
df['is_rainy'] = df['rain'] > rain_threshold

# Group by rainy/dry and calculate statistics
rainy_stats = df.groupby('is_rainy').agg({
    'tide_level': ['mean', 'std', 'min', 'max'],
    'water_level': ['mean', 'std', 'min', 'max']
})

# Flatten the MultiIndex columns
rainy_stats.columns = ['_'.join(col).strip() for col in rainy_stats.columns.values]
print(rainy_stats)

# Calculate the correlation between tide and water level during rainy vs. dry periods
rainy_corr, _ = pearsonr(df[df['is_rainy']]['tide_level'], df[df['is_rainy']]['water_level'])
dry_corr, _ = pearsonr(df[~df['is_rainy']]['tide_level'], df[~df['is_rainy']]['water_level'])

print(f"Tide-Water correlation during rainy periods: {rainy_corr:.4f}")
print(f"Tide-Water correlation during dry periods: {dry_corr:.4f}")

# Create a scatter plot with regression lines for rainy vs. dry periods
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df[df['is_rainy']], x='tide_level', y='water_level', 
                alpha=0.3, color='blue', label=f'Rainy (corr={rainy_corr:.3f})')
sns.scatterplot(data=df[~df['is_rainy']], x='tide_level', y='water_level', 
                alpha=0.3, color='red', label=f'Dry (corr={dry_corr:.3f})')

# Add regression lines
sns.regplot(data=df[df['is_rainy']], x='tide_level', y='water_level', 
           scatter=False, color='blue', line_kws={'linewidth': 2})
sns.regplot(data=df[~df['is_rainy']], x='tide_level', y='water_level', 
           scatter=False, color='red', line_kws={'linewidth': 2})

plt.title('Relationship Between Tide Level and Water Level in Rainy vs. Dry Periods')
plt.xlabel('Tide Level')
plt.ylabel('Water Level')
plt.legend()
plt.savefig(f'{plots_dir}/tide_water_rainy_vs_dry.png')

# Create a more detailed analysis of high-tide events
print("\nAnalyzing high-tide events...")
high_tide_threshold = df['tide_level'].quantile(0.90)
df['is_high_tide'] = df['tide_level'] > high_tide_threshold

# Group by high-tide/normal and calculate statistics
high_tide_stats = df.groupby('is_high_tide').agg({
    'water_level': ['mean', 'std', 'min', 'max'],
    'rain': ['mean', 'sum']
})

# Flatten the MultiIndex columns
high_tide_stats.columns = ['_'.join(col).strip() for col in high_tide_stats.columns.values]
print(high_tide_stats)

# Analyze hourly patterns of tides
print("\nAnalyzing hourly tidal patterns...")
hourly_tides = df.groupby(df.index.hour).agg({
    'tide_level': ['mean', 'std'],
    'water_level': ['mean', 'std']
})

# Flatten the MultiIndex columns
hourly_tides.columns = ['_'.join(col).strip() for col in hourly_tides.columns.values]
print(hourly_tides)

# Plot hourly patterns
plt.figure(figsize=(12, 6))
plt.plot(hourly_tides.index, hourly_tides['tide_level_mean'], 'b-', label='Mean Tide Level')
plt.fill_between(hourly_tides.index, 
                hourly_tides['tide_level_mean'] - hourly_tides['tide_level_std'], 
                hourly_tides['tide_level_mean'] + hourly_tides['tide_level_std'], 
                alpha=0.2, color='blue')
plt.plot(hourly_tides.index, hourly_tides['water_level_mean'], 'r-', label='Mean Water Level')
plt.fill_between(hourly_tides.index, 
                hourly_tides['water_level_mean'] - hourly_tides['water_level_std'], 
                hourly_tides['water_level_mean'] + hourly_tides['water_level_std'], 
                alpha=0.2, color='red')
plt.title('Hourly Tide and Water Level Patterns')
plt.xlabel('Hour of Day')
plt.ylabel('Level')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(f'{plots_dir}/hourly_tide_water_patterns.png')

# Find extreme tidal events and analyze their impact
print("\nAnalyzing extreme tidal events...")
extreme_high_tide = df['tide_level'].quantile(0.95)
extreme_dates = df[df['tide_level'] > extreme_high_tide].index

# Select a sample extreme tidal event for detailed analysis
if len(extreme_dates) > 0:
    sample_date = extreme_dates[len(extreme_dates) // 2]
    start_date = sample_date - timedelta(days=2)
    end_date = sample_date + timedelta(days=2)
    
    sample_period = df.loc[start_date:end_date]
    
    # Plot the extreme tidal event
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    ax[0].plot(sample_period.index, sample_period['tide_level'], 'b-')
    ax[0].axhline(y=extreme_high_tide, color='r', linestyle='--', label=f'Extreme Tide Threshold ({extreme_high_tide:.1f})')
    ax[0].set_title('Extreme Tidal Event Analysis')
    ax[0].set_ylabel('Tide Level')
    ax[0].legend()
    
    ax[1].plot(sample_period.index, sample_period['water_level'], 'g-')
    ax[1].set_ylabel('Water Level')
    
    ax[2].plot(sample_period.index, sample_period['rain'], 'r-')
    ax[2].set_ylabel('Rainfall')
    ax[2].set_xlabel('Date')
    
    # Format x-axis to show dates clearly
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H'))
    ax[2].xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/extreme_tidal_event.png')
    
    print(f"Sample extreme tidal event analyzed: {sample_date}")
else:
    print("No extreme tidal events found.")

print("\nTidal data analysis complete.")
