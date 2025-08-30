# label_data_with_proxies.py
# This script loads the merged dataset, defines flood events based on proxy
# conditions (high rain and high water level), and saves the new labeled dataset.

import pandas as pd

def label_flood_events(input_filepath="merged_flood_moon_data.csv", output_filepath="labeled_flood_data.csv"):
    """
    Creates a labeled dataset for flood classification by identifying hours
    with exceptionally high rainfall and water levels.
    """
    print(f"Loading data from '{input_filepath}'...")
    try:
        df = pd.read_csv(input_filepath, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"❌ Error: The file '{input_filepath}' was not found. Please ensure it is in the same directory.")
        return

    # --- Step 1: Define the Thresholds for a High-Risk Event ---
    # We define thresholds for different levels of rainfall intensity
    extreme_rain_threshold = df[df['rain'] > 0]['rain'].quantile(0.95)  # 95th percentile
    high_rain_threshold = df[df['rain'] > 0]['rain'].quantile(0.90)     # 90th percentile
    
    # Water level threshold at 95th percentile
    water_level_threshold = df['water_level'].quantile(0.95)

    print("\n--- Proxy Thresholds ---")
    print(f"Extreme Rainfall Threshold (95th percentile): {extreme_rain_threshold:.4f} mm/hr")
    print(f"High Rainfall Threshold (90th percentile): {high_rain_threshold:.4f} mm/hr")
    print(f"High Water Level Threshold (95th percentile): {water_level_threshold:.2f} cm")
    print("------------------------\n")

    # --- Step 2: Create the 'flood_event' Label ---
    # Calculate rolling sum of rain for the last 3 hours
    df['rolling_rain_3hr'] = df['rain'].rolling(window=3, min_periods=1).sum()
    
    # Define conditions for flood events:
    # 1. Extreme rain (95th percentile) AND elevated water level, OR
    # 2. Sustained high rain (90th percentile for 3 hours) AND elevated water level
    extreme_conditions = (df['rain'] >= extreme_rain_threshold) & (df['water_level'] >= water_level_threshold)
    sustained_conditions = (df['rolling_rain_3hr'] >= high_rain_threshold * 3) & (df['water_level'] >= water_level_threshold)
    
    df['flood_event'] = 0
    df.loc[extreme_conditions | sustained_conditions, 'flood_event'] = 1

    # --- Step 3: Analyze and Save the Labeled Data ---
    num_flood_events = df['flood_event'].sum()
    total_hours = len(df)
    event_percentage = (num_flood_events / total_hours) * 100

    print(f"Found {num_flood_events} hours matching the flood event criteria.")
    print(f"This represents {event_percentage:.2f}% of the total dataset.")

    # Save the new DataFrame to a CSV file
    df.to_csv(output_filepath)
    print(f"\n✅ Successfully created labeled dataset and saved it as '{output_filepath}'")

# --- Main execution block ---
if __name__ == "__main__":
    label_flood_events()
