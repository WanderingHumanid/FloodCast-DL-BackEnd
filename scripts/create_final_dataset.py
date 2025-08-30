# create_final_dataset.py
# This script combines the time-series data (weather, tides, flood events)
# with the static geographical data for each ward to create the final
# dataset for training the spatio-temporal model.

import pandas as pd

def combine_datasets(
    time_series_path="labeled_flood_data.csv",
    ward_features_path="ward_features.csv",
    output_path="spatio_temporal_flood_data.csv"
):
    """
    Performs a cross-join between the time-series data and the ward features
    to create a comprehensive dataset for spatio-temporal modeling.
    """
    print("Loading datasets...")
    try:
        time_series_df = pd.read_csv(time_series_path, index_col=0, parse_dates=True)
        ward_features_df = pd.read_csv(ward_features_path)
    except FileNotFoundError as e:
        print(f"❌ Error: A required file was not found: {e}")
        print("Please ensure both 'labeled_flood_data.csv' and 'ward_features.csv' exist.")
        return

    print("✅ Datasets loaded successfully.")
    
    # --- Perform a Cross Join ---
    # This creates a new row for every combination of a timestamp and a ward.
    # We add a temporary key to both dataframes to enable the merge.
    time_series_df['key'] = 1
    ward_features_df['key'] = 1

    print("Combining time-series data with ward features...")
    final_df = pd.merge(time_series_df, ward_features_df, on='key').drop('key', axis=1)

    # Set a multi-index of the timestamp and ward name for clarity
    final_df = final_df.set_index([final_df.index, 'ward_name'])
    
    # --- Save the Final Dataset ---
    final_df.to_csv(output_path)
    
    print(f"\nOriginal time-series rows: {len(time_series_df)}")
    print(f"Number of wards: {len(ward_features_df)}")
    print(f"Total rows in final dataset: {len(final_df)}")
    print(f"\n✅ Successfully created the final spatio-temporal dataset and saved it as '{output_path}'")

# --- Main execution block ---
if __name__ == "__main__":
    combine_datasets()
