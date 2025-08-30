# create_ward_features.py
# This script analyzes the geospatial data for each ward to create a set of
# unique geographical features that can be used by the AI model.

import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd

def calculate_geospatial_features(dem_path, wards_path, output_path="ward_features.csv"):
    """
    Calculates geographical features for each ward and saves them to a CSV file.

    Features calculated:
    - avg_elevation_m: The average elevation of the ward in meters.
    - min_elevation_m: The lowest point in the ward in meters.
    - low_lying_area_percent: The percentage of the ward's area below 2 meters.
    """
    print("Loading geospatial data...")
    try:
        # Modern geopandas handles the KML driver automatically.
        # We specify the driver here for clarity.
        wards = gpd.read_file(wards_path, driver='KML')
        dem = rasterio.open(dem_path)
    except Exception as e:
        print(f"❌ Error loading geospatial files: {e}")
        return

    # --- CORRECTED SECTION: Handle Coordinate Reference System (CRS) ---
    # KML files are typically in WGS84 (EPSG:4326). We'll set this explicitly.
    if wards.crs is None:
        print("Ward file has no CRS set. Assuming WGS84 (EPSG:4326).")
        wards.set_crs("EPSG:4326", inplace=True)
    
    # Ensure ward data is in the same CRS as the DEM
    wards = wards.to_crs(dem.crs)
    print("✅ Geospatial data loaded and reprojected successfully.")

    # --- NEW: Help user identify the correct column for ward names ---
    print(f"\nColumns found in KML file: {wards.columns.tolist()}")
    print("--> Please check if 'Name' is the correct column for ward names.\n")


    ward_features = []

    print("Calculating features for each ward...")
    # Iterate over each ward to calculate its features
    for index, ward in wards.iterrows():
        # IMPORTANT: You may need to change 'Name' to the correct column name from the list above.
        ward_name = ward.get('Name', f"Ward_{index}")
        try:
            # Clip the DEM raster to the boundary of the current ward
            ward_elevation_data, _ = mask(dem, [ward.geometry], crop=True)
            ward_elevation_data = np.squeeze(ward_elevation_data)
            
            # Filter out 'nodata' values
            valid_pixels = ward_elevation_data[ward_elevation_data > -1000]
            
            if valid_pixels.size == 0:
                print(f"  - ⚠️ Skipping '{ward_name}' (no valid elevation data).")
                continue

            # --- Calculate Features ---
            avg_elevation = float(valid_pixels.mean())
            min_elevation = float(valid_pixels.min())
            
            # Calculate the percentage of the area below 2 meters
            low_lying_pixels = valid_pixels[valid_pixels < 2.0]
            low_lying_percentage = (low_lying_pixels.size / valid_pixels.size) * 100

            ward_features.append({
                'ward_name': ward_name,
                'avg_elevation_m': avg_elevation,
                'min_elevation_m': min_elevation,
                'low_lying_area_percent': low_lying_percentage
            })
            print(f"  - Processed '{ward_name}'")

        except Exception as e:
            print(f"  - ⚠️ Could not process '{ward_name}': {e}")

    # Convert the list of features to a pandas DataFrame
    features_df = pd.DataFrame(ward_features)
    
    # Save the features to a CSV file
    features_df.to_csv(output_path, index=False)
    print(f"\n✅ Successfully calculated and saved features for {len(features_df)} wards to '{output_path}'")

# --- Main execution block ---
if __name__ == "__main__":
    DEM_FILE = "srtm_data.tif"
    WARDS_FILE = "kochi_wards.kml"
    calculate_geospatial_features(DEM_FILE, WARDS_FILE)
