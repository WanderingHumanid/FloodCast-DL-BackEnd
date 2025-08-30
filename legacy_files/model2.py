# flood_inundation_mapping.py
# This script integrates the AI model's prediction with geospatial data
# to create a ward-level flood risk map for Kochi for the next 24 hours.

import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import folium
import joblib
from datetime import datetime, timedelta

def prepare_features(df):
    """Prepares time-based and lag features for the dataframe."""
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)
    
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()

    # Add time-based features
    df_copy['hour'] = df_copy.index.hour
    df_copy['dayofweek'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    df_copy['dayofyear'] = df_copy.index.dayofyear

    base_features = [
        'water_level', 'wind_speed', 'temperature', 'dew_point',
        'sea_level_pressure', 'rain', 'moon_illumination_fraction'
    ]
    lags = [1, 2, 3, 6]
    for lag in lags:
        for feature in base_features:
            df_copy[f"{feature}_lag{lag}"] = df_copy[feature].shift(lag)
    
    # One-hot encode moon phases if they exist and are not already encoded
    if 'moon_phase' in df_copy.columns:
        df_copy = pd.get_dummies(df_copy, columns=['moon_phase'], prefix='phase', drop_first=True)

    return df_copy.dropna()

def predict_next_24_hours(model, latest_data_df):
    """
    Generates water level predictions for the next 24 hours iteratively.

    Args:
        model: The trained XGBoost model.
        latest_data_df (pd.DataFrame): A dataframe containing the most recent
                                       data, with features already prepared.

    Returns:
        list: A list of predicted water levels for the next 24 hours.
    """
    print("Generating 24-hour forecast...")
    predictions = []
    
    model_features = model.get_booster().feature_names
    
    # Start with the most recent complete row of data
    last_known_row = latest_data_df.iloc[[-1]].copy()

    for hour in range(1, 25):
        # Ensure the input has all the necessary columns in the correct order
        prediction_input = last_known_row.reindex(columns=model_features, fill_value=0)

        # Make a prediction for the next hour
        predicted_level = model.predict(prediction_input)[0]
        predictions.append(predicted_level)

        # --- Create the input for the *next* hour's prediction ---
        # This is the crucial change: we manually update the next row's features
        next_row = last_known_row.copy()
        
        # Update timestamp
        next_timestamp = next_row.index[0] + timedelta(hours=1)
        next_row.index = [next_timestamp]
        
        # Update time-based features
        next_row['hour'] = next_timestamp.hour
        next_row['dayofweek'] = next_timestamp.dayofweek
        next_row['month'] = next_timestamp.month
        next_row['dayofyear'] = next_timestamp.dayofyear

        # Update lag features by shifting them
        # Lag 6 becomes Lag 3 from the previous step
        for feature in ['water_level', 'wind_speed', 'temperature', 'dew_point', 'sea_level_pressure', 'rain', 'moon_illumination_fraction']:
             next_row[f'{feature}_lag6'] = last_known_row[f'{feature}_lag3'].values[0]
             next_row[f'{feature}_lag3'] = last_known_row[f'{feature}_lag2'].values[0]
             next_row[f'{feature}_lag2'] = last_known_row[f'{feature}_lag1'].values[0]
             next_row[f'{feature}_lag1'] = last_known_row[feature].values[0]

        # The most important update: the new 'water_level' is our prediction
        next_row['water_level'] = predicted_level
        
        # The next iteration will use this newly created row
        last_known_row = next_row

        print(f"  - Hour {hour:02d}: Predicted Water Level = {float(predicted_level):.2f} cm")
        
    return predictions

def analyze_flood_risk(predicted_water_level_cm, dem_path, wards_path):
    """
    Calculates the percentage of area flooded in each ward based on a
    predicted water level and a Digital Elevation Model (DEM).
    """
    print("\nStarting flood risk analysis based on peak predicted water level...")
    predicted_water_level_m = predicted_water_level_cm / 100.0
    print(f"Analyzing for peak water level: {predicted_water_level_cm:.2f} cm ({predicted_water_level_m:.2f} m)")

    try:
        wards = gpd.read_file(wards_path, driver='KML')
        dem = rasterio.open(dem_path)
    except Exception as e:
        print(f"❌ Error loading geospatial files: {e}")
        return None

    wards = wards.to_crs(dem.crs)
    print("✅ Geospatial data loaded successfully.")

    flood_risk_percentages = []
    for index, ward in wards.iterrows():
        try:
            ward_geom = [ward.geometry]
            ward_elevation_data, _ = mask(dem, ward_geom, crop=True)
            ward_elevation_data = np.squeeze(ward_elevation_data)
            valid_pixels = ward_elevation_data[ward_elevation_data > -1000]
            
            if valid_pixels.size == 0:
                flood_risk_percentages.append(0)
                continue

            flooded_pixels = valid_pixels[valid_pixels < predicted_water_level_m]
            percentage_flooded = (flooded_pixels.size / valid_pixels.size) * 100
            flood_risk_percentages.append(percentage_flooded)
        except Exception:
            flood_risk_percentages.append(0)

    wards['flood_risk_percent'] = flood_risk_percentages
    print("✅ Flood risk analysis complete for all wards.")
    return wards

def get_color(risk_percentage):
    """Returns a color based on the flood risk percentage."""
    if risk_percentage > 75:
        return '#000000'  # Black for 'Already Flooded'
    elif risk_percentage > 50:
        return '#FF0000'  # Red for 'High Risk'
    elif risk_percentage > 25:
        return '#FFFF00'  # Yellow for 'Moderate Risk'
    else:
        return '#008000'  # Green for 'Low Risk'

def create_flood_map(wards_with_risk, forecast_start_time, output_filename="kochi_flood_risk_map.html"):
    """
    Creates an interactive Folium map visualizing the flood risk for each ward.
    """
    if wards_with_risk is None:
        print("Cannot create map, analysis data is missing.")
        return
        
    print("Generating interactive flood risk map...")
    map_center = wards_with_risk.unary_union.centroid
    m = folium.Map(location=[map_center.y, map_center.x], zoom_start=12, tiles="CartoDB positron")

    # Style function to color wards based on risk
    style_function = lambda feature: {
        'fillColor': get_color(feature['properties']['flood_risk_percent']),
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.7
    }

    # Add the GeoJson layer with custom styling and tooltips
    folium.GeoJson(
        wards_with_risk,
        style_function=style_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=['Name', 'flood_risk_percent'],
            aliases=['Ward:', 'Max Risk in Next 24h:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),
            localize=True
        )
    ).add_to(m)

    # Add a title with the forecast period
    forecast_end_time = forecast_start_time + timedelta(hours=24)
    title_html = f'''
        <h3 align="center" style="font-size:16px"><b>Flood Risk Forecast for Kochi</b></h3>
        <p align="center" style="font-size:12px">
        Showing peak risk from {forecast_start_time.strftime('%Y-%m-%d %H:%M')} to {forecast_end_time.strftime('%Y-%m-%d %H:%M')}
        </p>
        '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Save the map
    m.save(output_filename)
    print(f"✅ Map saved successfully as '{output_filename}'")

# --- Main execution block ---
if __name__ == "__main__":
    # --- CONFIGURATION ---
    DEM_FILE = "srtm_data.tif"
    WARDS_FILE = "kochi_wards.kml"
    MODEL_FILE = "floodsense_xgb_model_tuned.pkl"
    DATA_FILE = "merged_flood_moon_data.csv"

    # --- LOAD DATA AND MODEL ---
    try:
        model = joblib.load(MODEL_FILE)
        full_df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
        full_df = full_df[full_df.index.year < 2025] # Filter out incomplete data
    except FileNotFoundError as e:
        print(f"❌ Error loading a required file: {e}")
        exit()

    # --- PREPARE FEATURES FOR THE ENTIRE DATASET ---
    print("Preparing features for the full dataset...")
    full_df_with_features = prepare_features(full_df)
    print("✅ Features prepared.")

    # --- GENERATE 24-HOUR FORECAST ---
    # We need a recent, complete row of data to start the forecast
    latest_data = full_df_with_features.tail(1)
    hourly_predictions = predict_next_24_hours(model, latest_data)
    
    # Find the highest predicted water level in the next 24 hours
    peak_water_level = max(hourly_predictions)
    
    # --- RUN ANALYSIS AND CREATE MAP ---
    wards_with_risk_data = analyze_flood_risk(peak_water_level, DEM_FILE, WARDS_FILE)
    
    # The forecast starts from the last known data point in our dataset
    forecast_start = full_df.index[-1] + timedelta(hours=1)
    create_flood_map(wards_with_risk_data, forecast_start)
