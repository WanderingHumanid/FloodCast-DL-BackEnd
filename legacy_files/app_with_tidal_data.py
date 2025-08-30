"""
Modified Flask Backend to include tidal data in flood predictions
"""

import flask
from flask_cors import CORS
import pandas as pd
import geopandas as gpd
import joblib
from datetime import datetime, timedelta
import numpy as np
import shap
import rasterio
from rasterio.mask import mask
from utils.validation_metrics import ValidationMetrics
import logging
import os

# --- Configure Logging ---
logging.basicConfig(
    filename="logs/app_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Initialize Flask App ---
app = flask.Flask(__name__)
CORS(app) 

# --- Load all necessary files once on startup ---
try:
    print("Loading models and data files...")
    # Check if the tidal model exists, otherwise use the original model
    if os.path.exists("models/floodsense_xgb_with_tidal_data.pkl"):
        REGRESSION_MODEL = joblib.load("models/floodsense_xgb_with_tidal_data.pkl")
        print("Using enhanced model with tidal data")
        USE_TIDAL_DATA = True
    else:
        REGRESSION_MODEL = joblib.load("models/floodsense_xgb_model_tuned.pkl")
        print("Using original model without tidal data")
        USE_TIDAL_DATA = False
    
    CLASSIFIER_MODEL = joblib.load("models/floodsense_spatio_temporal_classifier.pkl")
    
    # Load appropriate dataset based on which model we're using
    if USE_TIDAL_DATA:
        TIME_SERIES_DF = pd.read_csv("data/merged_flood_moon_tide_data.csv", index_col=0, parse_dates=True)
    else:
        TIME_SERIES_DF = pd.read_csv("data/merged_flood_moon_data.csv", index_col=0, parse_dates=True)
    
    TIME_SERIES_DF = TIME_SERIES_DF[TIME_SERIES_DF.index.year < 2025]
    WARD_FEATURES_DF = pd.read_csv("data/ward_features.csv")
    WARDS_GEO = gpd.read_file("data/kochi_wards.kml", driver='KML')

    # Load tidal data if available (even if not using tidal model yet)
    try:
        TIDAL_DF = pd.read_csv("data/kochi_tidal_data_2023_2025.csv", index_col=0, parse_dates=True)
        print("Tidal data loaded successfully")
        TIDAL_DATA_AVAILABLE = True
    except:
        print("Tidal data not available")
        TIDAL_DATA_AVAILABLE = False

    # Ensure ward name columns are all strings
    WARD_FEATURES_DF['ward_name'] = WARD_FEATURES_DF['ward_name'].astype(str)
    WARDS_GEO['Name'] = WARDS_GEO['Name'].astype(str)

    print("✅ Files loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ CRITICAL ERROR: Could not load a required file: {e}")
    exit()

# --- Feature Preparation Function ---
def prepare_time_features(df, include_tidal=USE_TIDAL_DATA):
    df.index = pd.to_datetime(df.index)
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    df_copy['dayofweek'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    df_copy['dayofyear'] = df_copy.index.dayofyear
    
    # Define base features based on available data
    base_features = ['water_level', 'wind_speed', 'temperature', 'dew_point', 
                    'sea_level_pressure', 'rain', 'moon_illumination_fraction']
    
    # Add tide_level to base features if we're using the tidal model
    if include_tidal and 'tide_level' in df_copy.columns:
        base_features.append('tide_level')
    
    lags = [1, 2, 3, 6]
    for lag in lags:
        for feature in base_features:
            df_copy[f"{feature}_lag{lag}"] = df_copy[feature].shift(lag)
    
    if 'moon_phase' in df_copy.columns:
        df_copy = pd.get_dummies(df_copy, columns=['moon_phase'], prefix='phase', drop_first=True)
    # Return without dropna() for forecasting loop
    return df_copy

# --- Water Level Prediction Function (using Regressor) ---
def predict_water_levels_24_hours(model, initial_df):
    model_features = model.get_booster().feature_names
    working_df = initial_df.copy()
    forecast_rows = []

    # Get future tidal data if available
    future_tidal_data = None
    if USE_TIDAL_DATA and TIDAL_DATA_AVAILABLE:
        # Get the last timestamp in our working df
        last_timestamp = working_df.index[-1]
        # Get the next 24 hours of tidal data
        future_tidal_data = TIDAL_DF.loc[last_timestamp:last_timestamp + timedelta(hours=24)]

    for i in range(24):
        last_known_row = working_df.dropna().iloc[[-1]]
        
        # If we're using tidal data and it's available, add the next hour's tidal data
        if USE_TIDAL_DATA and TIDAL_DATA_AVAILABLE and future_tidal_data is not None:
            next_timestamp = last_known_row.index[0] + timedelta(hours=1)
            if next_timestamp in future_tidal_data.index:
                last_known_row['tide_level'] = future_tidal_data.loc[next_timestamp, 'tide_level']
        
        prediction_input_untyped = last_known_row.reindex(columns=model_features, fill_value=0)
        prediction_input = prediction_input_untyped.apply(pd.to_numeric, errors='coerce').fillna(0)

        predicted_level = model.predict(prediction_input)[0]
        
        next_timestamp = last_known_row.index[0] + timedelta(hours=1)
        new_row = last_known_row.copy()
        new_row.index = [next_timestamp]
        new_row['water_level'] = predicted_level
        
        working_df = pd.concat([working_df, new_row])
        working_df = prepare_time_features(working_df, include_tidal=USE_TIDAL_DATA)
        
        forecast_rows.append(working_df.iloc[[-1]])
        
    return pd.concat(forecast_rows)

# --- Helper function for physical inundation ---
def analyze_physical_inundation(water_level_cm, dem_path, wards_geo):
    with rasterio.open(dem_path) as dem:
        wards_proj = wards_geo.to_crs(dem.crs)
        inundation_percentages = []
        water_level_m = water_level_cm / 100.0

        for _, ward in wards_proj.iterrows():
            try:
                ward_geom = [ward.geometry]
                ward_elevation_data, _ = mask(dem, ward_geom, crop=True)
                valid_pixels = ward_elevation_data[ward_elevation_data > -1000]
                if valid_pixels.size == 0:
                    inundation_percentages.append(0)
                    continue
                flooded_pixels = valid_pixels[valid_pixels < water_level_m]
                percentage = (flooded_pixels.size / valid_pixels.size) * 100
                inundation_percentages.append(percentage)
            except Exception:
                inundation_percentages.append(0)
        
        wards_geo['inundation_percent'] = inundation_percentages
    return wards_geo

# --- Helper function for time labels ---
def get_time_block(hour_index):
    """Convert hour index (0-23) to a time block label for better readability"""
    if hour_index < 6:
        return "Early Morning"
    elif hour_index < 12:
        return "Morning"
    elif hour_index < 18:
        return "Afternoon"
    else:
        return "Evening"

# --- Helper function for demo synthetic data ---
def generate_synthetic_probabilities(num_wards=75):
    """Generate synthetic probabilities for demonstration purposes"""
    # Create a realistic pattern with morning and evening peaks
    np.random.seed(42)  # For reproducibility
    
    # Base pattern with typical flood risk patterns (higher in morning/evening)
    hour = datetime.now().hour
    base_pattern = 0.1  # Base risk
    
    # Time-based adjustments (morning and evening peaks)
    if 5 <= hour < 9:  # Morning peak
        base_pattern = 0.2
    elif 17 <= hour < 21:  # Evening peak
        base_pattern = 0.25
    
    # 5% high risk, 15% medium risk, 80% low risk for easier visualization
    probabilities = np.zeros(num_wards)
    
    # High risk wards (5%)
    high_risk_count = int(num_wards * 0.05)
    high_risk_indices = np.random.choice(num_wards, high_risk_count, replace=False)
    probabilities[high_risk_indices] = np.random.uniform(0.6, 0.95, high_risk_count)
    
    # Medium risk wards (15%)
    remaining_indices = np.setdiff1d(np.arange(num_wards), high_risk_indices)
    medium_risk_count = int(num_wards * 0.15)
    medium_risk_indices = np.random.choice(remaining_indices, medium_risk_count, replace=False)
    probabilities[medium_risk_indices] = np.random.uniform(0.3, 0.6, medium_risk_count)
    
    # Low risk wards (80%) - already initialized to 0
    low_risk_indices = np.setdiff1d(remaining_indices, medium_risk_indices)
    probabilities[low_risk_indices] = np.random.uniform(0.01, 0.3, len(low_risk_indices))
    
    # Apply random variations around the base pattern
    probabilities = np.clip(probabilities * (base_pattern + np.random.uniform(-0.05, 0.05, num_wards)), 0, 1)
    
    return probabilities

# --- Flask Routes ---

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """Generate a 24-hour water level forecast and flood risk for all wards"""
    try:
        # Get last 48 hours of data for forecasting
        end_time = TIME_SERIES_DF.index.max()
        start_time = end_time - timedelta(hours=48)
        recent_data = TIME_SERIES_DF.loc[start_time:end_time].copy()
        
        # Prepare features for the model
        recent_data_features = prepare_time_features(recent_data)
        
        # Generate water level forecast for next 24 hours
        water_level_forecast = predict_water_levels_24_hours(REGRESSION_MODEL, recent_data_features)
        
        # Get the latest forecasted water level (the 24th hour)
        latest_water_level = water_level_forecast['water_level'].iloc[-1]
        
        # Log the forecasted water level for debugging
        logging.debug(f"Forecasted water level for 24hr mark: {latest_water_level}")
        
        # For demo purposes: Use synthetic probabilities instead of the real classifier
        # In production, you'd use CLASSIFIER_MODEL to predict ward-specific probabilities
        ward_probabilities = generate_synthetic_probabilities(len(WARDS_GEO))
        
        # Create ward-level risk data
        ward_risk_data = []
        for i, (_, ward) in enumerate(WARDS_GEO.iterrows()):
            flood_prob = ward_probabilities[i]
            
            # Determine risk level for color coding (red, orange, green)
            if flood_prob >= 0.6:  # High risk
                risk_level = "high"
                color = "#FF0000"  # Red
            elif flood_prob >= 0.3:  # Medium risk
                risk_level = "medium"
                color = "#FFA500"  # Orange
            else:  # Low risk
                risk_level = "low"
                color = "#00FF00"  # Green
                
            ward_risk_data.append({
                "name": ward['Name'],
                "floodProbability": float(flood_prob),
                "riskLevel": risk_level,
                "color": color
            })
        
        # Create hourly forecast data
        hourly_forecast = []
        for i, row in enumerate(water_level_forecast.itertuples()):
            forecast_time = row.Index
            forecast_hour = forecast_time.hour
            
            hourly_forecast.append({
                "hour": forecast_hour,
                "timeLabel": get_time_block(forecast_hour),
                "timestamp": forecast_time.strftime("%Y-%m-%d %H:%M:%S"),
                "waterLevel": float(row.water_level),
                "rainfall": float(row.rain) if hasattr(row, 'rain') else 0.0,
                "tideLevel": float(row.tide_level) if hasattr(row, 'tide_level') and USE_TIDAL_DATA else 0.0
            })
            
        # Construct the response
        response = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "currentWaterLevel": float(recent_data['water_level'].iloc[-1]),
            "forecastWaterLevel": float(latest_water_level),
            "wards": ward_risk_data,
            "hourlyForecast": hourly_forecast,
            "modelUsed": "XGBoost with Tidal Data" if USE_TIDAL_DATA else "XGBoost without Tidal Data"
        }
        
        # Add info about whether we're using tidal data
        response["usingTidalData"] = USE_TIDAL_DATA
        
        return flask.jsonify(response)
    
    except Exception as e:
        logging.error(f"Error in forecast API: {str(e)}", exc_info=True)
        return flask.jsonify({"error": str(e)}), 500

@app.route('/api/validation', methods=['GET'])
def get_validation_metrics():
    """Return model validation metrics for the dashboard"""
    try:
        # Use the validation metrics utility
        validator = ValidationMetrics()
        metrics = validator.get_all_metrics()
        
        # Add information about which model is being used
        metrics["modelInfo"] = {
            "usingTidalData": USE_TIDAL_DATA,
            "modelName": "XGBoost with Tidal Data" if USE_TIDAL_DATA else "XGBoost without Tidal Data",
            "tidalDataAvailable": TIDAL_DATA_AVAILABLE
        }
        
        return flask.jsonify(metrics)
    
    except Exception as e:
        logging.error(f"Error in validation API: {str(e)}", exc_info=True)
        return flask.jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Simple status endpoint to check if the API is running"""
    return flask.jsonify({
        "status": "ok",
        "version": "1.1.0",
        "usingTidalData": USE_TIDAL_DATA,
        "tidalDataAvailable": TIDAL_DATA_AVAILABLE
    })

# --- Start Flask App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
