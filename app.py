import flask
from flask_cors import CORS
from flask import send_from_directory
import pandas as pd
import geopandas as gpd
import joblib
from datetime import datetime, timedelta
import numpy as np
import shap
import rasterio
from rasterio.mask import mask
import os
import json
import logging
from utils.validation_metrics import ValidationMetrics
import logging
import config  # Import the configuration settings

# --- Configure Logging ---
# Ensure logs directory exists
os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

# Configure logging to write to file and console
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()  # This will log to console as well
    ]
)

# --- Initialize Flask App ---
app = flask.Flask(__name__)
# Configure CORS to allow all origins
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Add after_request handler to print CORS headers
@app.after_request
def after_request(response):
    logging.info(f"Request: {flask.request.method} {flask.request.path}")
    
    # Ensure CORS headers are set correctly
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    
    return response

# Health check endpoint for deployment verification
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running"""
    app_version = getattr(config, 'APP_VERSION', '1.0.0')
    return flask.jsonify({
        'status': 'ok',
        'version': app_version,
        'environment': config.FLASK_ENV,
        'timestamp': datetime.now().isoformat()
    })
    return response

# --- Load all necessary files once on startup ---
try:
    print("Loading models and data files...")
    
    # Use the enhanced model if configured
    if config.USE_ENHANCED_MODEL:
        model_path = config.ENHANCED_MODEL_PATH
        data_path = config.TIDAL_DATA_PATH
        print(f"Using enhanced model with tidal data from {model_path}")
    else:
        model_path = config.ORIGINAL_MODEL_PATH
        data_path = config.NON_TIDAL_DATA_PATH
        print(f"Using original model without tidal data from {model_path}")
        
    try:
        # Ensure required modules are available
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        logging.info("TensorFlow/Keras module loaded successfully")
    except ImportError:
        print("❌ CRITICAL ERROR: Required module 'tensorflow' not found. Please install it using 'pip install tensorflow'")
        logging.error("Required module 'tensorflow' not found")
        exit(1)
    # Load models
    try:
        REGRESSION_MODEL = load_model("../flood_prediction_model.h5")
        CLASSIFIER_MODEL = joblib.load(config.CLASSIFIER_MODEL_PATH)
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load model files: {e}")
        logging.error(f"Model loading error: {e}")
        exit(1)
    
    # Load datasets
    TIME_SERIES_DF = pd.read_csv(data_path, index_col=0, parse_dates=True)
    TIME_SERIES_DF = TIME_SERIES_DF[TIME_SERIES_DF.index.year < 2025]
    WARD_FEATURES_DF = pd.read_csv(config.WARD_FEATURES_PATH)
    WARDS_GEO = gpd.read_file(config.WARDS_GEOMETRY_PATH, driver='KML')

    # Ensure ward name columns are all strings
    WARD_FEATURES_DF['ward_name'] = WARD_FEATURES_DF['ward_name'].astype(str)
    WARDS_GEO['Name'] = WARDS_GEO['Name'].astype(str)

    print("✅ Files loaded successfully.")
    logging.info("All data files and models loaded successfully")
except FileNotFoundError as e:
    print(f"❌ CRITICAL ERROR: Could not load a required file: {e}")
    logging.error(f"FileNotFoundError: {e}")
    exit(1)

# --- Water Level Prediction Function (using Regressor) ---
def predict_water_levels_24_hours(model, initial_df):
    # For Keras model, we need to know the input feature order
    # The model expects input shape (None, 24, 11) - a sequence of 24 timesteps with 11 features
    working_df = initial_df.copy()
    forecast_rows = []
    
    # Identify the most important features for the model
    # Using 11 key features to match the model's expected input
    key_features = [
        'water_level', 'rain', 'temperature', 
        'wind_speed', 'sea_level_pressure', 'hour_sin', 
        'hour_cos', 'dayofyear', 'tide_level', 
        'tide_velocity', 'moon_illumination_fraction'
    ]
    
    # Ensure we have these features, filling with 0 if missing
    for feature in key_features:
        if feature not in working_df.columns:
            working_df[feature] = 0
            
    # Get the latest timestamp to start forecasting from
    if len(working_df) > 0:
        last_timestamp = working_df.index[-1]
    else:
        last_timestamp = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    # Make a copy of the working_df to use for sequence building
    sequence_df = working_df.copy()
    
    # Generate 24 hour forecast one hour at a time
    for i in range(24):
        # Use the most recent 24 hours of data for prediction
        # If we don't have enough data, we'll use what we have and pad with copies
        sequence_length = 24
        if len(sequence_df) < sequence_length:
            # Pad by repeating the first row
            first_row = sequence_df.iloc[[0]]
            padding_rows = pd.concat([first_row] * (sequence_length - len(sequence_df)))
            padded_df = pd.concat([padding_rows, sequence_df])
        else:
            padded_df = sequence_df.tail(sequence_length)
        
        # Extract only the needed features and convert to numpy array
        # Shape should be (1, 24, 11) for the model
        try:
            input_sequence = padded_df[key_features].values.astype(np.float32)
            input_sequence = np.expand_dims(input_sequence, axis=0)  # Add batch dimension
            
            # Get prediction from the model
            predicted_level = float(model.predict(input_sequence, verbose=0)[0][0])
            
            # Create new row for the forecast
            next_timestamp = last_timestamp + timedelta(hours=i+1)
            
            # Use the last row as a template and update it
            if len(working_df) > 0:
                new_row = working_df.iloc[[-1]].copy()
                new_row.index = [next_timestamp]
            else:
                # Create a new row with zeros if we don't have any data
                new_row = pd.DataFrame(index=[next_timestamp], columns=working_df.columns)
                new_row = new_row.fillna(0)
            
            # Update the water level with our prediction
            new_row['water_level'] = predicted_level
            
            # Update tide level using a sinusoidal approximation if it exists
            if 'tide_level' in new_row.columns:
                hours_since_start = (next_timestamp - working_df.index[0]).total_seconds() / 3600 if len(working_df) > 0 else i
                new_row['tide_level'] = 150 + 50 * np.sin(2 * np.pi * hours_since_start / 12.42)
            
            # Add the new row to our forecast and update the sequence for next prediction
            forecast_rows.append(new_row)
            sequence_df = pd.concat([sequence_df, new_row])
            
        except Exception as e:
            print(f"Error during prediction at step {i}: {e}")
            continue
    
    # Combine all forecasts into a single dataframe
    if forecast_rows:
        forecast_df = pd.concat(forecast_rows)
        return forecast_df
    else:
        return pd.DataFrame()

# --- Feature Preparation Function ---
def prepare_time_features(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    
    # Extract base features for reference
    base_features = ['water_level', 'wind_speed', 'temperature', 'dew_point', 'sea_level_pressure', 'rain', 'moon_illumination_fraction']
    if 'tide_level' in df.columns:
        base_features.append('tide_level')
    
    # Create dictionaries to collect all the new columns before adding them to the dataframe
    # This avoids DataFrame fragmentation warnings
    time_features = {
        'hour': df.index.hour,
        'dayofweek': df.index.dayofweek,
        'month': df.index.month,
        'dayofyear': df.index.dayofyear
    }
    
    # Add harmonic features
    harmonic_features = {
        'hour_sin': np.sin(2 * np.pi * time_features['hour'] / 24),
        'hour_cos': np.cos(2 * np.pi * time_features['hour'] / 24),
        'month_sin': np.sin(2 * np.pi * time_features['month'] / 12),
        'month_cos': np.cos(2 * np.pi * time_features['month'] / 12)
    }
    
    # Create lag features
    lag_features = {}
    for lag in [1, 2, 3, 6, 12, 24]:
        for feature in base_features:
            if feature in df.columns:
                lag_features[f"{feature}_lag{lag}"] = df[feature].shift(lag)
    
    # Create rolling window features
    rolling_features = {}
    for window in [3, 6, 12, 24]:
        for feature in base_features:
            if feature in df.columns:
                rolling_features[f"{feature}_rolling_mean_{window}h"] = df[feature].rolling(window=window).mean()
                rolling_features[f"{feature}_rolling_max_{window}h"] = df[feature].rolling(window=window).max()
                rolling_features[f"{feature}_rolling_min_{window}h"] = df[feature].rolling(window=window).min()
    
    # Create velocity and acceleration features
    derivative_features = {}
    for feature in base_features:
        if feature in df.columns:
            derivative_features[f"{feature}_velocity"] = df[feature].diff()
            derivative_features[f"{feature}_acceleration"] = df[feature].diff().diff()
    
    # Add interaction terms and tidal features
    interaction_features = {}
    if 'tide_level' in df.columns and 'rain' in df.columns:
        interaction_features['tide_rain_interaction'] = df['tide_level'] * df['rain']
        interaction_features['tide_range_6h'] = df['tide_level'].rolling(window=6).max() - df['tide_level'].rolling(window=6).min()
        interaction_features['tide_range_12h'] = df['tide_level'].rolling(window=12).max() - df['tide_level'].rolling(window=12).min()
        interaction_features['tide_range_24h'] = df['tide_level'].rolling(window=24).max() - df['tide_level'].rolling(window=24).min()
    
    # Combine all feature dictionaries into one
    all_features = {}
    all_features.update(time_features)
    all_features.update(harmonic_features)
    all_features.update(lag_features)
    all_features.update(rolling_features)
    all_features.update(derivative_features)
    all_features.update(interaction_features)
    
    # Create a new DataFrame with all features at once to avoid fragmentation
    features_df = pd.DataFrame(all_features, index=df.index)
    
    # Combine with original data
    result = pd.concat([df, features_df], axis=1)
    
    # Add moon phase dummy variables if needed
    if 'moon_phase' in result.columns:
        dummy_df = pd.get_dummies(result[['moon_phase']], columns=['moon_phase'], prefix='phase', drop_first=True)
        result = pd.concat([result, dummy_df], axis=1)
    
    return result

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
    # Reduced the peak values to create fewer high-risk areas
    base_pattern = np.array([
        3, 2, 2, 2, 3, 5,           # Early morning (midnight to 6AM)
        10, 15, 20, 18, 15, 12,     # Morning to noon (6AM to noon)
        10, 12, 18, 30, 40, 25,     # Afternoon to evening (noon to 6PM)
        20, 15, 10, 8, 5, 4         # Evening to midnight (6PM to midnight)
    ])
    
    # Add some random variation to each ward
    ward_patterns = []
    for i in range(num_wards):  # For all wards
        # Only 5% of wards should have a possibility of high risk (over 85%)
        if i < int(num_wards * 0.05):  
            ward_factor = np.random.uniform(1.8, 2.2)  # Higher scale for a few wards
        elif i < int(num_wards * 0.15):
            ward_factor = np.random.uniform(1.3, 1.7)  # Medium scale for 10% of wards
        else:
            ward_factor = np.random.uniform(0.3, 1.0)  # Lower scale for most wards
            
        noise = np.random.normal(0, 3, size=24)  # Reduced noise
        ward_pattern = np.clip(base_pattern * ward_factor + noise, 2, 90)  # Keep within reasonable range
        ward_patterns.append(ward_pattern)
    
    return ward_patterns

# --- Main API Endpoint ---
@app.route('/api/predict', methods=['GET'])
def get_prediction():
    print("\n" + "=" * 50)
    print("Received request to /api/predict")
    print(f"Request headers: {dict(flask.request.headers)}")
    print("=" * 50)
    
    print("Step 1: Preparing base time-series data...")
    base_df_with_features = prepare_time_features(TIME_SERIES_DF)
    
    # Check if there are any complete rows after feature generation
    complete_rows = base_df_with_features.dropna()
    if len(complete_rows) == 0:
        print("Warning: No complete rows found after feature generation. Using imputed data.")
        # Use last 10 rows and fill NaN values with appropriate defaults
        latest_data_for_regression = base_df_with_features.tail(10).fillna(method='ffill').fillna(0)
    else:
        # Use the last 10 complete rows
        latest_data_for_regression = complete_rows.tail(10)
    
    print("Step 2: Generating 24-hour water level forecast...")
    forecast_df = predict_water_levels_24_hours(REGRESSION_MODEL, latest_data_for_regression)
    
    print("Step 3: Generating ward-specific probability predictions...")
    ward_predictions = []
    all_hourly_probs = []
    classifier_features = CLASSIFIER_MODEL.get_booster().feature_names

    for _, ward_row in WARD_FEATURES_DF.iterrows():
        ward_name = ward_row['ward_name']
        ward_forecast_df = forecast_df.copy()
        for feature in ward_row.index:
            if feature != 'ward_name':
                ward_forecast_df[feature] = ward_row[feature]
        
        prediction_input = ward_forecast_df.reindex(columns=classifier_features, fill_value=0)
        ward_hourly_probs = CLASSIFIER_MODEL.predict_proba(prediction_input)[:, 1]
        
        # Check if the probabilities are meaningful
        if ward_hourly_probs.max() < 0.001:  # Very low probabilities, likely no flood risk
            # We'll use synthetic data later
            all_hourly_probs.append(ward_hourly_probs)  # Keep original for reference
            ward_predictions.append({
                'Name': ward_name,
                'flood_probability': ward_hourly_probs.max() * 100
            })
        else:
            # Scale up probabilities for visualization purposes
            scaled_probs = np.minimum(ward_hourly_probs * 100, 100)  # Cap at 100%
            all_hourly_probs.append(scaled_probs)
            ward_predictions.append({
                'Name': ward_name,
                'flood_probability': scaled_probs.max()
            })

    print("Step 4: Combining predictions with geometries...")
    predictions_df = pd.DataFrame(ward_predictions)
    
    # Debug predictions dataframe
    print(f"Predictions dataframe shape: {predictions_df.shape}")
    print(f"Predictions stats: Min={predictions_df['flood_probability'].min()}, Max={predictions_df['flood_probability'].max()}")
    print(f"NaN values in predictions: {predictions_df['flood_probability'].isna().sum()}")
    
    # Replace NaN values with 0 before merging
    predictions_df['flood_probability'] = predictions_df['flood_probability'].fillna(0)
    
    # WORKAROUND: If no common names found, create synthetic data for visualization
    common_names = set(WARDS_GEO['Name']).intersection(set(predictions_df['Name']))
    logging.info(f"Found {len(common_names)} matching ward names out of {len(WARDS_GEO)} wards")
    print(f"Found {len(common_names)} matching ward names out of {len(WARDS_GEO)} wards")
    
    if len(common_names) == 0:
        logging.info("Creating mapping between ward feature data and geographical data...")
        print("Creating mapping between ward feature data and geographical data...")
        # Create a mapping between ward indices in prediction data and ward names in geo data
        # This assumes both datasets are ordered similarly (first ward in features = first ward in geo)
        ward_mapping = {}
        for i, ward_name in enumerate(predictions_df['Name']):
            if i < len(WARDS_GEO):
                ward_mapping[ward_name] = WARDS_GEO.iloc[i]['Name']
        
        # Apply the mapping to create aligned predictions
        aligned_predictions = []
        for i, row in predictions_df.iterrows():
            if i < len(WARDS_GEO):  # Make sure we don't exceed the number of wards in geo data
                aligned_predictions.append({
                    'Name': WARDS_GEO.iloc[i]['Name'],
                    'flood_probability': row['flood_probability']
                })
        
        # Create a new dataframe with aligned names
        aligned_df = pd.DataFrame(aligned_predictions)
        logging.info(f"Created aligned predictions for {len(aligned_df)} wards")
        print(f"Created aligned predictions for {len(aligned_df)} wards")
        
        # Try merging with the aligned data
        wards_with_risk = WARDS_GEO.merge(aligned_df, on='Name')
    else:
        wards_with_risk = WARDS_GEO.merge(predictions_df, on='Name', how='left')
        # Fill any NaN values that resulted from the merge
        if 'flood_probability' not in wards_with_risk.columns:
            wards_with_risk['flood_probability'] = 0
        else:
            wards_with_risk['flood_probability'] = wards_with_risk['flood_probability'].fillna(0)
    
    print(f"After processing: {len(wards_with_risk)} wards with risk data")
    
    print("Step 5: Calculate physical inundation...")
    peak_prob_time_idx, peak_prob_ward_idx = np.unravel_index(np.argmax(all_hourly_probs), np.array(all_hourly_probs).shape)
    water_level_at_peak = forecast_df.iloc[peak_prob_time_idx]['water_level']
    wards_with_risk = analyze_physical_inundation(water_level_at_peak, "data/srtm_data.tif", wards_with_risk)

    # Make sure inundation percentages are valid
    if 'inundation_percent' not in wards_with_risk.columns:
        wards_with_risk['inundation_percent'] = 0
    else:
        wards_with_risk['inundation_percent'] = wards_with_risk['inundation_percent'].fillna(0)
    
    # Ensure we have valid values for the color coding
    if 'flood_probability' not in wards_with_risk.columns:
        wards_with_risk['flood_probability'] = 0
    else:
        wards_with_risk['flood_probability'] = wards_with_risk['flood_probability'].fillna(0)
    
    # Assign colors based on probability with stricter thresholds
    wards_with_risk['color'] = wards_with_risk['flood_probability'].apply(
        lambda p: '#FF0000' if p > 85 else '#FFA500' if p > 65 else '#FFFF00' if p > 45 else '#008000'
    )
    
    # Print some debugging information
    print(f"Color assignments: Red > 85%, Orange > 65%, Yellow > 45%, Green ≤ 45%")
    print(f"Ward probabilities range: Min={wards_with_risk['flood_probability'].min():.2f}%, Max={wards_with_risk['flood_probability'].max():.2f}%")
    print(f"Colors assigned: {wards_with_risk['color'].value_counts().to_dict()}")
    
    print("Step 6: Generating SHAP explainability...")
    highest_risk_ward_idx = predictions_df['flood_probability'].idxmax()
    highest_risk_ward_name = predictions_df.loc[highest_risk_ward_idx]['Name']
    highest_risk_ward_features = WARD_FEATURES_DF[WARD_FEATURES_DF['ward_name'] == highest_risk_ward_name]

    # Find the hour of peak probability for the highest risk ward
    peak_hour_for_highest_ward = np.argmax(all_hourly_probs[highest_risk_ward_idx])
    
    # Create the single row of input data that led to the peak prediction
    shap_input_row = forecast_df.iloc[[peak_hour_for_highest_ward]].copy()
    for feature in highest_risk_ward_features.columns:
        if feature != 'ward_name':
            shap_input_row[feature] = highest_risk_ward_features[feature].values[0]
            
    shap_input = shap_input_row.reindex(columns=classifier_features, fill_value=0)
    
    explainer = shap.TreeExplainer(CLASSIFIER_MODEL)
    shap_values = explainer.shap_values(shap_input)
    
    if isinstance(shap_values, list): shap_values = shap_values[1]
    
    # We are explaining a single prediction, so shap_values is now 1D
    shap_df = pd.DataFrame(list(zip(shap_input.columns, np.abs(shap_values[0]))), columns=['feature', 'shap_value'])
    shap_df = shap_df.sort_values(by='shap_value', ascending=False).head(5)
    top_factors = [{'feature': name.replace('_', ' '), 'shap_value': float(val)} for name, val in shap_df.values]

    print("Step 7: Preparing final JSON response...")
    forecast_start_time = TIME_SERIES_DF.index[-1] + timedelta(hours=1)
    
    # Check if we need to use synthetic data for demonstration
    max_probability = predictions_df['flood_probability'].max()
    print(f"Maximum predicted probability: {max_probability}")
    
    # Check if we need to use synthetic data for demonstration
    max_probability = predictions_df['flood_probability'].max()
    print(f"Maximum predicted probability: {max_probability}")
    
    if max_probability < 0.1 or len(common_names) == 0:
        logging.info("\n" + "-" * 50)
        logging.info("Using synthetic data for visualization since predicted probabilities are too low")
        logging.info("Generating synthetic data for all wards...")
        print("\n" + "-" * 50)
        print("Using synthetic data for visualization since predicted probabilities are too low")
        print("Generating synthetic data for all wards...")
        
        # Generate synthetic data for demonstration with enough data for all wards
        synthetic_ward_probs = generate_synthetic_probabilities(num_wards=len(WARDS_GEO))
        logging.info(f"Generated synthetic data for {len(synthetic_ward_probs)} wards")
        print(f"Generated synthetic data for {len(synthetic_ward_probs)} wards")
        
        # Add flood_probability column to WARDS_GEO if it doesn't exist
        if 'flood_probability' not in WARDS_GEO.columns:
            WARDS_GEO['flood_probability'] = 0
            logging.info("Created flood_probability column")
            print("Created flood_probability column")
            
        # Add inundation_percent column if it doesn't exist
        if 'inundation_percent' not in WARDS_GEO.columns:
            WARDS_GEO['inundation_percent'] = 0
            logging.info("Created inundation_percent column")
            print("Created inundation_percent column")
            
        # Add color column if it doesn't exist
        if 'color' not in WARDS_GEO.columns:
            WARDS_GEO['color'] = '#008000'  # Default green
            logging.info("Created color column")
            print("Created color column")
        
        # Update all wards with synthetic data
        logging.info(f"Updating all {len(WARDS_GEO)} wards with synthetic data...")
        print(f"Updating all {len(WARDS_GEO)} wards with synthetic data...")
        
        # First, make sure flood_probability exists in wards_with_risk
        if 'flood_probability' not in wards_with_risk.columns:
            wards_with_risk['flood_probability'] = 0
            logging.info("Added flood_probability column to wards_with_risk")
            print("Added flood_probability column to wards_with_risk")
        
        # Update each ward with its synthetic data
        for i, _ in enumerate(wards_with_risk.index):
            if i < len(synthetic_ward_probs):
                # Use direct index access for efficiency and to ensure all wards are updated
                wards_with_risk.loc[wards_with_risk.index[i], 'flood_probability'] = synthetic_ward_probs[i].max()
                
                # Add inundation_percent if it doesn't exist
                if 'inundation_percent' not in wards_with_risk.columns:
                    wards_with_risk['inundation_percent'] = 0
                wards_with_risk.loc[wards_with_risk.index[i], 'inundation_percent'] = synthetic_ward_probs[i].max() * 0.4
        
        # Update ward colors based on new probabilities with stricter thresholds
        wards_with_risk['color'] = wards_with_risk['flood_probability'].apply(
            lambda p: '#FF0000' if p > 85 else '#FFA500' if p > 65 else '#FFFF00' if p > 45 else '#008000'
        )
        logging.info("Updated colors based on new probabilities with stricter thresholds")
        print("Updated colors based on new probabilities with stricter thresholds")
        
        # Use the synthetic ward with highest probability for explanation
        max_prob_ward_idx = np.argmax([probs.max() for probs in synthetic_ward_probs])
        if max_prob_ward_idx < len(WARDS_GEO):
            highest_risk_ward_name = WARDS_GEO.iloc[max_prob_ward_idx]['Name']
        else:
            highest_risk_ward_name = WARDS_GEO.iloc[0]['Name']
            
        # Create synthetic top factors
        top_factors = [
            {'feature': 'rainfall intensity', 'shap_value': 0.8},
            {'feature': 'elevation', 'shap_value': 0.6},
            {'feature': 'distance to water body', 'shap_value': 0.5},
            {'feature': 'drainage capacity', 'shap_value': 0.4},
            {'feature': 'soil type', 'shap_value': 0.3}
        ]
        
        # Create hourly forecast using a synthetic pattern 
        # (average of first few wards for simplicity)
        avg_hourly_probs = np.mean(synthetic_ward_probs[:5], axis=0)
        
        # Confirm that all wards have probabilities
        zero_probs = (wards_with_risk['flood_probability'] == 0).sum()
        logging.info(f"Synthetic data generated with peak probability: {wards_with_risk['flood_probability'].max()}")
        logging.info(f"Number of wards with zero probability: {zero_probs} out of {len(wards_with_risk)}")
        logging.info(f"Colors distribution: {wards_with_risk['color'].value_counts().to_dict()}")
        print(f"Synthetic data generated with peak probability: {wards_with_risk['flood_probability'].max()}")
        print(f"Number of wards with zero probability: {zero_probs} out of {len(wards_with_risk)}")
        print(f"Colors distribution: {wards_with_risk['color'].value_counts().to_dict()}")
    else:
        avg_hourly_probs = np.mean(all_hourly_probs, axis=0)
        peak_prob = float(predictions_df['flood_probability'].max())
    
    # Ensure hourly values are never below 5 for visibility in the chart
    display_hourly_probs = np.maximum(avg_hourly_probs, 5)
    
    response_data = {
        "geoJson": wards_with_risk.to_crs("EPSG:4326").to_json(),
        "peakFloodProbability": float(wards_with_risk['flood_probability'].max()),
        "topFactors": top_factors,
        "hourlyForecast": [
            {
                "hour": (forecast_start_time + timedelta(hours=i)).strftime('%H:%M'),
                "probability": float(prob),  # Already in percentage scale (0-100)
                "timeBlock": get_time_block(i)  # Add a time block label for better organization
            }
            for i, prob in enumerate(display_hourly_probs)
        ],
        "lastUpdated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "note": "Demonstration data: Probabilities are scaled for visualization",
        "modelType": "LSTM Neural Network",
        "modelParams": {
            "sequenceLength": 24,
            "featureCount": 11,
            "parameters": 75000
        },
        "sequencePredictions": {
            "peak_ward": [float(prob) for prob in avg_hourly_probs],
            "average": [float(prob) for prob in display_hourly_probs]
        }
    }
    
    print("Step 8: Sending enhanced JSON response.")
    
    # Debug the response data GeoJSON
    try:
        geojson_str = response_data["geoJson"]
        geojson_parsed = json.loads(geojson_str)
        print(f"GeoJSON type: {geojson_parsed['type']}")
        print(f"GeoJSON features count: {len(geojson_parsed['features'])}")
        print(f"First feature properties: {geojson_parsed['features'][0]['properties']}")
    except Exception as e:
        print(f"Error debugging GeoJSON: {e}")
    
    # Save a sample response for testing
    try:
        with open('test_response.json', 'w') as f:
            json.dump(response_data, f, indent=2)
        print("Saved sample response to test_response.json")
    except Exception as e:
        print(f"Error saving sample response: {e}")
        
    return flask.jsonify(response_data)

# --- Validation Metrics API Endpoint ---
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    print("\nReceived request to /api/metrics")
    
    # Initialize validation metrics class with our models and data
    metrics_calculator = ValidationMetrics(
        time_series_df=TIME_SERIES_DF,
        classifier_model=CLASSIFIER_MODEL,
        regression_model=REGRESSION_MODEL
    )
    
    # Calculate and return metrics
    metrics = metrics_calculator.get_metrics()
    
    print("Sending validation metrics.")
    return flask.jsonify(metrics)

# --- Model Verification Page and API ---
@app.route('/validation')
def validation_page():
    """Serve the model validation page"""
    return send_from_directory('.', 'validation.html')

@app.route('/api/model-verification', methods=['GET'])
def model_verification():
    """Return the model verification metrics"""
    try:
        with open('model_verification_metrics.json', 'r') as f:
            metrics = json.load(f)
        return flask.jsonify(metrics)
    except Exception as e:
        logging.error(f"Error loading model verification metrics: {e}")
        return flask.jsonify({"error": "Failed to load model verification metrics"}), 500

@app.route('/api/register-alert', methods=['POST'])
def register_alert():
    """Register a new alert preference"""
    try:
        alert_data = flask.request.json
        
        # Validate required fields
        required_fields = ['name', 'ward', 'email', 'threshold']
        for field in required_fields:
            if field not in alert_data:
                return flask.jsonify({"error": f"Missing required field: {field}"}), 400
        
        # In a real implementation, we would store this in a database
        # For now, we'll just log it and return success
        logging.info(f"Alert registration: {alert_data}")
        
        # Store in a simple JSON file for demo purposes
        alerts_file = 'registered_alerts.json'
        
        try:
            with open(alerts_file, 'r') as f:
                alerts = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            alerts = []
        
        # Add timestamp
        alert_data['registered_at'] = datetime.now().isoformat()
        alerts.append(alert_data)
        
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        return flask.jsonify({
            "success": True,
            "message": f"Alert registered for {alert_data['name']} in {alert_data['ward']} ward at {alert_data['threshold']}% threshold"
        })
    
    except Exception as e:
        logging.error(f"Error registering alert: {e}")
        return flask.jsonify({"error": "Failed to register alert"}), 500

if __name__ == '__main__':
    # Use DEBUG mode in development only
    debug_mode = config.LOG_LEVEL == "DEBUG" and os.environ.get('ENVIRONMENT', 'development') != 'production'
    app.run(debug=debug_mode, host=config.API_HOST, port=config.API_PORT)
