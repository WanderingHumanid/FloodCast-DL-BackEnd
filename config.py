# FloodCast Configuration

# Model Selection
USE_ENHANCED_MODEL = True

# Model Paths
ENHANCED_MODEL_PATH = "models/floodsense_xgb_enhanced.pkl"
ORIGINAL_MODEL_PATH = "models/floodsense_xgb_model_tuned.pkl"
CLASSIFIER_MODEL_PATH = "models/floodsense_spatio_temporal_classifier.pkl"

# Data Paths
TIDAL_DATA_PATH = "data/merged_flood_moon_tide_data.csv"
NON_TIDAL_DATA_PATH = "data/merged_flood_moon_data.csv"
WARD_FEATURES_PATH = "data/ward_features.csv"
WARDS_GEOMETRY_PATH = "data/kochi_wards.kml"

# Forecast Settings
FORECAST_HOURS = 24

# API Settings
API_PORT = 5000
API_HOST = '127.0.0.1'

# Logging
LOG_FILE = "logs/app_debug.log"
LOG_LEVEL = "DEBUG"
