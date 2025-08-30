# FloodCast Configuration
import os
from distutils.util import strtobool

# Model Selection
USE_ENHANCED_MODEL = bool(strtobool(os.environ.get('USE_ENHANCED_MODEL', 'True')))

# Model Paths
ENHANCED_MODEL_PATH = os.environ.get('ENHANCED_MODEL_PATH', "models/floodsense_xgb_enhanced.pkl")
ORIGINAL_MODEL_PATH = os.environ.get('ORIGINAL_MODEL_PATH', "models/floodsense_xgb_model_tuned.pkl")
CLASSIFIER_MODEL_PATH = os.environ.get('CLASSIFIER_MODEL_PATH', "models/floodsense_spatio_temporal_classifier.pkl")

# Data Paths
TIDAL_DATA_PATH = os.environ.get('TIDAL_DATA_PATH', "data/merged_flood_moon_tide_data.csv")
NON_TIDAL_DATA_PATH = os.environ.get('NON_TIDAL_DATA_PATH', "data/merged_flood_moon_data.csv")
WARD_FEATURES_PATH = os.environ.get('WARD_FEATURES_PATH', "data/ward_features.csv")
WARDS_GEOMETRY_PATH = os.environ.get('WARDS_GEOMETRY_PATH', "data/kochi_wards.kml")

# Forecast Settings
FORECAST_HOURS = int(os.environ.get('FORECAST_HOURS', 24))

# API Settings
API_PORT = int(os.environ.get('PORT', os.environ.get('API_PORT', 5000)))
API_HOST = os.environ.get('API_HOST', '0.0.0.0')  # Use 0.0.0.0 to bind to all interfaces

# Logging
LOG_FILE = os.environ.get('LOG_FILE', "logs/app_debug.log")
LOG_LEVEL = os.environ.get('LOG_LEVEL', "DEBUG")
