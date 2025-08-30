# FloodCast Configuration
import os

# Helper function to parse boolean strings
def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f"Boolean value expected, got: {value}")

# App Information
APP_VERSION = os.environ.get('APP_VERSION', '1.0.0')
FLASK_ENV = os.environ.get('FLASK_ENV', 'development')

# Model Selection
USE_ENHANCED_MODEL = parse_bool(os.environ.get('USE_ENHANCED_MODEL', 'True'))

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

# Validate LOG_LEVEL - ensure it's a valid logging level
_log_level = os.environ.get('LOG_LEVEL', "DEBUG")
valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LOG_LEVEL = _log_level if _log_level in valid_log_levels else "INFO"
