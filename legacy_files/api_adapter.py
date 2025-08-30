"""
API Adapter for FloodCast - Provides compatibility between the new tidal-aware
backend and the existing frontend
"""

import flask
from flask_cors import CORS
import requests
import json
import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta

# --- Configure Logging ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Initialize Flask App ---
app = flask.Flask(__name__)
CORS(app)

# Backend URL
BACKEND_URL = "http://127.0.0.1:5000"

# --- Helper functions ---
def generate_mock_forecast_data():
    """Generate mock data for when the backend is unavailable"""
    # Get the current time
    now = datetime.now()
    
    # Generate mock ward data
    mock_wards = []
    # Assuming 75 wards
    for i in range(75):
        # Randomize the flood probability (5% high risk, 15% medium risk, 80% low risk)
        if i < 4:  # 5% high risk
            prob = np.random.uniform(0.6, 0.95)
            color = "#FF0000"  # Red
            risk = "high"
        elif i < 15:  # 15% medium risk
            prob = np.random.uniform(0.3, 0.6)
            color = "#FFA500"  # Orange
            risk = "medium"
        else:  # 80% low risk
            prob = np.random.uniform(0.01, 0.3)
            color = "#00FF00"  # Green
            risk = "low"
            
        mock_wards.append({
            "name": f"Ward {i+1}",
            "floodProbability": float(prob),
            "riskLevel": risk,
            "color": color
        })
    
    # Generate mock hourly forecast data
    mock_hourly = []
    for i in range(24):
        hour_time = now + timedelta(hours=i)
        mock_hourly.append({
            "hour": hour_time.hour,
            "timeLabel": "Morning" if 6 <= hour_time.hour < 12 else 
                         "Afternoon" if 12 <= hour_time.hour < 18 else 
                         "Evening" if 18 <= hour_time.hour < 22 else "Night",
            "timestamp": hour_time.strftime("%Y-%m-%d %H:%M:%S"),
            "waterLevel": float(500 + 100 * np.sin(i/4)),
            "rainfall": float(np.random.uniform(0, 0.5)),
            "tideLevel": float(300 + 200 * np.sin(i/12 + 2))
        })
        
    # Create the full mock response
    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "currentWaterLevel": float(550),
        "forecastWaterLevel": float(650),
        "wards": mock_wards,
        "hourlyForecast": mock_hourly,
        "modelUsed": "XGBoost with Tidal Data (Mock)",
        "usingTidalData": True
    }

# Load ward geojson once
try:
    WARDS_GEO = gpd.read_file("data/kochi_wards.kml", driver='KML')
except Exception as e:
    logging.error(f"Error loading ward data: {e}")
    WARDS_GEO = None

@app.route('/api/predict', methods=['GET'])
def proxy_predict():
    """Adapter for the /api/forecast endpoint"""
    try:
        # First try to get data from the new backend
        try:
            response = requests.get(f"{BACKEND_URL}/api/forecast", timeout=2)
            if response.ok:
                forecast_data = response.json()
            else:
                raise Exception("Backend service unavailable")
        except:
            # Fallback to mock data if backend is unavailable
            logging.warning("Backend unavailable, using mock data")
            forecast_data = generate_mock_forecast_data()
        
        # Create a GeoDataFrame from the wards data
        if WARDS_GEO is None:
            return flask.jsonify({"error": "Ward geodata not available"}), 500
        
        # Copy the GeoDataFrame to avoid modifying the original
        gdf = WARDS_GEO.copy()
        
        # Add flood probability data to the GeoDataFrame
        ward_data_dict = {ward["name"]: ward for ward in forecast_data["wards"]}
        
        # Add properties to the GeoDataFrame
        gdf["flood_probability"] = gdf["Name"].apply(lambda name: 
            ward_data_dict.get(name, {}).get("floodProbability", 0) * 100)  # Convert to percentage
        
        gdf["color"] = gdf["Name"].apply(lambda name: 
            ward_data_dict.get(name, {}).get("color", "#00FF00"))
        
        # Add inundation percentage (mock data since we don't have real DEM analysis)
        gdf["inundation_percent"] = gdf["flood_probability"] * 0.8  # Simple scaling
        
        # Convert to GeoJSON
        geo_json = json.loads(gdf.to_json())
        
        # Convert to GeoJSON
        geo_json = json.loads(gdf.to_json())
        
        # Format hourly forecast data
        hourly_forecast = []
        for entry in forecast_data["hourlyForecast"]:
            # Get the actual water level from the forecast
            water_level = entry["waterLevel"]
            
            # Use a more sophisticated conversion from water level to probability
            # Based on historical data analysis: water levels > 700 are high risk
            if water_level > 700:
                probability = 85 + (water_level - 700) / 10  # 85-100% for high levels
            elif water_level > 500:
                probability = 45 + (water_level - 500) / 5  # 45-85% for medium-high levels
            elif water_level > 300:
                probability = 20 + (water_level - 300) / 10  # 20-45% for medium levels
            else:
                probability = max(5, water_level / 15)  # 5-20% for low levels
            
            # Cap at 100%
            probability = min(100, probability)
            
            hourly_forecast.append({
                "hour": entry["timeLabel"],
                "probability": probability
            })
        
        # Mock top factors
        top_factors = [
            {"feature": "Water Level", "shap_value": 0.45},
            {"feature": "Rainfall", "shap_value": 0.35}
        ]
        
        # If using tidal data, add it as a top factor
        if forecast_data.get("usingTidalData", False):
            top_factors.insert(0, {"feature": "Tide Level", "shap_value": 0.65})
        
        # Create response in the format expected by the frontend
        response_data = {
            "geoJson": json.dumps(geo_json),
            "peakFloodProbability": max([ward["floodProbability"] for ward in forecast_data["wards"]]) * 100,
            "topFactors": top_factors,
            "hourlyForecast": hourly_forecast,
            "lastUpdated": forecast_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "modelInfo": {
                "usingTidalData": forecast_data.get("usingTidalData", False),
                "modelName": forecast_data.get("modelUsed", "Unknown")
            }
        }
        
        return flask.jsonify(response_data)
    
    except Exception as e:
        logging.error(f"Error in predict API: {str(e)}", exc_info=True)
        return flask.jsonify({"error": str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def proxy_metrics():
    """Adapter for the /api/validation endpoint"""
    try:
        # Try to get data from the new backend
        try:
            response = requests.get(f"{BACKEND_URL}/api/validation", timeout=2)
            if response.ok:
                metrics_data = response.json()
            else:
                raise Exception("Backend metrics unavailable")
        except:
            # Fallback to mock metrics if backend is unavailable
            logging.warning("Backend metrics unavailable, using mock data")
            metrics_data = generate_mock_metrics_data()
        
        # Return the data
        return flask.jsonify(metrics_data)
    
    except Exception as e:
        logging.error(f"Error in metrics API: {str(e)}", exc_info=True)
        return flask.jsonify({"error": str(e)}), 500

def generate_mock_metrics_data():
    """Generate mock metrics data for when the backend is unavailable"""
    return {
        "water_level_mae": 80.51,
        "water_level_rmse": 119.65,
        "water_level_r2": 0.7392,
        "water_level_mape": 15.3,
        "detection_rate": 85.2,
        "false_alarm_rate": 12.5,
        "lead_time_hours": 4.5,
        "confidence_level": 0.75,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "top_features": [
            {"feature": "tide_level", "importance": 0.35, "normalized_importance": 35.0},
            {"feature": "rain", "importance": 0.25, "normalized_importance": 25.0},
            {"feature": "temperature", "importance": 0.15, "normalized_importance": 15.0},
            {"feature": "moon_illumination_fraction", "importance": 0.10, "normalized_importance": 10.0},
            {"feature": "sea_level_pressure", "importance": 0.08, "normalized_importance": 8.0},
            {"feature": "wind_speed", "importance": 0.07, "normalized_importance": 7.0}
        ],
        "modelInfo": {
            "usingTidalData": True,
            "modelName": "XGBoost with Tidal Data (Mock)",
            "tidalDataAvailable": True
        }
    }

@app.route('/api/status', methods=['GET'])
def get_status():
    """Status endpoint"""
    return flask.jsonify({
        "status": "ok",
        "version": "1.0.0-adapter",
        "adapterMode": True
    })

# --- Start Flask App ---
if __name__ == '__main__':
    print("Starting FloodCast API Adapter on port 5001")
    print(f"Forwarding requests to backend at {BACKEND_URL}")
    app.run(debug=True, host='0.0.0.0', port=5001)
