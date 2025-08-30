# validate_frontend.py
# This script tests the backend API and verifies the response structure for the frontend

import requests
import json
import time
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import sys

def test_api_response():
    """Test the backend API response structure"""
    print("Testing backend API response...")
    
    try:
        # Make API request
        start_time = time.time()
        response = requests.get("http://127.0.0.1:5000/api/predict", timeout=30)
        end_time = time.time()
        
        # Check response status
        if response.status_code == 200:
            print(f"✅ API responded with status 200 (Success)")
            print(f"Response time: {end_time - start_time:.2f} seconds")
        else:
            print(f"❌ API responded with status {response.status_code}")
            return None
        
        # Parse JSON response
        try:
            data = response.json()
            print(f"✅ Response is valid JSON")
            return data
        except json.JSONDecodeError as e:
            print(f"❌ Error decoding JSON: {e}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to API: {e}")
        print("Make sure the backend server is running at http://127.0.0.1:5000")
        return None

def validate_geojson_structure(data):
    """Validate the GeoJSON structure in the response"""
    print("\nValidating GeoJSON structure...")
    
    if data is None or "geoJson" not in data:
        print("❌ No GeoJSON data found in the response")
        return False
    
    try:
        # Parse GeoJSON
        geojson_data = json.loads(data["geoJson"])
        
        # Basic structure validation
        if "type" not in geojson_data:
            print("❌ GeoJSON is missing 'type' field")
            return False
            
        if "features" not in geojson_data:
            print("❌ GeoJSON is missing 'features' array")
            return False
            
        features = geojson_data["features"]
        feature_count = len(features)
        print(f"✅ GeoJSON contains {feature_count} features")
        
        # Check first feature structure
        if feature_count > 0:
            sample_feature = features[0]
            print("\nSample feature properties:")
            if "properties" in sample_feature:
                properties = sample_feature["properties"]
                for key, value in properties.items():
                    print(f"  - {key}: {value}")
                
                # Check required properties
                required_props = ["Name", "flood_probability", "inundation_percent", "color"]
                missing_props = [prop for prop in required_props if prop not in properties]
                
                if missing_props:
                    print(f"❌ Missing required properties: {', '.join(missing_props)}")
                    return False
                else:
                    print("✅ All required properties present")
            else:
                print("❌ Feature is missing 'properties' object")
                return False
                
        # Convert to GeoDataFrame for visualization
        gdf = gpd.GeoDataFrame.from_features(features)
        
        # Create a simple visualization
        if len(gdf) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf.plot(column='flood_probability', ax=ax, legend=True, 
                    legend_kwds={'label': 'Flood Probability (%)'}, 
                    cmap='RdYlGn_r')
            ax.set_title('Flood Risk Visualization')
            plt.savefig('geojson_validation.png')
            print("✅ GeoJSON visualization saved to geojson_validation.png")
        
        return True
    except Exception as e:
        print(f"❌ Error validating GeoJSON: {e}")
        return False

def validate_hourly_forecast(data):
    """Validate the hourly forecast data structure"""
    print("\nValidating hourly forecast data...")
    
    if data is None or "hourlyForecast" not in data:
        print("❌ No hourly forecast data found in the response")
        return False
    
    try:
        hourly_data = data["hourlyForecast"]
        
        if not isinstance(hourly_data, list):
            print("❌ Hourly forecast data is not an array")
            return False
            
        print(f"✅ Hourly forecast contains {len(hourly_data)} time points")
        
        # Check data structure
        if len(hourly_data) > 0:
            sample_hour = hourly_data[0]
            print("\nSample hourly data point:")
            for key, value in sample_hour.items():
                print(f"  - {key}: {value}")
            
            # Check required properties
            required_props = ["hour", "probability"]
            missing_props = [prop for prop in required_props if prop not in sample_hour]
            
            if missing_props:
                print(f"❌ Missing required properties: {', '.join(missing_props)}")
                return False
            else:
                print("✅ All required properties present")
                
        # Create a dataframe for visualization
        df = pd.DataFrame(hourly_data)
        
        if len(df) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(df["hour"], df["probability"], marker='o')
            plt.title('24-Hour Flood Probability Forecast')
            plt.xlabel('Hour')
            plt.ylabel('Probability (%)')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('hourly_forecast_validation.png')
            print("✅ Hourly forecast visualization saved to hourly_forecast_validation.png")
        
        return True
    except Exception as e:
        print(f"❌ Error validating hourly forecast: {e}")
        return False

def validate_top_factors(data):
    """Validate the top factors data structure"""
    print("\nValidating top risk factors data...")
    
    if data is None or "topFactors" not in data:
        print("❌ No top factors data found in the response")
        return False
    
    try:
        factors_data = data["topFactors"]
        
        if not isinstance(factors_data, list):
            print("❌ Top factors data is not an array")
            return False
            
        print(f"✅ Response contains {len(factors_data)} risk factors")
        
        # Check data structure
        if len(factors_data) > 0:
            print("\nTop risk factors:")
            for i, factor in enumerate(factors_data, 1):
                factor_name = factor.get("feature", "Unknown")
                factor_value = factor.get("shap_value", 0)
                print(f"  {i}. {factor_name}: {factor_value}")
            
            # Check required properties in first factor
            sample_factor = factors_data[0]
            required_props = ["feature", "shap_value"]
            missing_props = [prop for prop in required_props if prop not in sample_factor]
            
            if missing_props:
                print(f"❌ Missing required properties: {', '.join(missing_props)}")
                return False
            else:
                print("✅ All required properties present")
        
        # Create a visualization
        if len(factors_data) > 0:
            factor_names = [f["feature"] for f in factors_data]
            factor_values = [f["shap_value"] for f in factors_data]
            
            plt.figure(figsize=(10, 6))
            plt.barh(factor_names, factor_values)
            plt.xlabel('Importance Value')
            plt.title('Top Risk Factors')
            plt.tight_layout()
            plt.savefig('risk_factors_validation.png')
            print("✅ Risk factors visualization saved to risk_factors_validation.png")
        
        return True
    except Exception as e:
        print(f"❌ Error validating top factors: {e}")
        return False

def validate_peak_probability(data):
    """Validate the peak flood probability value"""
    print("\nValidating peak flood probability...")
    
    if data is None or "peakFloodProbability" not in data:
        print("❌ No peak flood probability found in the response")
        return False
    
    try:
        peak_prob = data["peakFloodProbability"]
        
        if not isinstance(peak_prob, (int, float)):
            print(f"❌ Peak probability is not a number: {peak_prob}")
            return False
            
        print(f"✅ Peak flood probability: {peak_prob:.2f}%")
        
        if peak_prob < 0 or peak_prob > 100:
            print(f"❌ Warning: Probability value is outside the expected range [0-100]")
        
        return True
    except Exception as e:
        print(f"❌ Error validating peak probability: {e}")
        return False

def main():
    """Main validation function"""
    print("=== FloodSense Frontend Validation ===")
    
    # Test API response
    response_data = test_api_response()
    
    if response_data is None:
        print("\n❌ API validation failed. Unable to proceed with further validation.")
        return
    
    # Validate response components
    validation_results = []
    
    # 1. GeoJSON structure
    geojson_ok = validate_geojson_structure(response_data)
    validation_results.append(("GeoJSON Structure", geojson_ok))
    
    # 2. Hourly forecast
    forecast_ok = validate_hourly_forecast(response_data)
    validation_results.append(("Hourly Forecast Data", forecast_ok))
    
    # 3. Top factors
    factors_ok = validate_top_factors(response_data)
    validation_results.append(("Risk Factors Data", factors_ok))
    
    # 4. Peak probability
    peak_ok = validate_peak_probability(response_data)
    validation_results.append(("Peak Probability Value", peak_ok))
    
    # Print validation summary
    print("\n=== Validation Summary ===")
    for test, result in validation_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test}")
    
    # Overall assessment
    all_passed = all([result for _, result in validation_results])
    
    if all_passed:
        print("\n✅ All validation tests PASSED. The frontend should display correct data.")
    else:
        print("\n❌ Some validation tests FAILED. The frontend may not display data correctly.")
    
    print("\nValidation complete. Check the output files for visualizations.")

if __name__ == "__main__":
    main()
