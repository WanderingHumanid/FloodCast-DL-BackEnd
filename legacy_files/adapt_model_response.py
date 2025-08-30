"""
adapt_model_response.py

Adapter functions to make the improved model responses compatible with the frontend
"""

import json

def adapt_forecast_response(forecast_data, flood_risks):
    """
    Adapt the improved model forecast response to match what the frontend expects
    
    Args:
        forecast_data: List of hourly forecasts
        flood_risks: Dictionary with ward risk data
    
    Returns:
        dict: Adapted response that frontend can use
    """
    # Convert hourly forecast to frontend format
    hourly_forecast = []
    
    # Make sure we have 24 entries for hourly forecast
    max_entries = min(24, len(forecast_data))
    
    for i in range(max_entries):
        # Extract hour from timestamp (assumes ISO format)
        hour = forecast_data[i]['timestamp'].split('T')[1][:5] if 'T' in forecast_data[i]['timestamp'] else f"{i:02d}:00"
        
        # Calculate a more realistic probability value (as frontend expects probability not water level)
        # Map water level to a probability value between 0-100 based on thresholds
        water_level = forecast_data[i]['water_level']
        base_level = 100  # Base water level considered "normal"
        critical_level = 300  # Water level that would be 100% flood probability
        
        # Calculate probability as percentage between base and critical levels
        probability = max(0, min(100, ((water_level - base_level) / (critical_level - base_level)) * 100))
        
        hourly_forecast.append({
            'hour': hour,
            'probability': probability
        })
    
    # Get peak flood probability
    peak_probability = 0
    if flood_risks and 'peak_risk' in flood_risks:
        peak_probability = flood_risks['peak_risk']['probability']
    
    # Mock top factors that contribute to flood risk
    top_factors = [
        {'feature': 'Water Level', 'shap_value': 0.75},
        {'feature': 'Rainfall', 'shap_value': 0.65},
        {'feature': 'Tide Height', 'shap_value': 0.55},
        {'feature': 'Ground Elevation', 'shap_value': 0.45},
        {'feature': 'Drainage Capacity', 'shap_value': 0.35}
    ]
    
    # Create mock GeoJSON string with ward data
    mock_geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # Add ward features with risk data using proper coordinates
    if flood_risks and 'ward_risks' in flood_risks:
        # Real ward names in Kochi
        ward_names = [
            "Kadavanthra", "Elamkulam", "Vennala", "Palarivattom", "Kaloor", 
            "Panampilly Nagar", "Kochi Central", "Fort Kochi", "Mattancherry", 
            "Edappally", "Thrikkakara", "Kalamassery", "Maradu", "Thripunithura",
            "Ernakulam North", "Ernakulam South", "Pachalam", "Palluruthy", 
            "Vaduthala", "Vyttila", "Cheranalloor", "Vypeen"
        ]
        
        # Real Kochi ward boundary approximations
        # These are more realistic polygon coordinates based on the actual layout of Kochi wards
        ward_polygons = [
            # Kadavanthra (more detailed polygon)
            [[[76.294, 9.965], [76.301, 9.965], [76.304, 9.957], [76.294, 9.957], [76.294, 9.965]]],
            # Elamkulam
            [[[76.305, 9.959], [76.312, 9.959], [76.312, 9.952], [76.305, 9.952], [76.305, 9.959]]],
            # Vennala
            [[[76.323, 9.987], [76.331, 9.987], [76.331, 9.979], [76.323, 9.979], [76.323, 9.987]]],
            # Palarivattom
            [[[76.312, 9.983], [76.322, 9.983], [76.322, 9.975], [76.312, 9.975], [76.312, 9.983]]],
            # Kaloor
            [[[76.302, 9.981], [76.311, 9.981], [76.311, 9.972], [76.302, 9.972], [76.302, 9.981]]],
            # Panampilly Nagar
            [[[76.293, 9.964], [76.299, 9.964], [76.299, 9.958], [76.293, 9.958], [76.293, 9.964]]],
            # Kochi Central
            [[[76.271, 9.966], [76.279, 9.966], [76.279, 9.958], [76.271, 9.958], [76.271, 9.966]]],
            # Fort Kochi (peninsula shape)
            [[[76.238, 9.962], [76.244, 9.962], [76.244, 9.968], [76.238, 9.968], [76.238, 9.962]]],
            # Mattancherry
            [[[76.251, 9.957], [76.259, 9.957], [76.259, 9.950], [76.251, 9.950], [76.251, 9.957]]],
            # Edappally
            [[[76.314, 9.992], [76.324, 9.992], [76.324, 9.984], [76.314, 9.984], [76.314, 9.992]]],
            # Additional wards with more varied shapes
            # Thrikkakara (irregular pentagon)
            [[[76.335, 10.002], [76.341, 10.007], [76.347, 10.002], [76.343, 9.996], [76.335, 9.996], [76.335, 10.002]]],
            # Kalamassery (elongated)
            [[[76.328, 10.012], [76.340, 10.012], [76.340, 10.004], [76.328, 10.004], [76.328, 10.012]]],
            # Maradu (coastal shape)
            [[[76.291, 9.936], [76.306, 9.936], [76.306, 9.920], [76.291, 9.920], [76.291, 9.936]]],
            # Thripunithura (larger area)
            [[[76.314, 9.932], [76.334, 9.932], [76.334, 9.912], [76.314, 9.912], [76.314, 9.932]]],
            # Ernakulam North
            [[[76.282, 9.985], [76.292, 9.985], [76.292, 9.974], [76.282, 9.974], [76.282, 9.985]]],
            # Ernakulam South
            [[[76.280, 9.974], [76.290, 9.974], [76.290, 9.964], [76.280, 9.964], [76.280, 9.974]]],
            # Pachalam
            [[[76.287, 9.994], [76.297, 9.994], [76.297, 9.985], [76.287, 9.985], [76.287, 9.994]]],
            # Palluruthy
            [[[76.263, 9.928], [76.275, 9.928], [76.275, 9.915], [76.263, 9.915], [76.263, 9.928]]],
            # Vaduthala
            [[[76.284, 10.002], [76.294, 10.002], [76.294, 9.994], [76.284, 9.994], [76.284, 10.002]]],
            # Vyttila (triangular)
            [[[76.305, 9.950], [76.318, 9.950], [76.310, 9.936], [76.305, 9.950]]],
            # Cheranalloor
            [[[76.297, 10.008], [76.310, 10.008], [76.310, 9.998], [76.297, 9.998], [76.297, 10.008]]],
            # Vypeen (island shape)
            [[[76.256, 9.985], [76.268, 9.985], [76.274, 9.970], [76.268, 9.960], [76.256, 9.970], [76.256, 9.985]]]
        ]
        
        for i, (ward_num, probability) in enumerate(flood_risks['ward_risks'].items()):
            # Determine color based on probability
            if probability > 85:
                color = '#ff0000'  # Red
            elif probability > 65:
                color = '#ff9900'  # Orange
            elif probability > 45:
                color = '#ffff00'  # Yellow
            else:
                color = '#00ff00'  # Green
            
            # Get ward name from the list or use a default
            ward_name = ward_names[i % len(ward_names)] if i < len(ward_names) else f"Ward {ward_num}"
            
            # Get corresponding ward polygon or create a default if we run out of defined polygons
            ward_polygon = ward_polygons[i % len(ward_polygons)] if i < len(ward_polygons) else [
                [[76.28 + (i * 0.01), 9.97], 
                 [76.28 + (i * 0.01) + 0.01, 9.97], 
                 [76.28 + (i * 0.01) + 0.01, 9.97 + 0.01], 
                 [76.28 + (i * 0.01), 9.97 + 0.01], 
                 [76.28 + (i * 0.01), 9.97]]
            ]
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": ward_polygon
                },
                "properties": {
                    "Name": ward_name,
                    "ward_number": ward_num,
                    "flood_probability": probability,
                    "inundation_percent": min(100, probability * 0.8),  # Mock value
                    "color": color
                }
            }
            mock_geojson["features"].append(feature)
    
    # Build the final response object
    response = {
        "geoJson": json.dumps(mock_geojson),
        "peakFloodProbability": peak_probability,
        "topFactors": top_factors,
        "hourlyForecast": hourly_forecast,
        "lastUpdated": forecast_data[0]['timestamp'] if forecast_data else ""
    }
    
    return response
