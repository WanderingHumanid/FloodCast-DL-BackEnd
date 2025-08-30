"""
Test the GeoJSON format and coordinate structure
"""

import json

# Simplified version of the mock data
mock_geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[76.294, 9.965], [76.301, 9.965], [76.304, 9.957], [76.294, 9.957], [76.294, 9.965]]]
            },
            "properties": {
                "Name": "Kadavanthra",
                "ward_number": "1",
                "flood_probability": 85,
                "inundation_percent": 68,
                "color": "#ff9900"
            }
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[76.305, 9.959], [76.312, 9.959], [76.312, 9.952], [76.305, 9.952], [76.305, 9.959]]]
            },
            "properties": {
                "Name": "Elamkulam",
                "ward_number": "2",
                "flood_probability": 65,
                "inundation_percent": 52,
                "color": "#ffff00"
            }
        }
    ]
}

# Test that the format is correct
print("Testing GeoJSON format...")
print(f"Type: {mock_geojson['type']}")
print(f"Features count: {len(mock_geojson['features'])}")
print(f"First feature type: {mock_geojson['features'][0]['type']}")
print(f"First feature geometry type: {mock_geojson['features'][0]['geometry']['type']}")
print(f"First feature coordinates: {mock_geojson['features'][0]['geometry']['coordinates']}")

# Stringify and parse to ensure the format works
print("\nTesting JSON serialization and deserialization...")
json_str = json.dumps(mock_geojson)
print(f"JSON string length: {len(json_str)}")

# Now parse it back
parsed = json.loads(json_str)
print(f"Parsed type: {parsed['type']}")
print(f"Parsed features count: {len(parsed['features'])}")

# Double check the structure when used in a response
response = {
    "geoJson": json_str,
    "peakFloodProbability": 85,
    "topFactors": [{"feature": "Water Level", "shap_value": 0.75}],
    "hourlyForecast": [{"hour": "00:00", "probability": 50}],
    "lastUpdated": "2023-07-01T12:00:00Z"
}

# This simulates what the frontend would do
print("\nTesting frontend parsing...")
frontend_geojson = json.loads(response["geoJson"])
print(f"Frontend parsed type: {frontend_geojson['type']}")
print(f"Frontend parsed features count: {len(frontend_geojson['features'])}")
print(f"First feature name: {frontend_geojson['features'][0]['properties']['Name']}")
print(f"First feature color: {frontend_geojson['features'][0]['properties']['color']}")

print("\nTest completed successfully!")
