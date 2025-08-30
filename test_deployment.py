"""
Test script to verify the FloodCast backend API after deployment
"""

import os
import requests
import json
import sys

# Get the base URL from environment or use default Render URL
BASE_URL = os.environ.get('API_URL', 'https://floodcast-backend.onrender.com')

def test_api_health():
    """Test the API health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_prediction_endpoints():
    """Test the prediction endpoints with sample data"""
    sample_data = {
        "rainfall": 30.5,
        "soil_moisture": 0.4,
        "elevation": 10.2,
        "slope": 0.05,
        "river_distance": 1200,
        "ward": 2
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Prediction status: {response.status_code}")
        if response.status_code == 200:
            print(f"Prediction response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"Prediction error: {response.text}")
            return False
    except Exception as e:
        print(f"Prediction request failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("=== FloodCast Backend API Tests ===")
    print(f"Testing API at: {BASE_URL}")
    
    health_result = test_api_health()
    prediction_result = test_prediction_endpoints()
    
    print("\n=== Test Results ===")
    print(f"Health check: {'PASS' if health_result else 'FAIL'}")
    print(f"Prediction test: {'PASS' if prediction_result else 'FAIL'}")
    
    if health_result and prediction_result:
        print("\n🎉 All tests passed! The API is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
