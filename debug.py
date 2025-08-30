"""
Debug script to help troubleshoot Render deployment
"""

import os
import sys

def check_environment():
    """Print information about the environment"""
    print("=== Environment Information ===")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    
    # Check for required directories
    dirs_to_check = ['models', 'data', 'logs', 'utils']
    for d in dirs_to_check:
        if os.path.exists(d):
            print(f"Directory '{d}' exists and contains: {os.listdir(d)}")
        else:
            print(f"Directory '{d}' does not exist")
    
    # Print environment variables
    print("\n=== Environment Variables ===")
    for key, value in os.environ.items():
        print(f"{key}: {value}")

def check_imports():
    """Try importing required modules"""
    print("\n=== Import Checks ===")
    modules = [
        'flask', 'flask_cors', 'pandas', 'numpy', 'joblib', 
        'geopandas', 'shap', 'rasterio'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"Successfully imported {module}")
        except ImportError as e:
            print(f"Failed to import {module}: {e}")

if __name__ == "__main__":
    check_environment()
    check_imports()
    
    print("\nChecking for app.py...")
    if os.path.exists('app.py'):
        print("app.py exists")
        try:
            with open('app.py', 'r', encoding='utf-8') as f:
                first_few_lines = '\n'.join(f.readlines()[:10])
                print(f"First few lines of app.py:\n{first_few_lines}")
        except UnicodeDecodeError:
            print("Could not read app.py due to encoding issues, but the file exists")
    else:
        print("app.py does not exist!")
    
    print("\n=== Debug complete ===")
