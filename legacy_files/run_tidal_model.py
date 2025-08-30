import subprocess
import os

def run_app():
    """Run the new Flask app with tidal data"""
    print("Starting FloodCast server with tidal data support...")
    app_file = "app_with_tidal_data.py"
    
    # Check if the file exists
    if not os.path.exists(app_file):
        print(f"Error: {app_file} not found")
        return
    
    try:
        # Run the Flask app
        subprocess.run(["python", app_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Flask app: {e}")
    except KeyboardInterrupt:
        print("Server stopped by user")

if __name__ == "__main__":
    run_app()
