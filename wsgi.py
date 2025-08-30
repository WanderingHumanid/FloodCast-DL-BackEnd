"""
WSGI entry point for the FloodCast application
"""

import os
from app import app

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
