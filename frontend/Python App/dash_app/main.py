"""
Application entry point - initializes the Dash app, sets up caching,
registers callbacks, and runs the server.
"""

import os
import subprocess
from datetime import datetime
from flask_caching import Cache

# Import the app from layout module
from .layout import app

# Local imports
from .config_and_data import config, get_dashboard_file_path

# Register all callbacks (this imports all callback modules)
from .callbacks import register_callbacks

# Setup cache based on configuration
cache_config = config["app_settings"]["cache"]
cache = Cache(app.server, config={
    'CACHE_TYPE': cache_config["type"],
    'CACHE_DIR': cache_config["directory"],
    'CACHE_DEFAULT_TIMEOUT': cache_config["default_timeout"]
})

# Register all callbacks
register_callbacks(app, cache)

if __name__ == "__main__":
    # Check if dashboard file exists, create if not
    dashboard_file = get_dashboard_file_path()
    
    if not os.path.exists(dashboard_file):
        print(f"Dashboard file {dashboard_file} does not exist. Creating it...")
        subprocess.run(["python", "create_dash.py"])
    else:
        print(f"Dashboard file {dashboard_file} already exists. Using existing file.")
    
    # Get server settings from config
    server_config = config["app_settings"]["server"]
    host = server_config["host"]
    port = server_config["port"]
    debug = config["app_settings"]["debug_mode"]
    
    # Run the server
    app.run_server(debug=debug, host=host, port=port)
