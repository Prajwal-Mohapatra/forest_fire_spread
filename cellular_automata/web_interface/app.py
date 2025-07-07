#!/usr/bin/env python3
"""
Forest Fire Simulation Web Application
Main Flask application serving the frontend and API endpoints
"""

import os
import sys
from flask import Flask, render_template, send_from_directory

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "cellular_automata"))

# Import the API blueprint from api.py
from api import app as api_app

# Create main Flask app
app = Flask(__name__)

# Register API blueprint
app.register_blueprint(api_app, url_prefix='')

# Configure template and static folders
app.template_folder = 'templates'
app.static_folder = 'static'

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/demo')
def demo():
    """Serve the demo page"""
    return render_template('demo.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    print("🔥 Forest Fire Simulation Web Application")
    print("=" * 60)
    print("🌐 Frontend: http://localhost:5000")
    print("📡 API: http://localhost:5000/api")
    print("🎮 Demo: http://localhost:5000/demo")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
