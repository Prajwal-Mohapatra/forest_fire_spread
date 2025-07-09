#!/usr/bin/env python3
"""
Forest Fire Simulation Web Application
Main Flask application serving the React frontend and API endpoints
"""

import os
import sys
from flask import Flask, render_template, send_from_directory, send_file

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "cellular_automata"))

# Import the API blueprint from api.py
from api import app as api_app

# Define React app paths
REACT_BUILD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "build")

# Create main Flask app
app = Flask(__name__, static_folder=REACT_BUILD_FOLDER, static_url_path='/')

# Register API blueprint
app.register_blueprint(api_app, url_prefix='/api')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    """Serve the React app - any unmatched route will be handled by React Router"""
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_file(os.path.join(app.static_folder, 'index.html'))

# Keep the old routes for backwards compatibility during transition
@app.route('/old')
def old_index():
    """Serve the original application page"""
    return render_template('index.html')

@app.route('/demo')
def demo():
    """Serve the demo page"""
    return render_template('demo.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files from the original app"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("🔥 Forest Fire Simulation Web Application")
    print("=" * 60)
    print("🌐 Frontend: http://localhost:5000")
    print("📡 API: http://localhost:5000/api")
    print("🎮 Demo: http://localhost:5000/demo")
    print("=" * 60)
    print("NOTE: To use the React frontend, build it first with 'npm run build' in the frontend directory")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
