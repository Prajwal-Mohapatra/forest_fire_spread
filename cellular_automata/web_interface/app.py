#!/usr/bin/env python3
"""
Forest Fire Simulation Web Application
Main Flask application serving the React frontend and API endpoints
"""

import os
import sys
from flask import Flask, render_template, send_from_directory, send_file
from flask_cors import CORS

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "cellular_automata"))

# Import the API blueprint from api.py
from api import app as api_app

# Define React app paths
REACT_BUILD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "build")
TEMPLATE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Create main Flask app
app = Flask(__name__, 
            template_folder=TEMPLATE_FOLDER)

# Configure static file serving for React
# React build creates files with /static/ prefix, so we need to handle both
app.static_folder = REACT_BUILD_FOLDER
app.static_url_path = ''

# Enable CORS
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(project_root, "cellular_automata", "uploads")
app.config['OUTPUT_FOLDER'] = os.path.join(project_root, "cellular_automata", "outputs")

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Register API blueprint
app.register_blueprint(api_app, url_prefix='/api')

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
def react_static_files(filename):
    """Serve React static files (CSS, JS, images)"""
    static_dir = os.path.join(REACT_BUILD_FOLDER, 'static')
    if os.path.exists(os.path.join(static_dir, filename)):
        return send_from_directory(static_dir, filename)
    # Fallback to original static directory if file not found in React build
    return send_from_directory('static', filename)

@app.route('/manifest.json')
def manifest():
    """Serve React manifest.json"""
    manifest_path = os.path.join(REACT_BUILD_FOLDER, 'manifest.json')
    if os.path.exists(manifest_path):
        return send_file(manifest_path)
    return "Manifest not found", 404

@app.route('/favicon.ico')
def favicon():
    """Serve React favicon"""
    favicon_path = os.path.join(REACT_BUILD_FOLDER, 'favicon.ico')
    if os.path.exists(favicon_path):
        return send_file(favicon_path)
    return "Favicon not found", 404

@app.route('/<path:filename>')
def react_assets(filename):
    """Serve other React assets and handle React routing"""
    # Don't intercept API routes
    if filename.startswith('api/'):
        return "API route not found", 404
    
    # Don't intercept specific routes
    if filename in ['old', 'demo']:
        return "Route handled elsewhere", 404
    
    # Check if it's a static asset in the build directory
    asset_path = os.path.join(REACT_BUILD_FOLDER, filename)
    if os.path.exists(asset_path):
        return send_file(asset_path)
    
    # For all other routes (React Router), serve index.html
    index_path = os.path.join(REACT_BUILD_FOLDER, 'index.html')
    if os.path.exists(index_path):
        return send_file(index_path)
    else:
        return "React app not built. Please run 'npm run build' in the frontend directory.", 404

# React app routes (catch-all - must be last)
@app.route('/', defaults={'path': ''})
def serve_react(path=''):
    """Serve the React app - any unmatched route will be handled by React Router"""
    # For root path, serve React index.html
    index_path = os.path.join(REACT_BUILD_FOLDER, 'index.html')
    if os.path.exists(index_path):
        return send_file(index_path)
    else:
        return "React app not built. Please run 'npm run build' in the frontend directory.", 404

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
