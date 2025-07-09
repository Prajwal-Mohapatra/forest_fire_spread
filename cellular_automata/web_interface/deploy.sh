#!/usr/bin/env bash

# Build and Deploy Script for Forest Fire Simulation Web Interface
# This script builds the React app and configures the Flask backend to serve it

set -e

# Define directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
FLASK_DIR="$SCRIPT_DIR"

# Display header
echo "🔥 Forest Fire Simulation Web Interface Deployment"
echo "=================================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js and npm first."
    exit 1
fi

echo "✅ Node.js installed: $(node --version)"
echo "✅ npm installed: $(npm --version)"

# Install dependencies if node_modules doesn't exist
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "📦 Installing React dependencies..."
    cd "$FRONTEND_DIR" || exit 1
    npm install
    cd "$SCRIPT_DIR" || exit 1
else
    echo "✅ Dependencies already installed"
fi

# Build React app
echo "🏗️  Building React application..."
cd "$FRONTEND_DIR" || exit 1
npm run build
cd "$SCRIPT_DIR" || exit 1

echo "✅ React build completed"

# Set up Flask to serve the React app
echo "🔄 Setting up Flask to serve the React app..."
cp "$SCRIPT_DIR/app_with_react.py" "$SCRIPT_DIR/app.py"
echo "✅ Flask configuration updated"

echo "=================================================="
echo "🚀 Deployment complete!"
echo "🌐 Start the application with: python app.py"
echo "   Then access the web interface at: http://localhost:5000"
echo "=================================================="
