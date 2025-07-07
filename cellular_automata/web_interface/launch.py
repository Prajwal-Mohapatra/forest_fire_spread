#!/usr/bin/env python3
"""
Forest Fire Simulation Web Interface - Launch Script
Quick setup and launch script for the web application
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print application banner"""
    print("🔥" * 60)
    print("🔥  FOREST FIRE SIMULATION WEB INTERFACE")
    print("🔥  The Minions - Bharatiya Antariksh Hackathon 2025")
    print("🔥" * 60)
    print()

def check_dependencies():
    """Check if required Python packages are available"""
    print("📋 Checking dependencies...")
    
    required_packages = [
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
    ]
    
    missing_packages = []
    
    for package, display_name in required_packages:
        try:
            __import__(package)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name}")
            missing_packages.append(display_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        
        try:
            for package, display_name in required_packages:
                if display_name in missing_packages:
                    if package == 'flask_cors':
                        package = 'flask-cors'
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    print(f"✅ Installed {display_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e}")
            print("Please install them manually:")
            print("pip install flask flask-cors")
            return False
    
    print("✅ All dependencies satisfied\n")
    return True

def check_file_structure():
    """Check if all required files exist"""
    print("📁 Checking file structure...")
    
    required_files = [
        'app.py',
        'api.py',
        'templates/index.html',
        'static/css/main.css',
        'static/css/components.css',
        'static/js/config.js',
        'static/js/api.js',
        'static/js/ui-components.js',
        'static/js/map-handler.js',
        'static/js/simulation-manager.js',
        'static/js/chart-handler.js',
        'static/js/main.js',
        'static/images/logo.svg'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {len(missing_files)}")
        print("Please ensure all frontend files are in place.")
        return False
    
    print("✅ All files present\n")
    return True

def create_missing_directories():
    """Create any missing directories"""
    directories = [
        'templates',
        'static/css',
        'static/js',
        'static/images',
        'outputs',
        'uploads'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def start_application():
    """Start the Flask application"""
    print("🚀 Starting Forest Fire Simulation Web Application...")
    print("🌐 URL: http://localhost:5000")
    print("📡 API: http://localhost:5000/api")
    print("🎮 Demo: Try the demo scenarios in the sidebar")
    print("\n💡 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Start Flask app
        subprocess.run([sys.executable, 'app.py'], cwd=os.getcwd())
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

def open_browser():
    """Open the application in the default browser"""
    try:
        # Wait a moment for server to start
        time.sleep(2)
        webbrowser.open('http://localhost:5000')
        print("🌐 Opening application in browser...")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("Please open http://localhost:5000 manually")

def show_quick_start_guide():
    """Show quick start instructions"""
    print("\n📖 QUICK START GUIDE")
    print("-" * 30)
    print("1. 🗺️  The map shows Uttarakhand region")
    print("2. 🎯 Click 'Click to Ignite Mode' toggle")
    print("3. 🖱️  Click on map to add ignition points")
    print("4. ⚙️  Adjust weather parameters if needed")
    print("5. 🚀 Click 'Start Simulation' button")
    print("6. 📊 Watch real-time visualization and charts")
    print("7. 🎬 Use animation controls to replay")
    print("8. 📤 Export results when complete")
    print("\n🎮 OR try a demo scenario from the sidebar!")
    print("-" * 50)

def show_troubleshooting():
    """Show troubleshooting tips"""
    print("\n🔧 TROUBLESHOOTING")
    print("-" * 20)
    print("• Map not loading: Check internet connection (needs Leaflet CDN)")
    print("• Charts not showing: Verify Chart.js CDN is accessible")
    print("• API errors: Ensure backend dependencies are installed")
    print("• Simulation fails: Check console for error messages")
    print("• Performance issues: Close other browser tabs")
    print("\n📞 For support: Check the README.md file")
    print("-" * 50)

def main():
    """Main launch function"""
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print_banner()
    
    # Pre-flight checks
    if not check_dependencies():
        print("❌ Dependency check failed. Please install required packages.")
        return 1
    
    create_missing_directories()
    
    if not check_file_structure():
        print("❌ File structure check failed. Please ensure all files are present.")
        return 1
    
    # Show guides
    show_quick_start_guide()
    
    # Ask if user wants to auto-open browser
    try:
        auto_open = input("🌐 Open browser automatically? (y/n): ").lower().strip()
        if auto_open in ['y', 'yes', '']:
            # Start browser opener in background
            import threading
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
    except KeyboardInterrupt:
        print("\n🛑 Cancelled by user")
        return 0
    
    # Start the application
    try:
        start_application()
    except Exception as e:
        print(f"\n❌ Failed to start application: {e}")
        show_troubleshooting()
        return 1
    
    print("\n👋 Thank you for using Forest Fire Simulation!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
