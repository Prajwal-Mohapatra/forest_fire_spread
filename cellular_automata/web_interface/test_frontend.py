#!/usr/bin/env python3
"""
Test script for Forest Fire Simulation Web Interface
Runs basic tests to verify frontend components are working correctly
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_file_exists(file_path):
    """Check if a file exists"""
    if os.path.exists(file_path):
        print(f"✅ {file_path}")
        return True
    else:
        print(f"❌ {file_path}")
        return False

def check_frontend_files():
    """Check all frontend files exist"""
    print("🔍 Checking frontend files...")
    
    base_path = Path(__file__).parent
    files_to_check = [
        "app.py",
        "api.py",
        "templates/index.html",
        "static/css/main.css",
        "static/css/components.css",
        "static/js/config.js",
        "static/js/api.js",
        "static/js/ui-components.js",
        "static/js/map-handler.js",
        "static/js/simulation-manager.js",
        "static/js/chart-handler.js",
        "static/js/main.js",
        "static/images/logo.svg"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        full_path = base_path / file_path
        if not check_file_exists(full_path):
            all_exist = False
    
    return all_exist

def check_html_structure():
    """Check HTML template has required elements"""
    print("\n🔍 Checking HTML structure...")
    
    html_path = Path(__file__).parent / "templates/index.html"
    
    if not html_path.exists():
        print("❌ HTML template not found")
        return False
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    required_elements = [
        'id="map-display"',
        'id="start-simulation-btn"',
        'id="area-chart"',
        'id="intensity-chart"',
        'id="progress-chart"',
        'id="comparison-chart"',
        'data-demo="dehradun"',
        'data-demo="rishikesh"',
        'data-demo="nainital"',
        'ignition-mode-toggle',
        'wind-speed',
        'wind-direction',
        'temperature',
        'humidity'
    ]
    
    all_found = True
    for element in required_elements:
        if element in html_content:
            print(f"✅ {element}")
        else:
            print(f"❌ {element}")
            all_found = False
    
    return all_found

def check_javascript_syntax():
    """Check JavaScript files for basic syntax errors"""
    print("\n🔍 Checking JavaScript syntax...")
    
    js_files = [
        "static/js/config.js",
        "static/js/api.js",
        "static/js/ui-components.js",
        "static/js/map-handler.js",
        "static/js/simulation-manager.js",
        "static/js/chart-handler.js",
        "static/js/main.js"
    ]
    
    base_path = Path(__file__).parent
    all_valid = True
    
    for js_file in js_files:
        js_path = base_path / js_file
        if js_path.exists():
            try:
                # Basic syntax check - look for common issues
                with open(js_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for basic syntax issues
                if content.count('{') != content.count('}'):
                    print(f"❌ {js_file} - Mismatched braces")
                    all_valid = False
                elif content.count('(') != content.count(')'):
                    print(f"❌ {js_file} - Mismatched parentheses")
                    all_valid = False
                elif content.count('[') != content.count(']'):
                    print(f"❌ {js_file} - Mismatched brackets")
                    all_valid = False
                else:
                    print(f"✅ {js_file}")
                    
            except Exception as e:
                print(f"❌ {js_file} - Error reading file: {e}")
                all_valid = False
        else:
            print(f"❌ {js_file} - File not found")
            all_valid = False
    
    return all_valid

def start_development_server():
    """Start the Flask development server"""
    print("\n🚀 Starting development server...")
    
    app_path = Path(__file__).parent / "app.py"
    if not app_path.exists():
        print("❌ app.py not found")
        return None
    
    try:
        # Start Flask app in background
        process = subprocess.Popen(
            [sys.executable, str(app_path)],
            cwd=str(app_path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            response = requests.get("http://localhost:5000", timeout=5)
            if response.status_code == 200:
                print("✅ Server started successfully at http://localhost:5000")
                return process
            else:
                print(f"❌ Server returned status code: {response.status_code}")
                process.terminate()
                return None
        except requests.RequestException as e:
            print(f"❌ Server not responding: {e}")
            process.terminate()
            return None
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

def test_api_endpoints():
    """Test basic API endpoints"""
    print("\n🔍 Testing API endpoints...")
    
    base_url = "http://localhost:5000"
    endpoints_to_test = [
        "/",
        "/api/health",
        "/api/available_dates",
        "/api/config"
    ]
    
    all_working = True
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint}")
            else:
                print(f"❌ {endpoint} - Status: {response.status_code}")
                all_working = False
        except requests.RequestException as e:
            print(f"❌ {endpoint} - Error: {e}")
            all_working = False
    
    return all_working

def main():
    """Run all tests"""
    print("🔥 Forest Fire Simulation Frontend Test")
    print("=" * 50)
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Run tests
    tests = [
        ("Frontend Files", check_frontend_files),
        ("HTML Structure", check_html_structure),
        ("JavaScript Syntax", check_javascript_syntax),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Start server and test endpoints
    server_process = start_development_server()
    if server_process:
        results["API Endpoints"] = test_api_endpoints()
        
        # Keep server running for manual testing
        print(f"\n🌐 Server is running at http://localhost:5000")
        print("Press Ctrl+C to stop the server...")
        
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            server_process.terminate()
            server_process.wait()
    else:
        results["API Endpoints"] = False
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 30)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Frontend is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
