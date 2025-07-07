# ====================
# CA Engine Setup and Test Script
# ====================
"""
Complete setup and testing script for the Forest Fire Cellular Automata Engine.
Installs dependencies, tests functionality, and demonstrates integration.
"""

import os
import sys
import subprocess
import platform
from datetime import datetime

def print_banner():
    """Print project banner."""
    print("🔥" * 60)
    print("🔥 FOREST FIRE CELLULAR AUTOMATA ENGINE")
    print("🔥 Uttarakhand Fire Spread Simulation System")  
    print("🔥" * 60)
    print(f"📅 Setup Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Platform: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version}")
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    major, minor = sys.version_info[:2]
    
    if major == 3 and minor >= 8:
        print(f"✅ Python {major}.{minor} is compatible")
        return True
    else:
        print(f"❌ Python {major}.{minor} is not compatible (requires Python 3.8+)")
        return False

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    
    requirements_path = os.path.join(
        os.path.dirname(__file__), "requirements.txt"
    )
    
    if not os.path.exists(requirements_path):
        print(f"❌ Requirements file not found: {requirements_path}")
        return False
    
    try:
        # Install packages
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_path
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Requirements installed successfully")
            return True
        else:
            print(f"❌ Failed to install requirements:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Installation timeout (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def verify_tensorflow():
    """Verify TensorFlow installation and GPU availability."""
    print("🤖 Verifying TensorFlow...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} installed")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🎮 GPU devices found: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("⚠️  No GPU found, will use CPU")
        
        # Test basic operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = a + b
        print(f"🧮 TensorFlow test: [1,2,3] + [4,5,6] = {c.numpy()}")
        
        return True
        
    except ImportError:
        print("❌ TensorFlow not available")
        return False
    except Exception as e:
        print(f"❌ TensorFlow error: {e}")
        return False

def verify_geospatial():
    """Verify geospatial libraries."""
    print("🗺️  Verifying geospatial libraries...")
    
    libraries = ['rasterio', 'fiona', 'shapely']
    success = True
    
    for lib in libraries:
        try:
            __import__(lib)
            print(f"✅ {lib} available")
        except ImportError:
            print(f"❌ {lib} not available")
            success = False
    
    return success

def run_ca_tests():
    """Run CA engine tests."""
    print("🧪 Running CA engine tests...")
    
    test_script = os.path.join(os.path.dirname(__file__), "test_ca_engine.py")
    
    if not os.path.exists(test_script):
        print(f"❌ Test script not found: {test_script}")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, test_script
        ], capture_output=True, text=True, timeout=120)
        
        print("📋 Test Output:")
        print("-" * 40)
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Test Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ CA engine tests passed")
            return True
        else:
            print("❌ CA engine tests failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timeout (2 minutes)")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_ml_integration():
    """Test ML model integration."""
    print("🔗 Testing ML integration...")
    
    try:
        # Check if ML model files exist
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ml_model_path = os.path.join(
            project_root, "forest_fire_ml", "", "outputs", "final_model.h5"
        )
        
        if os.path.exists(ml_model_path):
            print(f"✅ ML model found: {ml_model_path}")
            
            # Test integration bridge
            integration_script = os.path.join(
                os.path.dirname(__file__), "integration", "ml_ca_bridge.py"
            )
            
            if os.path.exists(integration_script):
                print("✅ ML-CA bridge script available")
                return True
            else:
                print("❌ ML-CA bridge script not found")
                return False
        else:
            print("⚠️  ML model not found (using synthetic data)")
            return True
            
    except Exception as e:
        print(f"❌ ML integration test error: {e}")
        return False

def start_web_interface():
    """Start the web interface."""
    print("🌐 Starting web interface...")
    
    api_script = os.path.join(
        os.path.dirname(__file__), "web_interface", "api.py"
    )
    
    if not os.path.exists(api_script):
        print(f"❌ Web API script not found: {api_script}")
        return False
    
    print("🚀 Starting Flask development server...")
    print("   URL: http://localhost:5000")
    print("   Demo: http://localhost:5000/static/index.html")
    print("   Press Ctrl+C to stop")
    print()
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, api_script])
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Web server stopped")
        return True
    except Exception as e:
        print(f"❌ Web server error: {e}")
        return False

def create_project_structure():
    """Create necessary project directories."""
    print("📁 Creating project structure...")
    
    base_dir = os.path.dirname(__file__)
    
    directories = [
        "outputs",
        "test_data", 
        "uploads",
        "web_interface/templates",
        "integration"
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True

def generate_demo_data():
    """Generate demo data for testing."""
    print("📊 Generating demo data...")
    
    try:
        # Import test function
        sys.path.append(os.path.dirname(__file__))
        from test_ca_engine import create_synthetic_probability_map
        
        demo_dir = os.path.join(os.path.dirname(__file__), "test_data")
        os.makedirs(demo_dir, exist_ok=True)
        
        # Create multiple demo maps
        demo_maps = [
            ("demo_dehradun.tif", 300, 300),
            ("demo_rishikesh.tif", 250, 250),
            ("demo_nainital.tif", 200, 200)
        ]
        
        for filename, width, height in demo_maps:
            filepath = os.path.join(demo_dir, filename)
            if create_synthetic_probability_map(filepath, width, height):
                print(f"✅ Created demo map: {filename}")
            else:
                print(f"❌ Failed to create: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo data generation failed: {e}")
        return False

def print_summary(results):
    """Print setup summary."""
    print()
    print("📋 SETUP SUMMARY")
    print("=" * 50)
    
    total_checks = len(results)
    passed_checks = sum(1 for result in results.values() if result)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:.<30} {status}")
    
    print("=" * 50)
    print(f"Overall: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("🎉 All checks passed! CA engine is ready.")
        print()
        print("🚀 Next steps:")
        print("1. Run web interface: python cellular_automata/web_interface/api.py")
        print("2. Open browser: http://localhost:5000/static/index.html")
        print("3. Try demo scenarios or create custom simulations")
    else:
        print("⚠️  Some checks failed. Please review the output above.")
        print()
        print("💡 Common solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Check Python version (requires 3.8+)")
        print("- Verify TensorFlow GPU setup if needed")

def main():
    """Main setup function."""
    print_banner()
    
    # Track all setup steps
    results = {}
    
    # Step 1: Check Python version
    results["Python Version"] = check_python_version()
    
    # Step 2: Create project structure
    results["Project Structure"] = create_project_structure()
    
    # Step 3: Install requirements
    if results["Python Version"]:
        results["Package Installation"] = install_requirements()
    else:
        results["Package Installation"] = False
    
    # Step 4: Verify key libraries
    if results["Package Installation"]:
        results["TensorFlow Verification"] = verify_tensorflow()
        results["Geospatial Libraries"] = verify_geospatial()
    else:
        results["TensorFlow Verification"] = False
        results["Geospatial Libraries"] = False
    
    # Step 5: Generate demo data
    results["Demo Data Generation"] = generate_demo_data()
    
    # Step 6: Run tests
    if all([results["TensorFlow Verification"], results["Geospatial Libraries"]]):
        results["CA Engine Tests"] = run_ca_tests()
        results["ML Integration"] = test_ml_integration()
    else:
        results["CA Engine Tests"] = False
        results["ML Integration"] = False
    
    # Print summary
    print_summary(results)
    
    # Optionally start web interface
    if results["CA Engine Tests"]:
        print()
        start_web = input("🌐 Start web interface now? (y/n): ").lower().strip()
        if start_web in ['y', 'yes']:
            start_web_interface()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)
