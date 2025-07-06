# ====================
# CA Engine Test Script
# ====================
"""
Test script for the Forest Fire Cellular Automata Engine.
Tests basic functionality without requiring actual ML model outputs.
"""

import os
import sys
import numpy as np
from datetime import datetime

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_synthetic_probability_map(output_path: str, width: int = 500, height: int = 500):
    """Create a synthetic fire probability map for testing."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
        
        # Create synthetic probability data
        # Higher probabilities in the center, lower at edges
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Gaussian-like distribution with some noise
        distance = np.sqrt(X**2 + Y**2)
        base_prob = np.exp(-distance**2 / 0.5)
        noise = np.random.normal(0, 0.1, (height, width))
        probability = np.clip(base_prob + noise, 0, 1).astype(np.float32)
        
        # Add some high-risk areas
        probability[100:150, 100:150] = np.clip(probability[100:150, 100:150] + 0.3, 0, 1)
        probability[300:350, 300:350] = np.clip(probability[300:350, 300:350] + 0.4, 0, 1)
        
        # Define geographic bounds (roughly Uttarakhand region)
        bounds = (77.0, 29.5, 81.0, 31.5)  # (west, south, east, north)
        transform = from_bounds(*bounds, width, height)
        
        # Save as GeoTIFF
        profile = {
            'driver': 'GTiff',
            'dtype': rasterio.float32,
            'nodata': 0,
            'width': width,
            'height': height,
            'count': 1,
            'crs': 'EPSG:4326',
            'transform': transform
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(probability, 1)
        
        print(f"✅ Synthetic probability map created: {output_path}")
        print(f"   Shape: {probability.shape}")
        print(f"   Range: [{probability.min():.3f}, {probability.max():.3f}]")
        print(f"   Mean: {probability.mean():.3f}")
        
        return output_path
        
    except ImportError:
        print("❌ Rasterio not available, cannot create synthetic data")
        return None
    except Exception as e:
        print(f"❌ Failed to create synthetic probability map: {e}")
        return None

def test_ca_engine_basic():
    """Test basic CA engine functionality."""
    print("\n🧪 Testing CA Engine Basic Functionality")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from cellular_automata.ca_engine import ForestFireCA, run_quick_simulation
        print("✅ CA engine imports successful")
        
        # Create synthetic data
        print("\n📊 Creating synthetic test data...")
        test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        prob_map_path = os.path.join(test_data_dir, "synthetic_probability.tif")
        
        if not create_synthetic_probability_map(prob_map_path, width=200, height=200):
            print("❌ Cannot proceed without synthetic data")
            return False
        
        # Test CA engine initialization
        print("\n🔧 Testing CA engine initialization...")
        ca_engine = ForestFireCA(use_gpu=False)  # Use CPU for testing
        print("✅ CA engine initialized")
        
        # Test probability map loading
        print("\n📥 Testing probability map loading...")
        success = ca_engine.load_base_probability_map(prob_map_path)
        if not success:
            print("❌ Failed to load probability map")
            return False
        print("✅ Probability map loaded successfully")
        
        # Test simulation initialization
        print("\n🔥 Testing simulation initialization...")
        ignition_points = [(79.0, 30.5)]  # Center of Uttarakhand
        weather_params = {
            'wind_direction': 45,
            'wind_speed': 15,
            'temperature': 30,
            'relative_humidity': 40
        }
        
        scenario_id = ca_engine.initialize_simulation(
            ignition_points=ignition_points,
            weather_params=weather_params,
            simulation_hours=3
        )
        print(f"✅ Simulation initialized: {scenario_id}")
        
        # Test simulation steps
        print("\n⏰ Testing simulation steps...")
        for hour in range(1, 4):
            fire_state, stats = ca_engine.step_simulation()
            print(f"   Hour {hour}: {stats['total_burning_cells']} cells burning, "
                  f"max intensity: {stats['max_intensity']:.3f}")
        
        print("✅ Simulation steps completed successfully")
        
        # Test current state retrieval
        current_state = ca_engine.get_current_state()
        print(f"✅ Current state retrieved: shape {current_state.shape}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please install required packages: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_quick_simulation():
    """Test the quick simulation function."""
    print("\n🚀 Testing Quick Simulation Function")
    print("=" * 50)
    
    try:
        from cellular_automata.ca_engine import run_quick_simulation
        
        # Create synthetic data
        test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        prob_map_path = os.path.join(test_data_dir, "synthetic_probability_quick.tif")
        
        if not create_synthetic_probability_map(prob_map_path, width=100, height=100):
            print("❌ Cannot proceed without synthetic data")
            return False
        
        # Run quick simulation
        print("\n🔥 Running quick simulation...")
        results = run_quick_simulation(
            probability_map_path=prob_map_path,
            ignition_points=[(79.0, 30.5)],
            weather_params={
                'wind_direction': 180,  # South wind
                'wind_speed': 20,
                'temperature': 35,
                'relative_humidity': 25
            },
            simulation_hours=2,
            output_dir=os.path.join(test_data_dir, "quick_sim_output")
        )
        
        print(f"✅ Quick simulation completed!")
        print(f"   Scenario ID: {results['scenario_id']}")
        print(f"   Hours simulated: {results['total_hours_simulated']}")
        print(f"   Frames saved: {len(results['frame_paths'])}")
        
        # Print final statistics
        if results['hourly_statistics']:
            final_stats = results['hourly_statistics'][-1]
            print(f"   Final burned area: {final_stats['burned_area_km2']:.2f} km²")
            print(f"   Final burning cells: {final_stats['total_burning_cells']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick simulation test failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability and TensorFlow setup."""
    print("\n🖥️  Testing GPU and TensorFlow Setup")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        print(f"📊 TensorFlow version: {tf.__version__}")
        
        # Check GPU availability
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU devices found: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("⚠️  No GPU devices found, using CPU")
        
        # Test basic TensorFlow operations
        print("\n🧮 Testing TensorFlow operations...")
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = tf.add(a, b)
        print(f"✅ TensorFlow operations working: {c.numpy()}")
        
        return True
        
    except ImportError:
        print("❌ TensorFlow not available")
        return False
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def cleanup_test_data():
    """Clean up test data directory."""
    print("\n🧹 Cleaning up test data...")
    test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    
    try:
        import shutil
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
        print("✅ Test data cleaned up")
    except Exception as e:
        print(f"⚠️  Could not clean up test data: {e}")

def main():
    """Run all CA engine tests."""
    print("🔥 Forest Fire CA Engine Test Suite")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("GPU/TensorFlow Setup", test_gpu_availability),
        ("Basic CA Engine", test_ca_engine_basic),
        ("Quick Simulation", test_quick_simulation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "✅ PASSED" if success else "❌ FAILED"
        except Exception as e:
            results[test_name] = f"❌ ERROR: {e}"
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        print(f"{test_name:.<30} {result}")
    
    # Overall result
    passed = sum(1 for r in results.values() if r.startswith("✅"))
    total = len(results)
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CA engine is ready for integration.")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
    
    # Cleanup
    cleanup_test_data()
    
    print(f"\n🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
