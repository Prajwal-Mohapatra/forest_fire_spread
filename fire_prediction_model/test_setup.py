#!/usr/bin/env python3
"""
Test script to debug generator issues and TensorFlow configuration.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_tensorflow_setup():
    """Test TensorFlow configuration."""
    print("🔧 Testing TensorFlow setup...")
    
    # Configure TensorFlow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration warning: {e}")
    
    # Test basic TensorFlow operations
    try:
        x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        y = tf.matmul(x, x)
        print(f"✅ TensorFlow basic operations work: {y.numpy()}")
        return True
    except Exception as e:
        print(f"❌ TensorFlow basic operations failed: {e}")
        return False

def test_generator():
    """Test the data generator."""
    print("\n📊 Testing data generator...")
    
    try:
        from dataset.loader import FireDatasetGenerator
        import glob
        
        # Find some test files
        base_dir = '/home/swayam/projects/forest_fire_spread/fire-probability-prediction-map-unstacked-data/dataset_stacked'
        test_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_04_0[1-3].tif')))
        
        if not test_files:
            print("❌ No test files found, skipping generator test")
            return False
        
        print(f"📁 Found {len(test_files)} test files")
        
        # Create small test generator
        gen = FireDatasetGenerator(
            test_files[:2],  # Use only 2 files
            batch_size=2,
            n_patches_per_img=5,  # Small number of patches
            shuffle=False
        )
        
        print(f"✅ Generator created successfully with {len(gen)} batches")
        
        # Test fetching a batch
        print("🔄 Testing batch generation...")
        X, Y = gen[0]
        
        print(f"✅ Batch generated successfully:")
        print(f"   Input shape: {X.shape}, dtype: {X.dtype}")
        print(f"   Target shape: {Y.shape}, dtype: {Y.dtype}")
        print(f"   Input range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"   Target range: [{Y.min():.3f}, {Y.max():.3f}]")
        print(f"   Fire pixel ratio: {Y.mean():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation."""
    print("\n🏗️ Testing model creation...")
    
    try:
        from model.resunet_a import build_resunet_a
        
        model = build_resunet_a(
            input_shape=(256, 256, 9),
            num_classes=1,
            use_enhanced_aspp=False  # Start with simpler model
        )
        
        print(f"✅ Model created successfully with {model.count_params():,} parameters")
        
        # Test model compilation
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model compiled successfully")
        
        # Test prediction with dummy data
        dummy_input = np.random.rand(1, 256, 256, 9).astype(np.float32)
        pred = model.predict(dummy_input, verbose=0)
        
        print(f"✅ Model prediction works: output shape {pred.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test generator + model integration."""
    print("\n🔗 Testing generator + model integration...")
    
    try:
        from dataset.loader import FireDatasetGenerator
        from model.resunet_a import build_resunet_a
        import glob
        
        # Setup
        base_dir = '/home/swayam/projects/forest_fire_spread/fire-probability-prediction-map-unstacked-data/dataset_stacked'
        test_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_04_0[1-2].tif')))
        
        if not test_files:
            print("❌ No test files found, skipping integration test")
            return False
        
        # Create minimal generator
        gen = FireDatasetGenerator(
            test_files,
            batch_size=1,
            n_patches_per_img=2,
            shuffle=False
        )
        
        # Create minimal model
        model = build_resunet_a(
            input_shape=(256, 256, 9),
            num_classes=1,
            use_enhanced_aspp=False
        )
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Test one step of training
        print("🔄 Testing one training step...")
        X, Y = gen[0]
        
        # Single training step
        loss = model.train_on_batch(X, Y)
        print(f"✅ Training step successful, loss: {loss}")
        
        # Test evaluation
        eval_loss = model.test_on_batch(X, Y)
        print(f"✅ Evaluation step successful, loss: {eval_loss}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 FOREST FIRE MODEL DIAGNOSTIC TESTS")
    print("=" * 60)
    
    tests = [
        ("TensorFlow Setup", test_tensorflow_setup),
        ("Data Generator", test_generator),
        ("Model Creation", test_model_creation),
        ("Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All tests passed! Your setup should work.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        
        # Provide specific recommendations
        if not results.get("TensorFlow Setup", False):
            print("\n💡 TensorFlow Issues:")
            print("   - Try: pip install tensorflow==2.12.0")
            print("   - Check GPU drivers if using GPU")
        
        if not results.get("Data Generator", False):
            print("\n💡 Data Generator Issues:")
            print("   - Check if data files exist at specified path")
            print("   - Try: pip install rasterio")
        
        if not results.get("Model Creation", False):
            print("   - Model architecture might have issues")
            print("   - Check model imports")

if __name__ == "__main__":
    main()
