#!/usr/bin/env python3
"""
Ultra-lightweight training script for very limited GPU memory (GTX 1650 Ti or similar)
This script uses the smallest possible configuration to fit in ~4GB GPU memory.
"""

import os
import gc
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def configure_ultra_low_memory():
    """Configure TensorFlow for ultra-low memory usage"""
    print("🔧 Configuring TensorFlow for ultra-low memory usage...")
    
    # GPU configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth (critical)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration warning: {e}")
    
    # Ultra-conservative environment settings
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging
    
    # Enable XLA for memory optimization
    tf.config.optimizer.set_jit(True)
    
    print("✅ Ultra-low memory configuration complete")

# Configure immediately
configure_ultra_low_memory()

from model.resunet_a import build_resunet_a
from dataset.loader import FireDatasetGenerator
from utils.metrics import iou_score, dice_coef
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import yaml

class MemoryMonitor(tf.keras.callbacks.Callback):
    """Monitor GPU memory usage during training"""
    
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()  # Force garbage collection
        tf.keras.backend.clear_session()  # Clear any orphaned tensors
        print(f"💾 Epoch {epoch + 1} complete - Memory cleaned")

def find_optimal_batch_size():
    """Find the largest batch size that fits in memory"""
    print("🔍 Finding optimal batch size for ultra-low memory...")
    
    # Ultra-small model for testing
    test_shape = (128, 128, 6)  # Even smaller patches
    
    try:
        test_model = build_resunet_a(
            input_shape=test_shape,
            num_classes=1,
            use_enhanced_aspp=False
        )
        test_model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Test batch sizes starting from 1
        for batch_size in [1, 2]:
            try:
                print(f"Testing batch size {batch_size}...")
                dummy_input = tf.random.normal((batch_size,) + test_shape)
                dummy_target = tf.random.normal((batch_size, test_shape[0], test_shape[1], 1))
                
                # Test forward pass
                with tf.GradientTape() as tape:
                    pred = test_model(dummy_input, training=True)
                    loss = tf.keras.losses.binary_crossentropy(dummy_target, pred)
                
                # Test backward pass
                grads = tape.gradient(loss, test_model.trainable_variables)
                
                print(f"✅ Batch size {batch_size} works")
                optimal_batch_size = batch_size
                
                # Clean up
                del dummy_input, dummy_target, pred, loss, grads
                gc.collect()
                
            except tf.errors.ResourceExhaustedError:
                print(f"❌ Batch size {batch_size} failed - OOM")
                break
                
        del test_model
        gc.collect()
        
        return optimal_batch_size
        
    except Exception as e:
        print(f"❌ Batch size testing failed: {e}")
        return 1  # Fall back to batch size 1

def main():
    print("🚀 Starting Ultra-Lightweight Fire Prediction Training")
    print("=" * 60)
    
    # Ultra-conservative configuration
    config = {
        'data': {
            'patch_size': 128,  # Very small patches
            'channels': 6,      # Minimum channels
            'train_samples_per_epoch': 50,   # Very few samples
            'val_samples_per_epoch': 20
        },
        'training': {
            'epochs': 3,        # Short training for testing
            'learning_rate': 1e-4,
            'patience': 2,
            'min_lr': 1e-6
        }
    }
    
    # Find optimal batch size
    batch_size = find_optimal_batch_size()
    print(f"💡 Using batch size: {batch_size}")
    
    if batch_size < 1:
        print("❌ Cannot fit even batch size 1 in memory!")
        return
    
    # Update config
    config['training']['batch_size'] = batch_size
    
    # Create output directories
    os.makedirs('outputs/checkpoints', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)
    
    print("📊 Building ultra-lightweight model...")
    model = build_resunet_a(
        input_shape=(config['data']['patch_size'], config['data']['patch_size'], config['data']['channels']),
        num_classes=1,
        use_enhanced_aspp=False  # Disable ASPP for memory
    )
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Ultra-lightweight optimizer
    optimizer = Adam(learning_rate=config['training']['learning_rate'], epsilon=1e-7)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[iou_score, dice_coef, 'accuracy']
    )
    
    print("✅ Model compiled")
    
    # Check if we have data
    data_path = "/home/swayam/projects/forest_fire_spread/fire-probability-prediction-map-unstacked-data/dataset_stacked"
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        print("Creating dummy training to test model...")
        
        # Create dummy data for testing
        X_dummy = np.random.rand(10, config['data']['patch_size'], config['data']['patch_size'], config['data']['channels']).astype(np.float32)
        y_dummy = np.random.randint(0, 2, (10, config['data']['patch_size'], config['data']['patch_size'], 1)).astype(np.float32)
        
        print("📈 Running test training with dummy data...")
        
        # Minimal callbacks
        callbacks = [
            ModelCheckpoint(
                'outputs/checkpoints/ultra_light_test.weights.h5',
                monitor='loss',
                save_best_only=True,
                save_weights_only=True,
                mode='min',
                verbose=1
            ),
            MemoryMonitor(),
            CSVLogger('outputs/logs/ultra_light_test.csv')
        ]
        
        # Test training for just 1 epoch
        history = model.fit(
            X_dummy, y_dummy,
            batch_size=batch_size,
            epochs=1,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Test training completed successfully!")
        print(f"Final loss: {history.history['loss'][-1]:.4f}")
        
        return
    
    # If real data exists, set up generators
    print("📂 Setting up data generators...")
    
    train_generator = FireDatasetGenerator(
        data_dir=data_path,
        batch_size=batch_size,
        patch_size=config['data']['patch_size'],
        samples_per_epoch=config['data']['train_samples_per_epoch'],
        channels=config['data']['channels'],
        validation_split=0.8,  # 80% train, 20% val
        is_training=True,
        shuffle=True
    )
    
    val_generator = FireDatasetGenerator(
        data_dir=data_path,
        batch_size=batch_size,
        patch_size=config['data']['patch_size'],
        samples_per_epoch=config['data']['val_samples_per_epoch'],
        channels=config['data']['channels'],
        validation_split=0.8,
        is_training=False,
        shuffle=False
    )
    
    print(f"✅ Generators created - Train: {len(train_generator)}, Val: {len(val_generator)}")
    
    # Ultra-minimal callbacks
    callbacks = [
        ModelCheckpoint(
            'outputs/checkpoints/ultra_light_model.weights.h5',
            monitor='val_iou_score',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config['training']['patience'],
            min_lr=config['training']['min_lr'],
            verbose=1
        ),
        MemoryMonitor(),
        CSVLogger('outputs/logs/ultra_light_training.csv')
    ]
    
    print("📈 Starting ultra-lightweight training...")
    
    try:
        history = model.fit(
            train_generator,
            epochs=config['training']['epochs'],
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Ultra-lightweight training completed successfully!")
        
        # Print final metrics
        final_metrics = {k: v[-1] for k, v in history.history.items()}
        print("\n📊 Final Metrics:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Try reducing batch size or patch size further")

if __name__ == "__main__":
    main()
