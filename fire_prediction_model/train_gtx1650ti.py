#!/usr/bin/env python3
"""
Memory-optimized training script for GTX 1650 Ti (4GB VRAM).
Includes automatic batch size adjustment and memory monitoring.
"""

import os
import sys
import gc
import numpy as np
import tensorflow as tf
import yaml
import glob
import warnings
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure TensorFlow for limited GPU memory
def configure_tensorflow_for_gtx1650ti():
    """Optimized TensorFlow configuration for GTX 1650 Ti."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Critical: Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # For GTX 1650 Ti, use memory growth instead of hard limit
            print(f"✅ GPU memory growth enabled for GTX 1650 Ti")
            
        except RuntimeError as e:
            print(f"⚠️ GPU configuration warning: {e}")
    
    # Environment variables for stability
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    # Enable mixed precision to save memory
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("✅ Mixed precision enabled (saves ~50% GPU memory)")
    except:
        print("⚠️ Mixed precision not available, using float32")

# Configure immediately
configure_tensorflow_for_gtx1650ti()

# Import project modules after TF configuration
from model.resunet_a import build_resunet_a
from dataset.loader import FireDatasetGenerator
from utils.metrics import (
    iou_score, dice_coef, focal_loss, combined_loss, compute_class_weights,
    precision_score, recall_score, f1_score
)
from utils.versioning import ModelVersionManager
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau

class MemoryMonitor(tf.keras.callbacks.Callback):
    """Monitor GPU memory usage during training."""
    def on_epoch_end(self, epoch, logs=None):
        if tf.config.experimental.list_physical_devices('GPU'):
            try:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                current_mb = mem_info['current'] / 1024 / 1024
                peak_mb = mem_info['peak'] / 1024 / 1024
                print(f"GPU Memory - Current: {current_mb:.0f}MB / Peak: {peak_mb:.0f}MB")
                
                # Force garbage collection if memory usage is high
                if current_mb > 3000:  # If using more than 3GB
                    gc.collect()
                    tf.keras.backend.clear_session()
                    
            except Exception as e:
                print(f"Memory monitoring failed: {e}")

def auto_adjust_batch_size():
    """Automatically determine optimal batch size for GTX 1650 Ti."""
    print("🔧 Auto-adjusting batch size for your hardware...")
    
    # Start with a conservative batch size for GTX 1650 Ti
    optimal_batch_size = 2
    
    try:
        # Create a minimal model to test memory usage
        test_model = build_resunet_a(
            input_shape=(256, 256, 9),
            num_classes=1,
            use_enhanced_aspp=False  # Disable ASPP to save memory
        )
        
        test_model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Test different batch sizes
        for batch_size in [2, 4, 6, 8]:
            try:
                dummy_input = np.random.rand(batch_size, 256, 256, 9).astype(np.float32)
                dummy_target = np.random.rand(batch_size, 256, 256, 1).astype(np.float32)
                
                # Test if this batch size works
                test_model.train_on_batch(dummy_input, dummy_target)
                optimal_batch_size = batch_size
                print(f"✅ Batch size {batch_size} works")
                
            except tf.errors.ResourceExhaustedError:
                print(f"❌ Batch size {batch_size} causes OOM")
                break
            except Exception as e:
                print(f"⚠️ Batch size {batch_size} failed: {e}")
                break
        
        # Clean up
        del test_model
        tf.keras.backend.clear_session()
        gc.collect()
        
    except Exception as e:
        print(f"⚠️ Auto-adjustment failed: {e}, using conservative batch size")
        optimal_batch_size = 2
    
    print(f"🎯 Optimal batch size for your GPU: {optimal_batch_size}")
    return optimal_batch_size

def load_optimized_config():
    """Load configuration optimized for GTX 1650 Ti."""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}
    
    # Override with GTX 1650 Ti optimizations
    config.update({
        'model': {
            'use_enhanced_aspp': False,  # Disable to save memory
            'input_shape': [256, 256, 9],
            'num_classes': 1
        },
        'training': {
            'batch_size': auto_adjust_batch_size(),
            'epochs': 20,  # Reduced for testing
            'learning_rate': 1e-4,
            'use_focal_loss': True,
            'use_combined_loss': False,
            'n_patches_per_img_train': 15,  # Reduced
            'n_patches_per_img_val': 8     # Reduced
        },
        'loss': {
            'focal_alpha': 0.25,
            'focal_gamma': 2.0
        }
    })
    
    return config

def main():
    """Main training function with memory optimization."""
    print("🚀 Starting Memory-Optimized Training for GTX 1650 Ti")
    print("=" * 60)
    
    # Load optimized configuration
    config = load_optimized_config()
    print(f"📋 Configuration optimized for GTX 1650 Ti")
    print(f"🔧 Batch size: {config['training']['batch_size']}")
    print(f"🔧 Enhanced ASPP: {config['model']['use_enhanced_aspp']}")
    
    # Initialize version manager
    version_manager = ModelVersionManager(base_dir="outputs")
    
    # Setup paths
    base_dir = '/home/swayam/projects/forest_fire_spread/fire-probability-prediction-map-unstacked-data/dataset_stacked'
    train_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_04_*.tif')))
    val_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_05_0[1-3]*.tif')))  # Reduced validation files
    
    print(f"📁 Found {len(train_files)} training files")
    print(f"📁 Found {len(val_files)} validation files")
    
    if not train_files:
        print("❌ No training files found!")
        return
    
    # Create memory-efficient generators
    batch_size = config['training']['batch_size']
    train_gen = FireDatasetGenerator(
        train_files,
        batch_size=batch_size,
        n_patches_per_img=config['training']['n_patches_per_img_train'],
        shuffle=True,
        fire_sample_ratio=0.7  # Higher fire ratio for better learning
    )
    
    val_gen = FireDatasetGenerator(
        val_files,
        batch_size=batch_size,
        n_patches_per_img=config['training']['n_patches_per_img_val'],
        shuffle=False,
        fire_sample_ratio=0.5
    )
    
    print(f"📊 Training batches: {len(train_gen)}")
    print(f"📊 Validation batches: {len(val_gen)}")
    
    # Compute class weights
    try:
        fire_weight, no_fire_weight = compute_class_weights(train_gen, num_samples=5)
    except Exception as e:
        print(f"⚠️ Class weight computation failed: {e}")
        fire_weight, no_fire_weight = 10.0, 1.0
    
    # Create model
    print("🏗️ Creating optimized model...")
    model = build_resunet_a(
        input_shape=tuple(config['model']['input_shape']),
        num_classes=config['model']['num_classes'],
        use_enhanced_aspp=config['model']['use_enhanced_aspp']
    )
    
    print(f"🔢 Model parameters: {model.count_params():,}")
    
    # Setup loss function
    if config['training']['use_focal_loss']:
        loss_fn = focal_loss(
            alpha=fire_weight/(fire_weight + no_fire_weight),
            gamma=config['loss']['focal_gamma']
        )
        loss_name = "Focal_Loss"
    else:
        loss_fn = 'binary_crossentropy'
        loss_name = "BCE"
    
    # Compile model with mixed precision considerations
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
    if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[iou_score, dice_coef, 'accuracy']
    )
    
    print(f"✅ Model compiled with {loss_name}")
    
    # Setup callbacks with memory monitoring
    os.makedirs('outputs/checkpoints', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            'outputs/checkpoints/model_best_gtx1650ti.weights.h5',
            monitor='val_iou_score',
            save_best_only=True,
            save_weights_only=True,  # Save weights only to save space
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,  # Reduced patience
            min_lr=1e-7,
            verbose=1
        ),
        MemoryMonitor(),
        CSVLogger('outputs/logs/training_gtx1650ti.csv', append=True)
    ]
    
    # Training with memory optimization
    epochs = config['training']['epochs']
    print(f"\n🚀 Starting training for {epochs} epochs...")
    
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            workers=1,  # Single worker to prevent multiprocessing issues
            use_multiprocessing=False,
            max_queue_size=5,  # Reduced queue size
            shuffle=False  # Generator handles shuffling
        )
        
        print("✅ Training completed successfully!")
        
        # Save model version
        training_metadata = {
            'config': config,
            'loss_function': loss_name,
            'hardware': 'GTX_1650_Ti',
            'batch_size': batch_size,
            'epochs': epochs,
            'class_weights': {'fire': float(fire_weight), 'no_fire': float(no_fire_weight)},
            'metrics': {
                'val_iou_score': history.history['val_iou_score'],
                'val_dice_coef': history.history['val_dice_coef'],
                'val_loss': history.history['val_loss']
            }
        }
        
        version_id = version_manager.save_model_version(model, training_metadata)
        print(f"💾 Model version saved: {version_id}")
        
        # Plot results
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f'{loss_name} Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['iou_score'], label='Train IoU')
        plt.plot(history.history['val_iou_score'], label='Val IoU')
        plt.title('IoU Score')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history.history['dice_coef'], label='Train Dice')
        plt.plot(history.history['val_dice_coef'], label='Val Dice')
        plt.title('Dice Coefficient')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'outputs/logs/training_results_gtx1650ti_{version_id}.png', dpi=150)
        plt.show()
        
    except tf.errors.ResourceExhaustedError as e:
        print(f"❌ Out of GPU memory: {e}")
        print("💡 Try reducing batch size further or disable ASPP module")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == "__main__":
    main()
