import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from model.resunet_a import build_resunet_a
from dataset.loader import FireDatasetGenerator
from utils.metrics import iou_score, dice_coef
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# ====== Enhanced GPU Memory Logger ======
class GPUMemoryLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                mem = tf.config.experimental.get_memory_info('GPU:0')
                used = np.round(mem['current'] / 1024 / 1024)
                total = np.round(mem['peak'] / 1024 / 1024)
                print(f"GPU Memory - Used: {used} MB / Total Peak: {total} MB")
            except:
                pass

class ClassBalanceLogger(tf.keras.callbacks.Callback):
    """Log class distribution periodically."""
    def __init__(self, generator, log_freq=5):
        self.generator = generator
        self.log_freq = log_freq
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_freq == 0:
            # Sample some batches to check class balance
            pos, neg = 0, 0
            for i in range(min(3, len(self.generator))):
                _, masks = self.generator[i]
                pos += np.sum(masks)
                neg += np.sum(1 - masks)
            
            total = pos + neg
            pos_ratio = pos / total if total > 0 else 0
            print(f"Epoch {epoch} - Fire pixel ratio: {pos_ratio:.4f}")

# ====== Corrected Paths (Local) ======
base_dir = '/home/swayam/projects/forest_fire_spread/fire-probability-prediction-map-unstacked-data/dataset_stacked'
train_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_04_*.tif')))
val_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_05_0[1-7]*.tif')))
test_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_05_2[3-9]*.tif')))

print(f"Found {len(train_files)} training files")
print(f"Found {len(val_files)} validation files")
print(f"Found {len(test_files)} test files")

if len(train_files) == 0:
    print("❌ No training files found! Check the path.")
    exit(1)

# ====== Enhanced Data Generators ======
train_gen = FireDatasetGenerator(
    train_files, 
    batch_size=4,  # Reduced for memory efficiency
    n_patches_per_img=30,  # Increased patches
    shuffle=True
)
val_gen = FireDatasetGenerator(
    val_files, 
    batch_size=4, 
    n_patches_per_img=20,
    shuffle=False  # Don't shuffle validation
)

# ====== Proper Class Weight Estimation ======
print("Computing comprehensive class weights...")
pos, neg = 0, 0
sample_batches = min(20, len(train_gen))  # Sample more batches

for i in range(sample_batches):
    try:
        _, masks = train_gen[i]
        pos += np.sum(masks)
        neg += np.sum(1 - masks)
    except Exception as e:
        print(f"Error in batch {i}: {e}")
        continue

total_pixels = pos + neg
if total_pixels > 0:
    pos_weight = neg / pos if pos > 0 else 1.0
    neg_weight = 1.0
    
    # Cap extreme weights
    pos_weight = min(pos_weight, 100.0)
    
    print(f"✅ Class weights - Fire: {pos_weight:.4f}, No-Fire: {neg_weight:.4f}")
    print(f"Fire pixel ratio: {pos/total_pixels:.6f}")
else:
    pos_weight = neg_weight = 1.0
    print("⚠️ Could not compute class weights, using balanced weights")

# ====== Enhanced Loss Functions ======
def focal_loss(alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance."""
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        
        # Compute focal weight
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        
        # Compute focal loss
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        focal = focal_weight * bce
        
        return tf.reduce_mean(focal)
    return focal_loss_fixed

def combined_loss(y_true, y_pred):
    """Combination of focal loss and dice loss."""
    focal = focal_loss(alpha=0.25, gamma=2.0)(y_true, y_pred)
    
    # Dice loss
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    dice_loss = 1 - dice
    
    return focal + dice_loss

# ====== Model with Proper Optimizer ======
model = build_resunet_a(input_shape=(256, 256, 9))

# Enhanced optimizer with learning rate scheduling
initial_lr = 1e-4
optimizer = tf.keras.optimizers.Adam(
    learning_rate=initial_lr,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)

model.compile(
    optimizer=optimizer,
    loss=combined_loss,
    metrics=[iou_score, dice_coef, 'binary_accuracy']
)

print(f"Model compiled with combined loss and {model.count_params():,} parameters")

# ====== Enhanced Callbacks ======
os.makedirs('outputs/checkpoints', exist_ok=True)
os.makedirs('outputs/logs', exist_ok=True)

# Model checkpointing
checkpoint_cb = ModelCheckpoint(
    'outputs/checkpoints/model_best.h5',
    monitor='val_iou_score',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Early stopping
early_stop_cb = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

# Learning rate reduction
lr_reduce_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-7,
    verbose=1
)

# Logging callbacks
tensorboard_cb = TensorBoard(
    log_dir='outputs/logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

csv_logger_cb = CSVLogger('outputs/logs/training_log.csv', append=True)
gpu_logger_cb = GPUMemoryLogger()
balance_logger_cb = ClassBalanceLogger(train_gen)

callbacks = [
    checkpoint_cb,
    early_stop_cb, 
    lr_reduce_cb,
    tensorboard_cb,
    csv_logger_cb,
    gpu_logger_cb,
    balance_logger_cb
]

# ====== Training with Validation ======
print("Starting training...")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,  # Increased epochs with early stopping
    callbacks=callbacks,
    verbose=1
)

# ====== Enhanced Visualization ======
def plot_training_history(history):
    """Enhanced training history plotting."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # IoU
    axes[0, 1].plot(history.history['iou_score'], label='Train IoU', linewidth=2)
    axes[0, 1].plot(history.history['val_iou_score'], label='Val IoU', linewidth=2)
    axes[0, 1].set_title('IoU Score', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Dice
    axes[0, 2].plot(history.history['dice_coef'], label='Train Dice', linewidth=2)
    axes[0, 2].plot(history.history['val_dice_coef'], label='Val Dice', linewidth=2)
    axes[0, 2].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Dice')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(history.history['binary_accuracy'], label='Train Accuracy', linewidth=2)
    axes[1, 0].plot(history.history['val_binary_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1, 0].set_title('Binary Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], linewidth=2, color='orange')
        axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Training Summary
    axes[1, 2].axis('off')
    best_val_iou = max(history.history['val_iou_score'])
    best_val_dice = max(history.history['val_dice_coef'])
    min_val_loss = min(history.history['val_loss'])
    
    summary_text = f"""
    Training Summary:
    
    Best Validation IoU: {best_val_iou:.4f}
    Best Validation Dice: {best_val_dice:.4f}
    Min Validation Loss: {min_val_loss:.4f}
    
    Total Epochs: {len(history.history['loss'])}
    Total Parameters: {model.count_params():,}
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, 
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig('outputs/logs/enhanced_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot results
plot_training_history(history)

print("✅ Training completed!")
print(f"📊 Training history saved to: outputs/logs/enhanced_training_curves.png")
print(f"💾 Best model saved to: outputs/checkpoints/model_best.h5")
