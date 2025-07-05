import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from model.resunet_a import build_resunet_a
from dataset.loader import FireDatasetGenerator
from utils.metrics import (
    iou_score, dice_coef, focal_loss, combined_loss, compute_class_weights,
    precision_score, recall_score, f1_score
)
from utils.versioning import ModelVersionManager
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
import yaml

# ====== Enhanced Training Configuration ======
def load_training_config(config_path="config.yaml"):
    """Load training configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # Default configuration if file not found
        return {
            'model': {
                'use_enhanced_aspp': False,
                'input_shape': [256, 256, 9],
                'num_classes': 1
            },
            'training': {
                'batch_size': 8,
                'epochs': 30,
                'learning_rate': 1e-4,
                'use_focal_loss': True,
                'use_combined_loss': False,
                'n_patches_per_img_train': 50,
                'n_patches_per_img_val': 20
            },
            'loss': {
                'focal_alpha': 0.25,
                'focal_gamma': 2.0,
                'dice_weight': 0.5,
                'focal_weight': 0.5
            }
        }

# Load configuration
config = load_training_config()
print(f"📋 Loaded training configuration")

# ====== Initialize Model Version Manager ======
version_manager = ModelVersionManager(base_dir="outputs")
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

# ====== Paths Configuration ======
# Update paths for local environment
base_dir = '/home/swayam/projects/forest_fire_spread/fire-probability-prediction-map-unstacked-data/dataset_stacked'
if not os.path.exists(base_dir):
    # Fallback to Kaggle paths if local data not found
    base_dir = '/kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked'

train_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_04_*.tif')))
val_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_05_0[1-7]*.tif')))
test_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_05_2[3-9]*.tif')))

print(f"Found {len(train_files)} training files")
print(f"Found {len(val_files)} validation files")
print(f"Found {len(test_files)} test files")

# ====== Data Generators ======
batch_size = config['training']['batch_size']
train_gen = FireDatasetGenerator(
    train_files, 
    batch_size=batch_size, 
    n_patches_per_img=config['training']['n_patches_per_img_train'], 
    shuffle=True
)
val_gen = FireDatasetGenerator(
    val_files, 
    batch_size=batch_size, 
    n_patches_per_img=config['training']['n_patches_per_img_val'], 
    shuffle=False
)

# ====== Compute Proper Class Weights ======
fire_weight, no_fire_weight = compute_class_weights(train_gen, num_samples=10)

# ====== Loss Function Selection ======
# Choose between focal loss, combined loss, or weighted BCE
USE_FOCAL_LOSS = config['training']['use_focal_loss']
USE_COMBINED_LOSS = config['training']['use_combined_loss']

if USE_COMBINED_LOSS:
    loss_fn = combined_loss(
        alpha=config['loss']['focal_alpha'], 
        gamma=config['loss']['focal_gamma'],
        dice_weight=config['loss']['dice_weight'],
        focal_weight=config['loss']['focal_weight']
    )
    loss_name = "Combined_Focal_Dice"
elif USE_FOCAL_LOSS:
    loss_fn = focal_loss(
        alpha=fire_weight/(fire_weight + no_fire_weight), 
        gamma=config['loss']['focal_gamma']
    )
    loss_name = "Focal_Loss"
else:
    # Fallback to weighted BCE
    def weighted_bce_fixed(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weights = y_true * fire_weight + (1 - y_true) * no_fire_weight
        return tf.reduce_mean(bce * weights)
    loss_fn = weighted_bce_fixed
    loss_name = "Weighted_BCE"

print(f"Using loss function: {loss_name}")

# ====== Enhanced Model with Configuration ======
model = build_resunet_a(
    input_shape=tuple(config['model']['input_shape']),
    num_classes=config['model']['num_classes'],
    use_enhanced_aspp=config['model']['use_enhanced_aspp']
)

# Enhanced metrics list
metrics_list = [iou_score, dice_coef, precision_score, recall_score, f1_score]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
    loss=loss_fn,
    metrics=metrics_list
)

print(f"Model compiled with {loss_name}")
print(f"Enhanced ASPP: {config['model']['use_enhanced_aspp']}")
print(f"Total parameters: {model.count_params():,}")
print(f"Metrics: {[m.__name__ for m in metrics_list]}")

# ====== Callbacks ======
os.makedirs('outputs/checkpoints', exist_ok=True)
os.makedirs('outputs/logs', exist_ok=True)

checkpoint_cb = ModelCheckpoint(
    'outputs/checkpoints/model_best.h5',
    monitor='val_iou_score',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1
)
tensorboard_cb = TensorBoard(log_dir='outputs/logs', histogram_freq=1)
csv_logger_cb = CSVLogger('outputs/logs/training_log.csv', append=True)
gpu_logger_cb = GPUMemoryLogger()
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint_cb, tensorboard_cb, csv_logger_cb, gpu_logger_cb, reduce_lr_cb]

# ====== Train with Enhanced Logging ======
epochs = config['training']['epochs']
print(f"\n🚀 Starting training for {epochs} epochs...")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=callbacks
)

# ====== Save Model Version with Metadata ======
training_metadata = {
    'config': config,
    'loss_function': loss_name,
    'optimizer': 'Adam',
    'learning_rate': config['training']['learning_rate'],
    'batch_size': batch_size,
    'epochs': epochs,
    'class_weights': {
        'fire_weight': float(fire_weight),
        'no_fire_weight': float(no_fire_weight)
    },
    'metrics': {
        'val_iou_score': history.history['val_iou_score'],
        'val_dice_coef': history.history['val_dice_coef'],
        'val_precision_score': history.history['val_precision_score'],
        'val_recall_score': history.history['val_recall_score'],
        'val_f1_score': history.history['val_f1_score'],
        'val_loss': history.history['val_loss']
    },
    'training_files_count': len(train_files),
    'validation_files_count': len(val_files)
}

# Save versioned model
version_id = version_manager.save_model_version(model, training_metadata)

# ====== Enhanced Plotting with All Metrics ======
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Training Results - {loss_name} - Version {version_id}', fontsize=16)

# Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_title(f'{loss_name} Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# IoU
axes[0, 1].plot(history.history['iou_score'], label='Train IoU', linewidth=2)
axes[0, 1].plot(history.history['val_iou_score'], label='Val IoU', linewidth=2)
axes[0, 1].set_title('IoU Score')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('IoU')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Dice
axes[0, 2].plot(history.history['dice_coef'], label='Train Dice', linewidth=2)
axes[0, 2].plot(history.history['val_dice_coef'], label='Val Dice', linewidth=2)
axes[0, 2].set_title('Dice Coefficient')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Dice')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision_score'], label='Train Precision', linewidth=2)
axes[1, 0].plot(history.history['val_precision_score'], label='Val Precision', linewidth=2)
axes[1, 0].set_title('Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall_score'], label='Train Recall', linewidth=2)
axes[1, 1].plot(history.history['val_recall_score'], label='Val Recall', linewidth=2)
axes[1, 1].set_title('Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# F1 Score
axes[1, 2].plot(history.history['f1_score'], label='Train F1', linewidth=2)
axes[1, 2].plot(history.history['val_f1_score'], label='Val F1', linewidth=2)
axes[1, 2].set_title('F1 Score')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('F1')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = f'outputs/logs/training_results_{version_id}.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

# ====== Final Results Summary ======
best_epoch = np.argmax(history.history['val_iou_score'])
print(f"\n🎯 Training completed!")
print(f"📊 Best epoch: {best_epoch + 1}")
print(f"📊 Best validation metrics (epoch {best_epoch + 1}):")
print(f"   📈 IoU: {history.history['val_iou_score'][best_epoch]:.4f}")
print(f"   � Dice: {history.history['val_dice_coef'][best_epoch]:.4f}")
print(f"   📈 Precision: {history.history['val_precision_score'][best_epoch]:.4f}")
print(f"   📈 Recall: {history.history['val_recall_score'][best_epoch]:.4f}")
print(f"   📈 F1: {history.history['val_f1_score'][best_epoch]:.4f}")
print(f"💾 Model version saved: {version_id}")
print(f"💾 Best weights saved: outputs/checkpoints/model_best.h5")
print(f"📊 Training plots saved: {plot_path}")

# Print version manager summary
version_manager.print_version_summary()
