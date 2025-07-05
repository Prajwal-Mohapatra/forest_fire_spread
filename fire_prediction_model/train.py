import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from model.resunet_a import build_resunet_a
from dataset.loader import FireDatasetGenerator
from utils.metrics import iou_score, dice_coef, focal_loss, combined_loss, compute_class_weights
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau

# ====== GPU Memory Logger Callback ======
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
base_dir = './fire-probability-prediction-map-unstacked-data/dataset_stacked'
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
train_gen = FireDatasetGenerator(train_files, batch_size=8, n_patches_per_img=50, shuffle=True)
val_gen = FireDatasetGenerator(val_files, batch_size=8, n_patches_per_img=20, shuffle=False)

# ====== Compute Proper Class Weights ======
fire_weight, no_fire_weight = compute_class_weights(train_gen, num_samples=10)

# ====== Loss Function Selection ======
# Choose between focal loss, combined loss, or weighted BCE
USE_FOCAL_LOSS = True
USE_COMBINED_LOSS = False

if USE_COMBINED_LOSS:
    loss_fn = combined_loss(alpha=fire_weight/(fire_weight + no_fire_weight), gamma=2.0)
    loss_name = "Combined_Focal_Dice"
elif USE_FOCAL_LOSS:
    loss_fn = focal_loss(alpha=fire_weight/(fire_weight + no_fire_weight), gamma=2.0)
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

# ====== Model ======
model = build_resunet_a(input_shape=(256, 256, 9))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss_fn,
    metrics=[iou_score, dice_coef]
)

print(f"Model compiled with {loss_name}")
print(f"Total parameters: {model.count_params():,}")

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

# ====== Train ======
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks
)

# ====== Plot Loss and Metrics ======
plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title(f'{loss_name} Loss')
plt.legend()

# IoU
plt.subplot(1, 3, 2)
plt.plot(history.history['iou_score'], label='Train IoU')
plt.plot(history.history['val_iou_score'], label='Val IoU')
plt.title('IoU Score')
plt.legend()

# Dice
plt.subplot(1, 3, 3)
plt.plot(history.history['dice_coef'], label='Train Dice')
plt.plot(history.history['val_dice_coef'], label='Val Dice')
plt.title('Dice Coefficient')
plt.legend()

plt.tight_layout()
plt.savefig('outputs/logs/loss_metric_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n🎯 Training completed!")
print(f"📊 Best validation IoU: {max(history.history['val_iou_score']):.4f}")
print(f"📊 Best validation Dice: {max(history.history['val_dice_coef']):.4f}")
print(f"💾 Model saved to: outputs/checkpoints/model_best.h5")
