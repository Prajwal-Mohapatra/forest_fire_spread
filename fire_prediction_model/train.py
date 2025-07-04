import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from model.resunet_a import build_resunet_a
from dataset.loader import FireDatasetGenerator
from dataset.preprocess import compute_class_weight
from utils.metrics import iou_score, dice_coef
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

# ====== Weighted Binary Cross-Entropy ======
def weighted_bce(y_true, y_pred):
    weights = compute_class_weight(y_true)
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * weights)

# ====== GPU Memory Logger Callback ======
class GPUMemoryLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                mem = tf.config.experimental.get_memory_info('GPU:0')
                used = np.round(mem['current'] / 1024 / 1024)
                total = np.round(mem['peak'] / 1024 / 1024)
                print(f"🔋 GPU Memory - Used: {used} MB / Total Peak: {total} MB")
            except:
                pass

# ====== Paths ======
base_dir = '/home/swayam/projects/forest_fire_spread/fire-probability-prediction-map-unstacked-data/dataset_stacked'
train_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_04_*.tif')))
val_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_05_0[1-7]*.tif')))
test_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_05_2[3-9]*.tif')))

# ====== Model ======
model = build_resunet_a(input_shape=(256, 256, 9))
model.compile(optimizer='adam', loss=weighted_bce, metrics=[iou_score, dice_coef])

# ====== Data ======
train_gen = FireDatasetGenerator(train_files, batch_size=8, n_patches_per_img=60)
val_gen = FireDatasetGenerator(val_files, batch_size=8, n_patches_per_img=20)

# ====== Callbacks ======
os.makedirs('outputs/checkpoints', exist_ok=True)
os.makedirs('outputs/logs', exist_ok=True)

checkpoint_cb = ModelCheckpoint('outputs/checkpoints/model_best.h5', save_best_only=True)
tensorboard_cb = TensorBoard(log_dir='outputs/logs')
csv_logger_cb = CSVLogger('outputs/logs/training_log.csv', append=True)
gpu_logger_cb = GPUMemoryLogger()

callbacks = [checkpoint_cb, tensorboard_cb, csv_logger_cb, gpu_logger_cb]

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
plt.title('Loss')
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
plt.savefig('outputs/logs/loss_metric_curves.png')
plt.show()
