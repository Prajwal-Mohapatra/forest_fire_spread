import os
import glob
import re
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from model.resunet_a import build_resunet_a
from dataset.loader import FireDatasetGenerator
from utils.metrics import iou_score, dice_coef
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

# ====== GPU Memory Logger Callback ======
class GPUMemoryLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                mem = tf.config.experimental.get_memory_info('GPU:0')
                used = np.round(mem['current'] / 1024 / 1024)
                peak = np.round(mem['peak']    / 1024 / 1024)
                print(f"GPU Memory - Used: {used} MB / Peak: {peak} MB")
            except:
                pass

# ====== Paths & File Splits ======
base_dir  = '/kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked'
pattern   = os.path.join(base_dir, 'stack_2016_*.tif')
all_files = sorted(glob.glob(pattern))

# regex to extract month and day from filenames like stack_2016_05_03.tif
date_re = re.compile(r'stack_\d{4}_(\d{2})_(\d{2})\.tif$')

train_files, val_files, test_files = [], [], []
for f in all_files:
    m = date_re.search(os.path.basename(f))
    if not m:
        continue
    month, day = m.group(1), int(m.group(2))
    if month == '04':                      # April 1–30
        train_files.append(f)
    elif month == '05' and 1 <= day <= 7:  # May 1–7
        val_files.append(f)
    elif month == '05' and 23 <= day <= 29:# May 23–29
        test_files.append(f)

print(f"Train days: {len(train_files)} ⇒ Pairs: {max(0, len(train_files)-1)}")
print(f"Val   days: {len(val_files)} ⇒ Pairs: {max(0, len(val_files)-1)}")
print(f"Test  days: {len(test_files)} ⇒ Pairs: {max(0, len(test_files)-1)}")

# ====== Data Generators ======
train_gen = FireDatasetGenerator(
    tif_paths=train_files,
    patch_size=256,
    batch_size=8,
    n_patches_per_day=50,
    shuffle=True
)
val_gen = FireDatasetGenerator(
    tif_paths=val_files,
    patch_size=256,
    batch_size=8,
    n_patches_per_day=20,
    shuffle=False
)

# ====== Estimate class weights ======
print("Estimating fire/no-fire class balance...")
pos = neg = 0
for i in range(5):
    _, masks = train_gen[i]
    pos += np.sum(masks == 1)
    neg += np.sum(masks == 0)

w_pos = neg / (pos + neg + 1e-6)
w_neg = pos / (pos + neg + 1e-6)
print(f"✅ Class weights: Fire = {w_pos:.4f}, NoFire = {w_neg:.4f}")

def weighted_bce_fixed(y_true, y_pred):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    weights = y_true * w_pos + (1 - y_true) * w_neg
    return tf.reduce_mean(bce * weights)

# ====== Build & Compile Model ======
model = build_resunet_a(input_shape=(256, 256, 9))
model.compile(
    optimizer='adam',
    loss=weighted_bce_fixed,
    metrics=[iou_score, dice_coef]
)

# ====== Callbacks ======
os.makedirs('outputs/checkpoints', exist_ok=True)
os.makedirs('outputs/logs', exist_ok=True)

checkpoint_cb  = ModelCheckpoint('outputs/checkpoints/best_model.h5', save_best_only=True)
tensorboard_cb = TensorBoard(log_dir='outputs/logs')
csv_logger_cb  = CSVLogger('outputs/logs/training.csv', append=False)
gpu_logger_cb  = GPUMemoryLogger()

callbacks = [checkpoint_cb, tensorboard_cb, csv_logger_cb, gpu_logger_cb]

# ====== Train ======
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks
)

# ====== Plot Loss and Metrics ======
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'],     '-o', label='Train Loss')
plt.plot(history.history['val_loss'], '-o', label='Val Loss')
plt.title('Loss'); plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['iou_score'],     '-o', label='Train IoU')
plt.plot(history.history['val_iou_score'], '-o', label='Val IoU')
plt.title('IoU'); plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['dice_coef'],     '-o', label='Train Dice')
plt.plot(history.history['val_dice_coef'], '-o', label='Val Dice')
plt.title('Dice'); plt.legend()

plt.tight_layout()
plt.savefig('outputs/logs/metrics.png')
plt.show()
