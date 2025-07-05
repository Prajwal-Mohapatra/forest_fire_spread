import os
import glob
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

# ====== Paths ======
base_dir = '/kaggle/input/stacked-fire-probability-prediction-dataset/dataset_stacked'
all_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_*.tif')))

# Split by date:
train_files = [f for f in all_files if '_04_' in f]        # April
val_files   = [f for f in all_files if '_05_0[1-7]' in f]  # May 1–7
test_files  = [f for f in all_files if '_05_2[3-9]' in f]  # May 23–29

# ====== Data Generators ======
# use `n_patches_per_day` instead of per_img
train_gen = FireDatasetGenerator(
    tif_paths=train_files,
    patch_size=256,
    batch_size=8,
    n_patches_per_day=50,    # how many patches to sample from each consecutive day‐pair
    shuffle=True,
    augment_fn=None
)

val_gen = FireDatasetGenerator(
    tif_paths=val_files,
    patch_size=256,
    batch_size=8,
    n_patches_per_day=20,
    shuffle=False,
    augment_fn=None
)

# ====== Estimate class weights ======
print("Estimating fire/no-fire class balance...")
pos = neg = 0
# sample a few batches to get approximate ratio
for i in range(5):
    _, masks = train_gen[i]
    pos += np.sum(masks == 1)
    neg += np.sum(masks == 0)

w_pos = neg / (pos + neg + 1e-6)
w_neg = pos / (pos + neg + 1e-6)
print(f"✅ Class weights: Fire = {w_pos:.4f}, NoFire = {w_neg:.4f}")

# ====== Weighted BCE using fixed weights ======
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
    callbacks=callbacks,
    use_multiprocessing=False,
    workers=1
)

# ====== Plot Loss and Metrics ======
plt.figure(figsize=(15, 4))

# Loss
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss'); plt.legend()

# IoU
plt.subplot(1, 3, 2)
plt.plot(history.history['iou_score'], label='Train IoU')
plt.plot(history.history['val_iou_score'], label='Val IoU')
plt.title('IoU'); plt.legend()

# Dice
plt.subplot(1, 3, 3)
plt.plot(history.history['dice_coef'], label='Train Dice')
plt.plot(history.history['val_dice_coef'], label='Val Dice')
plt.title('Dice'); plt.legend()

plt.tight_layout()
plt.savefig('outputs/logs/metrics.png')
plt.show()
