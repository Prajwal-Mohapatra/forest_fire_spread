import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from model.resunet_a import build_resunet_a
from dataset.loader import FireDatasetGenerator
from utils.metrics import iou_score, dice_coef
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

# ====== Paths ======
base_dir = '/content/fire-probability-prediction-map-u.../dataset_unstacked/dataset_stacked'
train_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_04_*.tif')))
val_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_05_0[1-7]*.tif')))
test_files = sorted(glob.glob(os.path.join(base_dir, 'stack_2016_05_2[3-9]*.tif')))

# ====== Model ======
model = build_resunet_a(input_shape=(256, 256, 9))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[iou_score, dice_coef])

# ====== Data ======
train_gen = FireDatasetGenerator(train_files, batch_size=8, n_patches_per_img=60)
val_gen = FireDatasetGenerator(val_files, batch_size=8, n_patches_per_img=20)

# ====== Callbacks ======
os.makedirs('outputs/checkpoints', exist_ok=True)
os.makedirs('outputs/logs', exist_ok=True)

checkpoint_cb = ModelCheckpoint('outputs/checkpoints/model_best.h5', save_best_only=True)
tensorboard_cb = TensorBoard(log_dir='outputs/logs')
csv_logger_cb = CSVLogger('outputs/logs/training_log.csv', append=True)

# ====== Train ======
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[checkpoint_cb, tensorboard_cb, csv_logger_cb]
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
