import os
import numpy as np
import rasterio
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle
import random

class FireDatasetGenerator(Sequence):
    def __init__(self, tif_paths, patch_size=256, batch_size=8, n_patches_per_img=50,
                 shuffle=True, augment_fn=None):
        self.tif_paths = tif_paths
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.n_patches_per_img = n_patches_per_img
        self.shuffle = shuffle
        self.augment_fn = augment_fn
        self.samples = self._generate_patch_coords()
    
    def _generate_patch_coords(self):
        all_samples = []
        for tif in self.tif_paths:
            with rasterio.open(tif) as src:
                h, w = src.height, src.width
            for _ in range(self.n_patches_per_img):
                x = random.randint(0, w - self.patch_size)
                y = random.randint(0, h - self.patch_size)
                all_samples.append((tif, x, y))
        return shuffle(all_samples) if self.shuffle else all_samples

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, idx):
        batch_samples = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, Y = [], []
        for tif_path, x, y in batch_samples:
            with rasterio.open(tif_path) as src:
                patch = src.read(window=rasterio.windows.Window(x, y, self.patch_size, self.patch_size))
                patch = np.moveaxis(patch, 0, -1)  # (H, W, C)
            img = patch[:, :, :9].astype('float32') / 255.0
            mask = (patch[:, :, 9] > 0).astype('float32')  # Fire mask
            mask = np.expand_dims(mask, axis=-1)
            if self.augment_fn:
                augmented = self.augment_fn(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']
            X.append(img)
            Y.append(mask)
        return np.array(X), np.array(Y)
