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
        print(f"✅ Dataset loaded successfully! {len(self.samples)} patches available.")

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
        X, Y = [], []
        i = idx * self.batch_size
        retries = 0
        # max_retries = 1000

        while len(X) < self.batch_size:
            tif_path, x, y = self.samples[i % len(self.samples)]
            i += 1
            retries += 1

            with rasterio.open(tif_path) as src:
                patch = src.read(window=rasterio.windows.Window(x, y, self.patch_size, self.patch_size),
                                boundless=True, fill_value=0).astype('float32')
            patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
            patch = np.moveaxis(patch, 0, -1)

            img = patch[:, :, :9]
            mask = (patch[:, :, 9] > 0).astype('float32')
            mask = np.expand_dims(mask, -1)

            if np.std(img) < 1e-5 or np.all(mask == 0):
                continue  # Skip bad patch

            if self.augment_fn:
                augmented = self.augment_fn(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']

            X.append(img)
            Y.append(mask)

        if len(X) == 0:
            raise RuntimeError(f"❌ Failed to collect valid patches for batch {idx}.")

        return np.array(X), np.array(Y)

