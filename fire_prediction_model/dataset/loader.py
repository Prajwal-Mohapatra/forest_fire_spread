import os
import numpy as np
import rasterio
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle
import random
from dataset.preprocess import normalize_patch  # Imported robust normalization

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
        batch_samples = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, Y = [], []
        for tif_path, x, y in batch_samples:
            with rasterio.open(tif_path) as src:
                window = rasterio.windows.Window(x, y, self.patch_size, self.patch_size)

                # Safe read with boundary handling + fill for missing values
                patch = src.read(window=window, boundless=True, fill_value=0).astype('float32')

                # Remove NaN/inf before any processing
                patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

                patch = np.moveaxis(patch, 0, -1)  # (H, W, C)

            print("Patch shape:", patch.shape)
            print("Raw patch min/max:", np.nanmin(patch), np.nanmax(patch))
            print("Any NaN in patch?", np.isnan(patch).any())


            # Normalize image bands (first 9)
            img = normalize_patch(patch[:, :, :9])

            # Prepare binary fire mask from 10th band
            mask = (patch[:, :, 9] > 0).astype('float32')
            mask = np.expand_dims(mask, axis=-1)

            # Optional augmentation
            if self.augment_fn:
                augmented = self.augment_fn(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']

            X.append(img)
            Y.append(mask)

        return np.array(X), np.array(Y)
