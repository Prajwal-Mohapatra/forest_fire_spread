import os
import numpy as np
import rasterio
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle
import random
from dataset.preprocess import normalize_patch

class FireDatasetGenerator(Sequence):
    def __init__(self, tif_paths, patch_size=256, batch_size=8, n_patches_per_img=50,
                 shuffle=True, augment_fn=None, fire_sample_ratio=0.6, **kwargs):
        """
        Improved Fire Dataset Generator with better class balance handling.
        
        Args:
            fire_sample_ratio: Ratio of patches that should contain fire pixels (0.0-1.0)
        """
        super().__init__(**kwargs)

        self.tif_paths = tif_paths
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.n_patches_per_img = n_patches_per_img
        self.shuffle = shuffle
        self.augment_fn = augment_fn
        self.fire_sample_ratio = fire_sample_ratio
        self.samples = self._generate_patch_coords()
        print(f"✅ Dataset loaded successfully! {len(self.samples)} patches available.")

    def _generate_patch_coords(self):
        """Generate both fire and non-fire patch coordinates."""
        fire_samples = []
        regular_samples = []
        
        for tif in self.tif_paths:
            with rasterio.open(tif) as src:
                h, w = src.height, src.width
                
                # Sample fire patches by checking fire mask
                fire_data = src.read(10)  # Fire mask is band 10
                fire_indices = np.where(fire_data > 0)
                
                if len(fire_indices[0]) > 0:
                    # Generate fire-containing patches
                    n_fire_patches = int(self.n_patches_per_img * self.fire_sample_ratio)
                    for _ in range(n_fire_patches):
                        # Sample around fire locations
                        fire_idx = random.randint(0, len(fire_indices[0]) - 1)
                        fire_y, fire_x = fire_indices[0][fire_idx], fire_indices[1][fire_idx]
                        
                        # Add some randomness around fire location
                        offset_x = random.randint(-self.patch_size//2, self.patch_size//2)
                        offset_y = random.randint(-self.patch_size//2, self.patch_size//2)
                        
                        x = max(0, min(w - self.patch_size, fire_x + offset_x))
                        y = max(0, min(h - self.patch_size, fire_y + offset_y))
                        
                        fire_samples.append((tif, x, y, True))  # Mark as fire sample
                
                # Generate regular random patches
                n_regular_patches = self.n_patches_per_img - len([s for s in fire_samples if s[0] == tif])
                for _ in range(n_regular_patches):
                    x = random.randint(0, w - self.patch_size)
                    y = random.randint(0, h - self.patch_size)
                    regular_samples.append((tif, x, y, False))  # Mark as regular sample
        
        # Combine and shuffle
        all_samples = fire_samples + regular_samples
        return shuffle(all_samples) if self.shuffle else all_samples

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, idx):
        X, Y = [], []
        i = idx * self.batch_size
        retries = 0
        max_retries = 200

        while len(X) < self.batch_size and retries < max_retries:
            sample_info = self.samples[i % len(self.samples)]
            tif_path, x, y = sample_info[:3]
            i += 1
            retries += 1

            try:
                with rasterio.open(tif_path) as src:
                    patch = src.read(
                        window=rasterio.windows.Window(x, y, self.patch_size, self.patch_size),
                        boundless=True, fill_value=0
                    ).astype('float32')
            except Exception as e:
                print(f"⚠️ Skipping invalid patch from {tif_path}: {e}")
                continue

            patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
            patch = np.moveaxis(patch, 0, -1)  # (H, W, C)

            img = normalize_patch(patch[:, :, :9])
            mask = (patch[:, :, 9] > 0).astype('float32')
            mask = np.expand_dims(mask, -1)

            # Quality check - skip patches with very low variance
            if np.std(img) < 1e-5:
                continue

            if self.augment_fn:
                augmented = self.augment_fn(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']

            X.append(img)
            Y.append(mask)

        # Fallback if still no valid patches
        if len(X) == 0:
            print(f"⚠️ No valid patches in batch {idx}, using fallback sampling...")
            while len(X) < self.batch_size:
                tif_path, x, y = random.choice(self.samples)[:3]
                with rasterio.open(tif_path) as src:
                    patch = src.read(
                        window=rasterio.windows.Window(x, y, self.patch_size, self.patch_size),
                        boundless=True, fill_value=0
                    ).astype('float32')
                patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
                patch = np.moveaxis(patch, 0, -1)

                img = normalize_patch(patch[:, :, :9])
                mask = (patch[:, :, 9] > 0).astype('float32')
                mask = np.expand_dims(mask, -1)

                X.append(img)
                Y.append(mask)

        return np.array(X), np.array(Y)
