import os
import re
import numpy as np
import rasterio
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle
import random
from dataset.preprocess import normalize_patch

class FireDatasetGenerator(Sequence):
    def __init__(self, tif_paths, patch_size=256, batch_size=8,
                 n_patches_per_day=50, shuffle=True, augment_fn=None):
        """
        tif_paths: list of paths like '.../stack_YYYY_MM_DD.tif'
        n_patches_per_day: how many patches to sample per day-pair
        """
        # 1) Sort by date extracted from filename
        date_pattern = re.compile(r'(\d{4}_\d{2}_\d{2})')
        self.tif_paths = sorted(
            tif_paths,
            key=lambda p: date_pattern.search(p).group(1)
        )
        # 2) Build list of (day_t, day_t+1) pairs
        self.pairs = [
            (self.tif_paths[i], self.tif_paths[i+1])
            for i in range(len(self.tif_paths)-1)
        ]
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.n_patches_per_day = n_patches_per_day
        self.shuffle = shuffle
        self.augment_fn = augment_fn

        # 3) Pre-generate sample coords for each day-pair
        self.samples = self._generate_patch_coords()
        print(f"✅ Dataset loaded: {len(self.pairs)} day-pairs, "
              f"{len(self.samples)} total patch-locations.")

    def _generate_patch_coords(self):
        all_samples = []
        for day_t, day_t1 in self.pairs:
            with rasterio.open(day_t) as src:
                h, w = src.height, src.width
            # sample n coords per pair
            for _ in range(self.n_patches_per_day):
                x = random.randint(0, w - self.patch_size)
                y = random.randint(0, h - self.patch_size)
                all_samples.append((day_t, day_t1, x, y))
        return shuffle(all_samples) if self.shuffle else all_samples

    def __len__(self):
        # total samples divided by batch_size
        return len(self.samples) // self.batch_size

    def __getitem__(self, idx):
        X, Y = [], []
        base = idx * self.batch_size
        retries = 0
        max_retries = 200

        while len(X) < self.batch_size and retries < max_retries:
            day_t, day_t1, x, y = self.samples[(base + retries) % len(self.samples)]
            retries += 1

            # read Day t bands 1–9
            try:
                with rasterio.open(day_t) as src_t:
                    patch_t = src_t.read(
                        window=rasterio.windows.Window(x, y, self.patch_size, self.patch_size),
                        boundless=True, fill_value=0
                    ).astype('float32')
                with rasterio.open(day_t1) as src_t1:
                    patch_t1 = src_t1.read(
                        window=rasterio.windows.Window(x, y, self.patch_size, self.patch_size),
                        boundless=True, fill_value=0
                    ).astype('float32')
            except Exception:
                continue  # skip invalid

            # clean & reorder [C,H,W] → [H,W,C]
            patch_t  = np.nan_to_num(patch_t,  nan=0., posinf=0., neginf=0.)
            patch_t1 = np.nan_to_num(patch_t1, nan=0., posinf=0., neginf=0.)
            patch_t   = np.moveaxis(patch_t,  0, -1)
            patch_t1  = np.moveaxis(patch_t1, 0, -1)

            # normalize Day t features
            img = normalize_patch(patch_t[:, :, :9])

            # Day t+1 fire mask
            mask = (patch_t1[:, :, 9] > 0).astype('float32')
            mask = np.expand_dims(mask, -1)

            # bias toward fire: require at least 2 fire pixels
            if np.sum(mask) < 2:
                continue

            # optional augmentation
            if self.augment_fn:
                aug = self.augment_fn(image=img, mask=mask)
                img, mask = aug['image'], aug['mask']

            X.append(img)
            Y.append(mask)

        # fallback: if no fire-containing patches found
        if len(X) == 0:
            print(f"⚠️ Batch {idx} had no fire; falling back to random day-pair sampling.")
            while len(X) < self.batch_size:
                day_t, day_t1, x, y = random.choice(self.samples)
                # (read & process exactly as above)
                with rasterio.open(day_t) as src_t:
                    patch_t = np.nan_to_num(
                        src_t.read(window=rasterio.windows.Window(x, y, self.patch_size, self.patch_size),
                                   boundless=True, fill_value=0).astype('float32'),
                        nan=0., posinf=0., neginf=0.
                    )
                with rasterio.open(day_t1) as src_t1:
                    patch_t1 = np.nan_to_num(
                        src_t1.read(window=rasterio.windows.Window(x, y, self.patch_size, self.patch_size),
                                    boundless=True, fill_value=0).astype('float32'),
                        nan=0., posinf=0., neginf=0.
                    )
                patch_t  = np.moveaxis(patch_t,  0, -1)
                patch_t1 = np.moveaxis(patch_t1, 0, -1)
                img  = normalize_patch(patch_t[:, :, :9])
                mask = (patch_t1[:, :, 9] > 0).astype('float32')
                mask = np.expand_dims(mask, -1)
                X.append(img); Y.append(mask)

        return np.stack(X,  axis=0), np.stack(Y, axis=0)
