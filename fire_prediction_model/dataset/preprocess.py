import numpy as np

def normalize_patch(patch):
    """
    Robust per-band normalization for a 9‑band input patch.
    Cleans NaNs/Infs and scales each band to [0, 1].
    Input shape: (H, W, 9)
    """
    norm_patch = np.zeros_like(patch, dtype=np.float32)
    for b in range(patch.shape[-1]):
        band = patch[:, :, b].astype(np.float32)
        band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
        b_min = np.min(band)
        b_max = np.max(band)
        if b_max > b_min:
            norm_patch[:, :, b] = (band - b_min) / (b_max - b_min)
        else:
            norm_patch[:, :, b] = 0.0
    return norm_patch

def compute_class_weight(mask_batch):
    """
    Computes pixel-wise class weights for fire/no-fire imbalance.
    Returns a (batch_size, H, W, 1) array of per-pixel weights.
    """
    weights = []
    for mask in mask_batch:
        pos = np.sum(mask == 1)
        neg = np.sum(mask == 0)
        total = pos + neg
        if total == 0:
            weights.append(np.ones_like(mask))  # fallback
        else:
            fire_weight = total / (2. * pos + 1e-6)
            nofire_weight = total / (2. * neg + 1e-6)
            w = np.where(mask == 1, fire_weight, nofire_weight)
            weights.append(w)
    return np.stack(weights, axis=0).astype('float32')
