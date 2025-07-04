import numpy as np

def normalize_patch(patch):
    """
    Normalizes input 9-band patch (excluding label).
    - Input shape: (H, W, 9)
    - Output shape: same, with each band normalized to [0, 1]
    """
    patch = np.nan_to_num(patch)  # Replace NaNs/Infs with 0
    norm_patch = np.zeros_like(patch)

    for b in range(patch.shape[-1]):
        band = patch[:, :, b]
        band_min = np.nanmin(band)
        band_max = np.nanmax(band)
        if band_max > band_min:
            norm_patch[:, :, b] = (band - band_min) / (band_max - band_min)
        else:
            norm_patch[:, :, b] = 0  # or 0.5 if you want neutral input

    return norm_patch


def compute_class_weight(mask_batch):
    """
    Dynamically compute foreground/background weights
    from batch of masks (batch_size, H, W, 1)
    Returns a per-sample weight map
    """
    weights = []
    for mask in mask_batch:
        pos = np.sum(mask == 1)
        neg = np.sum(mask == 0)
        total = pos + neg
        w = np.where(mask == 1, total / (2. * pos + 1e-6), total / (2. * neg + 1e-6))
        weights.append(w)
    return np.stack(weights, axis=0)
