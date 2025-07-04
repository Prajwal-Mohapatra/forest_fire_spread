import numpy as np

def normalize_patch(patch):
    """
    Robust normalization for a 9-band input patch.
    Cleans each band individually and scales to [0, 1].
    Assumes input patch shape = (H, W, 9)
    """
    norm_patch = np.zeros_like(patch, dtype=np.float32)

    for b in range(patch.shape[-1]):
        band = patch[:, :, b]

        # ✅ Clean per-band NaN, Inf
        band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)

        b_min = np.min(band)
        b_max = np.max(band)

        # ✅ Safe normalization with divide-by-zero protection
        if b_max > b_min:
            norm_patch[:, :, b] = (band - b_min) / (b_max - b_min)
        else:
            norm_patch[:, :, b] = 0.0  # fallback for constant band

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
