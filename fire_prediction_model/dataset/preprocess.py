import numpy as np

def normalize_patch(patch):
    """
    Robust normalization for 9-band input patch.
    Replaces NaNs/Infs and avoids division by zero.
    """
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
    norm_patch = np.zeros_like(patch, dtype=np.float32)

    for b in range(patch.shape[-1]):
        band = patch[:, :, b]
        b_min = np.min(band)
        b_max = np.max(band)
        if b_max > b_min:
            norm_patch[:, :, b] = (band - b_min) / (b_max - b_min)
        else:
            norm_patch[:, :, b] = 0.0  # safe fallback

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
