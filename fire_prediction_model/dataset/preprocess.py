import numpy as np

def normalize_patch(patch):
    """
    Normalizes input 9-band patch (excluding label).
    - Assumes patch shape = (H, W, 9)
    - Normalize each band to [0, 1] range
    """
    patch = np.clip(patch, 0, 1)  # assuming values are already scaled
    return patch

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
