import numpy as np

def normalize_patch(patch):
    """
    Robust per-band normalization for a 9‑band input patch.
    Cleans NaNs/Infs and scales each band to [0, 1].
    Input shape: (H, W, 9)
    """
    # Prepare output array
    norm_patch = np.zeros_like(patch, dtype=np.float32)

    for b in range(patch.shape[-1]):
        band = patch[:, :, b].astype(np.float32)

        # 1) Clean up any NaN or infinite values
        band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) Compute min/max
        b_min = np.min(band)
        b_max = np.max(band)

        # 3) Safe normalization (avoid divide-by-zero)
        if b_max > b_min:
            norm_patch[:, :, b] = (band - b_min) / (b_max - b_min)
        else:
            # constant band or empty → fill with zeros
            norm_patch[:, :, b] = 0.0

    return norm_patch
