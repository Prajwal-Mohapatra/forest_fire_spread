import rasterio
import numpy as np
from model.resunet_a import build_resunet_a
from dataset.preprocess import normalize_patch
import tensorflow as tf
from tqdm import tqdm

def predict_fire_map(tif_path, model_path, output_path, patch_size=256):
    model = build_resunet_a(input_shape=(patch_size, patch_size, 9))
    model.load_weights(model_path)

    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        profile = src.profile.copy()
        data = src.read(out_dtype='float32')  # shape: (10, H, W)
    
    data = np.moveaxis(data, 0, -1)  # (H, W, 10)
    
    # Use consistent preprocessing with training
    input_data = data[:, :, :9]  # Remove fire mask band
    
    output_mask = np.zeros((h, w), dtype='float32')

    for i in tqdm(range(0, h, patch_size)):
        for j in range(0, w, patch_size):
            patch = input_data[i:i+patch_size, j:j+patch_size, :]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                # Pad patch to required size for edge cases
                padded_patch = np.zeros((patch_size, patch_size, 9), dtype='float32')
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded_patch
            
            # Apply same normalization as training
            patch_normalized = normalize_patch(patch)
            pred = model.predict(np.expand_dims(patch_normalized, 0), verbose=0)[0, :, :, 0]
            
            # Only use the actual patch size for output
            actual_h = min(patch_size, h - i)
            actual_w = min(patch_size, w - j)
            output_mask[i:i+actual_h, j:j+actual_w] = pred[:actual_h, :actual_w]

    profile.update({
        'count': 1,
        'dtype': 'float32',
        'compress': 'deflate',
        'predictor': 2,
        'zlevel': 6
    })

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(output_mask, 1)

    print(f"✅ Saved predicted fire map to: {output_path}")
