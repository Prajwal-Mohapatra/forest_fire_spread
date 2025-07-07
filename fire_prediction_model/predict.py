import rasterio
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Handle relative imports for both local and external usage
try:
    from .model.resunet_a import build_resunet_a
except ImportError:
    try:
        from model.resunet_a import build_resunet_a
    except ImportError:
        # Fallback for when imported from different directory structures
        import sys
        import os
        current_dir = os.path.dirname(__file__)
        model_dir = os.path.join(current_dir, 'model')
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from resunet_a import build_resunet_a

def predict_fire_map(tif_path, model_path, output_path, patch_size=256):
    model = build_resunet_a(input_shape=(patch_size, patch_size, 9))
    model.load_weights(model_path)

    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        profile = src.profile.copy()
        data = src.read(out_dtype='float32')  # shape: (10, H, W)
    
    data = np.moveaxis(data, 0, -1)  # (H, W, 10)
    input_data = data[:, :, :9] / 255.0

    output_mask = np.zeros((h, w), dtype='float32')

    for i in tqdm(range(0, h, patch_size)):
        for j in range(0, w, patch_size):
            patch = input_data[i:i+patch_size, j:j+patch_size, :]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue  # skip edge
            pred = model.predict(np.expand_dims(patch, 0))[0, :, :, 0]
            output_mask[i:i+patch_size, j:j+patch_size] = pred

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
