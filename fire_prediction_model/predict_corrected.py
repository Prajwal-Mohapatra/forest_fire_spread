import rasterio
import numpy as np
from model.resunet_a import build_resunet_a
from dataset.preprocess import normalize_patch  # Use consistent normalization
import tensorflow as tf
from tqdm import tqdm

def predict_fire_map_corrected(tif_path, model_path, output_path, patch_size=256):
    """
    Corrected prediction function with consistent preprocessing.
    """
    model = build_resunet_a(input_shape=(patch_size, patch_size, 9))
    model.load_weights(model_path)

    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        profile = src.profile.copy()
        data = src.read(out_dtype='float32')  # shape: (10, H, W)
    
    data = np.moveaxis(data, 0, -1)  # (H, W, 10)
    
    # Use consistent normalization - CRITICAL FIX
    input_data = data[:, :, :9]  # Remove target band
    
    output_mask = np.zeros((h, w), dtype='float32')
    overlap = patch_size // 4  # Use overlapping patches for better results

    for i in tqdm(range(0, h, patch_size - overlap)):
        for j in range(0, w, patch_size - overlap):
            # Extract patch with proper bounds checking
            i_end = min(i + patch_size, h)
            j_end = min(j + patch_size, w)
            
            if i_end - i < patch_size or j_end - j < patch_size:
                # Pad the patch if needed
                patch = np.zeros((patch_size, patch_size, 9), dtype='float32')
                actual_patch = input_data[i:i_end, j:j_end, :]
                patch[:actual_patch.shape[0], :actual_patch.shape[1], :] = actual_patch
            else:
                patch = input_data[i:i_end, j:j_end, :]
            
            # Apply consistent normalization
            normalized_patch = normalize_patch(patch)
            
            # Predict
            pred = model.predict(np.expand_dims(normalized_patch, 0), verbose=0)[0, :, :, 0]
            
            # Handle overlapping regions by averaging
            actual_h = min(patch_size, h - i)
            actual_w = min(patch_size, w - j)
            
            if overlap > 0 and i > 0 and j > 0:
                # Average overlapping regions
                existing = output_mask[i:i+actual_h, j:j+actual_w]
                new_pred = pred[:actual_h, :actual_w]
                output_mask[i:i+actual_h, j:j+actual_w] = (existing + new_pred) / 2
            else:
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

def batch_predict_with_tta(model, patches, tta_transforms=None):
    """
    Prediction with Test Time Augmentation for better results.
    """
    if tta_transforms is None:
        tta_transforms = [
            lambda x: x,  # Original
            lambda x: np.fliplr(x),  # Horizontal flip
            lambda x: np.flipud(x),  # Vertical flip
            lambda x: np.rot90(x, k=1),  # 90° rotation
        ]
    
    predictions = []
    
    for transform in tta_transforms:
        # Apply transformation
        transformed_patches = np.array([transform(patch) for patch in patches])
        
        # Predict
        pred = model.predict(transformed_patches, verbose=0)
        
        # Reverse transformation
        if transform == np.fliplr:
            pred = np.array([np.fliplr(p) for p in pred])
        elif transform == np.flipud:
            pred = np.array([np.flipud(p) for p in pred])
        elif transform == lambda x: np.rot90(x, k=1):
            pred = np.array([np.rot90(p, k=-1) for p in pred])
        
        predictions.append(pred)
    
    # Average all predictions
    return np.mean(predictions, axis=0)
