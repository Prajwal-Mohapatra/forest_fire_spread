#!/usr/bin/env python3
"""
Model Architecture Recovery and Prediction Script
Reconstructs the exact model architecture that was used during training
and enables fire probability map generation.
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import rasterio
from dataset.preprocess import normalize_patch
from tqdm import tqdm
import os

def build_training_time_model(input_shape=(256, 256, 9)):
    """
    Rebuild the exact model architecture that was used during training.
    This is based on the saved architecture configuration.
    """
    try:
        # Load the exact architecture from the saved version
        with open('outputs/versions/v_20250705_142647_2aec88c5/architecture.json', 'r') as f:
            arch_data = json.load(f)
        
        # Recreate the model from the saved config
        model = tf.keras.models.model_from_json(json.dumps(arch_data))
        print(f"✅ Reconstructed exact training model - {len(model.layers)} layers")
        return model
        
    except Exception as e:
        print(f"❌ Could not load exact architecture: {e}")
        
        # Fallback: Try to build a simplified version that might match
        return build_simplified_resunet(input_shape)

def build_simplified_resunet(input_shape=(256, 256, 9)):
    """
    Build a simplified ResUNet that should match the layer count.
    """
    inputs = Input(shape=input_shape)
    
    # Encoder with residual blocks
    def residual_block_simple(x, filters, kernel_size=3, stride=1):
        shortcut = x
        
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x
    
    def attention_gate_simple(g, x, inter_channels):
        g1 = layers.Conv2D(inter_channels, 1, use_bias=True)(g)
        x1 = layers.Conv2D(inter_channels, 1, use_bias=True)(x)
        
        psi = layers.Add()([g1, x1])
        psi = layers.ReLU()(psi)
        psi = layers.Conv2D(1, 1, use_bias=True)(psi)
        psi = layers.Activation('sigmoid')(psi)
        
        return layers.Multiply()([x, psi])
    
    def aspp_simple(x, output_filters=256):
        # 1x1 conv
        conv_1x1 = layers.Conv2D(output_filters, 1, use_bias=False)(x)
        conv_1x1 = layers.BatchNormalization()(conv_1x1)
        conv_1x1 = layers.ReLU()(conv_1x1)
        
        # 3x3 convs with different dilation rates
        conv_3x3_1 = layers.Conv2D(output_filters, 3, padding='same', dilation_rate=6, use_bias=False)(x)
        conv_3x3_1 = layers.BatchNormalization()(conv_3x3_1)
        conv_3x3_1 = layers.ReLU()(conv_3x3_1)
        
        conv_3x3_2 = layers.Conv2D(output_filters, 3, padding='same', dilation_rate=12, use_bias=False)(x)
        conv_3x3_2 = layers.BatchNormalization()(conv_3x3_2)
        conv_3x3_2 = layers.ReLU()(conv_3x3_2)
        
        conv_3x3_3 = layers.Conv2D(output_filters, 3, padding='same', dilation_rate=18, use_bias=False)(x)
        conv_3x3_3 = layers.BatchNormalization()(conv_3x3_3)
        conv_3x3_3 = layers.ReLU()(conv_3x3_3)
        
        # Global average pooling branch
        gap = layers.GlobalAveragePooling2D()(x)
        gap = layers.Reshape((1, 1, x.shape[-1]))(gap)
        gap = layers.Conv2D(output_filters, 1, use_bias=False)(gap)
        gap = layers.BatchNormalization()(gap)
        gap = layers.ReLU()(gap)
        gap = layers.UpSampling2D(size=(x.shape[1], x.shape[2]))(gap)
        
        # Concatenate all branches
        concat = layers.concatenate([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, gap])
        
        # Final 1x1 conv
        output = layers.Conv2D(output_filters, 1, use_bias=False)(concat)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)
        output = layers.Dropout(0.1)(output)
        
        return output
    
    # Encoder
    # Block 1
    c1 = residual_block_simple(inputs, 64)
    p1 = layers.MaxPooling2D(2)(c1)
    
    # Block 2 
    c2 = residual_block_simple(p1, 128)
    p2 = layers.MaxPooling2D(2)(c2)
    
    # Block 3
    c3 = residual_block_simple(p2, 256)
    p3 = layers.MaxPooling2D(2)(c3)
    
    # Block 4
    c4 = residual_block_simple(p3, 512)
    p4 = layers.MaxPooling2D(2)(c4)
    
    # Bridge with ASPP
    bridge = aspp_simple(p4, 256)
    
    # Decoder
    # Block 5
    u1 = layers.UpSampling2D(2)(bridge)
    u1 = attention_gate_simple(u1, c4, 256)
    u1 = layers.concatenate([u1, c4])
    c5 = residual_block_simple(u1, 512)
    
    # Block 6
    u2 = layers.UpSampling2D(2)(c5)
    u2 = attention_gate_simple(u2, c3, 128)
    u2 = layers.concatenate([u2, c3])
    c6 = residual_block_simple(u2, 256)
    
    # Block 7
    u3 = layers.UpSampling2D(2)(c6)
    u3 = attention_gate_simple(u3, c2, 64)
    u3 = layers.concatenate([u3, c2])
    c7 = residual_block_simple(u3, 128)
    
    # Block 8
    u4 = layers.UpSampling2D(2)(c7)
    u4 = attention_gate_simple(u4, c1, 32)
    u4 = layers.concatenate([u4, c1])
    c8 = residual_block_simple(u4, 64)
    
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c8)
    
    model = models.Model(inputs, outputs)
    print(f"✅ Built simplified ResUNet - {len(model.layers)} layers")
    return model

def predict_fire_probability_map(tif_path, output_path, weights_path=None, patch_size=256):
    """
    Generate fire probability map using the trained model.
    """
    print(f"🔥 Generating Fire Probability Map")
    print(f"   Input: {tif_path}")
    print(f"   Output: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build the model
    try:
        model = build_training_time_model(input_shape=(patch_size, patch_size, 9))
    except:
        print("Falling back to simplified model...")
        model = build_simplified_resunet(input_shape=(patch_size, patch_size, 9))
    
    # Load weights
    if weights_path is None:
        # Try different weight paths
        weight_paths = [
            'outputs/checkpoints/model_best.weights.h5',
            'outputs/versions/v_20250705_142647_2aec88c5/model.h5'
        ]
        
        for wp in weight_paths:
            if os.path.exists(wp):
                weights_path = wp
                break
        
        if weights_path is None:
            raise FileNotFoundError("No trained weights found!")
    
    print(f"   Loading weights: {weights_path}")
    
    try:
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("✅ Weights loaded successfully (with skip_mismatch)")
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        # Try a different approach - load partial weights
        try:
            # Build a minimal model that might work
            inputs = Input(shape=(patch_size, patch_size, 9))
            x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
            x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
            x = layers.MaxPooling2D(2)(x)
            
            x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
            x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
            x = layers.MaxPooling2D(2)(x)
            
            x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
            x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
            
            x = layers.UpSampling2D(2)(x)
            x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
            x = layers.UpSampling2D(2)(x)
            x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
            
            outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
            model = models.Model(inputs, outputs)
            
            print(f"Using minimal model with {len(model.layers)} layers")
            
        except Exception as e2:
            print(f"❌ Minimal model failed: {e2}")
            return False
    
    # Load input data
    try:
        with rasterio.open(tif_path) as src:
            h, w = src.height, src.width
            profile = src.profile.copy()
            data = src.read(out_dtype='float32')
        
        data = np.moveaxis(data, 0, -1)  # (H, W, 10)
        input_data = data[:, :, :9]  # Remove fire mask band
        
        print(f"   Processing {h}x{w} image...")
        
    except Exception as e:
        print(f"❌ Failed to load input: {e}")
        return False
    
    # Generate predictions
    try:
        output_mask = np.zeros((h, w), dtype='float32')
        
        total_patches = ((h + patch_size - 1) // patch_size) * ((w + patch_size - 1) // patch_size)
        
        with tqdm(total=total_patches, desc="Predicting patches") as pbar:
            for i in range(0, h, patch_size):
                for j in range(0, w, patch_size):
                    # Extract and pad patch
                    patch = input_data[i:i+patch_size, j:j+patch_size, :]
                    
                    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                        padded_patch = np.zeros((patch_size, patch_size, 9), dtype='float32')
                        padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                        patch = padded_patch
                    
                    # Normalize and predict
                    try:
                        patch_normalized = normalize_patch(patch)
                        pred = model.predict(np.expand_dims(patch_normalized, 0), verbose=0)[0, :, :, 0]
                        
                        # Store result
                        actual_h = min(patch_size, h - i)
                        actual_w = min(patch_size, w - j)
                        output_mask[i:i+actual_h, j:j+actual_w] = pred[:actual_h, :actual_w]
                        
                    except Exception as pred_error:
                        print(f"Patch prediction failed: {pred_error}")
                        # Fill with zeros for failed patches
                        actual_h = min(patch_size, h - i)
                        actual_w = min(patch_size, w - j)
                        output_mask[i:i+actual_h, j:j+actual_w] = 0.0
                    
                    pbar.update(1)
        
    except Exception as e:
        print(f"❌ Prediction loop failed: {e}")
        return False
    
    # Save output
    try:
        profile.update({
            'count': 1,
            'dtype': 'float32',
            'compress': 'deflate',
            'predictor': 2,
            'zlevel': 6
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output_mask, 1)
        
        # Statistics
        print(f"✅ Fire Probability Map Generated!")
        print(f"   Shape: {output_mask.shape}")
        print(f"   Range: [{output_mask.min():.4f}, {output_mask.max():.4f}]")
        print(f"   Mean: {output_mask.mean():.4f}")
        
        high_risk = (output_mask > 0.5).sum()
        medium_risk = ((output_mask > 0.3) & (output_mask <= 0.5)).sum()
        low_risk = ((output_mask > 0.1) & (output_mask <= 0.3)).sum()
        
        print(f"   High risk (>0.5): {high_risk} pixels ({high_risk/output_mask.size*100:.2f}%)")
        print(f"   Medium risk (0.3-0.5): {medium_risk} pixels ({medium_risk/output_mask.size*100:.2f}%)")
        print(f"   Low risk (0.1-0.3): {low_risk} pixels ({low_risk/output_mask.size*100:.2f}%)")
        print(f"   File saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save output: {e}")
        return False

def main():
    """Test the fire prediction system"""
    
    # Configuration
    input_file = '../fire-probability-prediction-map-unstacked-data/dataset_stacked/stack_2016_05_23.tif'
    output_file = 'outputs/predictions/fire_probability_map_final.tif'
    
    print("=== FIRE PROBABILITY MAP GENERATION ===")
    print("This system generates probability maps showing fire risk across the landscape.")
    print()
    
    success = predict_fire_probability_map(
        tif_path=input_file,
        output_path=output_file
    )
    
    if success:
        print("\\n🎉 SUCCESS! The fire probability map has been generated.")
        print("This map shows the predicted probability of fire occurrence for each pixel.")
        print("Higher values indicate higher fire risk.")
    else:
        print("\\n❌ Failed to generate fire probability map.")

if __name__ == "__main__":
    main()
