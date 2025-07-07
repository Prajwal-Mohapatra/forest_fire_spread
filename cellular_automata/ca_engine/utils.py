# ====================
# CA Engine Utilities
# ====================
"""
Utility functions for the cellular automata fire spread simulation.
Handles coordinate transformations, data loading, and geometric calculations.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import tensorflow as tf
from typing import Tuple, List, Dict, Optional, Union
import os
from datetime import datetime, timedelta
import json

# Global variable to track GPU setup status
_gpu_configured = False

def setup_tensorflow_gpu():
    """Configure TensorFlow for optimal GPU usage."""
    global _gpu_configured
    
    # Check if already configured
    if _gpu_configured:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU already configured: {len(gpus)} device(s) available")
            return True
        else:
            print("⚠️ No GPU found, using CPU")
            return False
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configured: {len(gpus)} device(s) available")
            _gpu_configured = True
            return True
        except RuntimeError as e:
            print(f"⚠️ GPU setup failed: {e}")
            _gpu_configured = True  # Mark as attempted to avoid repeated failures
            return False
    else:
        print("⚠️ No GPU found, using CPU")
        _gpu_configured = True
        return False

def load_probability_map(file_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load ML-generated fire probability map from .tif file.
    
    Args:
        file_path: Path to probability map .tif file
        
    Returns:
        probability_array: 2D numpy array with fire probabilities (0-1)
        metadata: Dictionary with geospatial metadata
    """
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)
            
            # Handle edge cases
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
            data = np.clip(data, 0.0, 1.0)
            
            metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'bounds': src.bounds,
                'shape': data.shape,
                'resolution': (src.transform.a, abs(src.transform.e))
            }
            
        print(f"✅ Loaded probability map: {data.shape}, range: [{data.min():.3f}, {data.max():.3f}]")
        return data, metadata
        
    except Exception as e:
        print(f"❌ Failed to load probability map {file_path}: {e}")
        raise e

def load_environmental_layers(metadata: Dict, config) -> Dict[str, np.ndarray]:
    """
    Load environmental layers (DEM, LULC, GHSL) matching the probability map extent.
    
    Args:
        metadata: Geospatial metadata from probability map
        config: Configuration module
        
    Returns:
        Dictionary of environmental arrays
    """
    env_layers = {}
    
    # For now, create synthetic environmental layers
    # In full implementation, these would be loaded from actual data files
    shape = metadata['shape']
    
    # Synthetic DEM (elevation/slope)
    env_layers['elevation'] = np.random.normal(1000, 500, shape).astype(np.float32)
    env_layers['slope'] = np.random.uniform(0, 45, shape).astype(np.float32)
    
    # Synthetic LULC (fuel load)
    env_layers['fuel_load'] = np.random.uniform(0.2, 1.0, shape).astype(np.float32)
    
    # Synthetic GHSL (barriers)
    env_layers['barriers'] = (np.random.random(shape) < 0.05).astype(np.float32)
    
    print(f"✅ Generated synthetic environmental layers for shape {shape}")
    return env_layers

def pixel_to_geo(row: int, col: int, transform) -> Tuple[float, float]:
    """Convert pixel coordinates to geographic coordinates."""
    x, y = rasterio.transform.xy(transform, row, col)
    return x, y

def geo_to_pixel(x: float, y: float, transform) -> Tuple[int, int]:
    """Convert geographic coordinates to pixel coordinates."""
    row, col = rasterio.transform.rowcol(transform, x, y)
    return row, col

def calculate_wind_influence(wind_direction: float, wind_speed: float, 
                           dx: int, dy: int) -> float:
    """
    Calculate wind influence on fire spread direction.
    
    Args:
        wind_direction: Wind direction in degrees (meteorological convention)
        wind_speed: Wind speed in km/h
        dx, dy: Direction vector from source to target cell
        
    Returns:
        Wind influence multiplier (0.1 to 3.0)
    """
    # Convert wind direction to radians
    wind_rad = np.radians(wind_direction)
    wind_vector = np.array([np.sin(wind_rad), np.cos(wind_rad)])
    
    # Direction vector (normalized)
    if dx == 0 and dy == 0:
        return 1.0
    
    direction_vector = np.array([dx, dy])
    direction_norm = np.linalg.norm(direction_vector)
    if direction_norm == 0:
        return 1.0
    
    direction_vector = direction_vector / direction_norm
    
    # Calculate dot product (alignment with wind)
    alignment = np.dot(wind_vector, direction_vector)
    
    # Convert to influence multiplier
    base_influence = 1.0
    wind_factor = 1.0 + (wind_speed / 50.0) * alignment  # Max 50 km/h wind
    
    return np.clip(wind_factor, 0.1, 3.0)

def calculate_slope_influence(elevation_grid: np.ndarray, 
                            from_row: int, from_col: int,
                            to_row: int, to_col: int) -> float:
    """
    Calculate slope influence on fire spread.
    Fire spreads faster uphill.
    
    Args:
        elevation_grid: 2D elevation array
        from_row, from_col: Source cell coordinates
        to_row, to_col: Target cell coordinates
        
    Returns:
        Slope influence multiplier
    """
    if (from_row < 0 or from_row >= elevation_grid.shape[0] or
        from_col < 0 or from_col >= elevation_grid.shape[1] or
        to_row < 0 or to_row >= elevation_grid.shape[0] or
        to_col < 0 or to_col >= elevation_grid.shape[1]):
        return 1.0
    
    from_elevation = elevation_grid[from_row, from_col]
    to_elevation = elevation_grid[to_row, to_col]
    
    # Calculate slope angle
    distance = np.sqrt((to_row - from_row)**2 + (to_col - from_col)**2) * 30  # 30m resolution
    if distance == 0:
        return 1.0
    
    elevation_diff = to_elevation - from_elevation
    slope_angle = np.arctan(elevation_diff / distance)
    
    # Uphill increases spread rate, downhill decreases it
    slope_factor = 1.0 + 0.5 * np.sin(slope_angle)  # Range: 0.5 to 1.5
    
    return np.clip(slope_factor, 0.5, 2.0)

def create_ignition_points(grid_shape: Tuple[int, int], 
                          ignition_coords: List[Tuple[float, float]],
                          transform) -> np.ndarray:
    """
    Create ignition mask from geographic coordinates.
    
    Args:
        grid_shape: Shape of the simulation grid
        ignition_coords: List of (longitude, latitude) tuples
        transform: Rasterio transform object
        
    Returns:
        Binary ignition mask
    """
    ignition_mask = np.zeros(grid_shape, dtype=np.float32)
    
    for lon, lat in ignition_coords:
        try:
            # Validate coordinate ranges (approximate bounds for Uttarakhand)
            if not (77.0 <= lon <= 82.0 and 28.0 <= lat <= 32.0):
                print(f"⚠️ Ignition point ({lon:.4f}, {lat:.4f}) outside expected geographic bounds (Uttarakhand region)")
                continue
                
            row, col = geo_to_pixel(lon, lat, transform)
            
            # Validate pixel bounds
            if 0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]:
                ignition_mask[row, col] = 1.0
                print(f"✅ Ignition point set at pixel ({row}, {col}) for coords ({lon:.4f}, {lat:.4f})")
            else:
                print(f"⚠️ Ignition point ({lon:.4f}, {lat:.4f}) maps to pixel ({row}, {col}) outside grid bounds {grid_shape}")
        except Exception as e:
            print(f"⚠️ Failed to convert ignition point ({lon}, {lat}): {e}")
    
    return ignition_mask

def save_simulation_frame(fire_state: np.ndarray, 
                         hour: int, 
                         metadata: Dict, 
                         output_dir: str,
                         scenario_id: str) -> str:
    """
    Save simulation frame as GeoTIFF.
    
    Args:
        fire_state: Current fire state grid
        hour: Simulation hour
        metadata: Geospatial metadata
        output_dir: Output directory
        scenario_id: Unique scenario identifier
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"fire_spread_{scenario_id}_hour_{hour:02d}.tif"
    filepath = os.path.join(output_dir, filename)
    
    profile = {
        'driver': 'GTiff',
        'dtype': rasterio.float32,
        'nodata': 0,
        'width': fire_state.shape[1],
        'height': fire_state.shape[0],
        'count': 1,
        'crs': metadata['crs'],
        'transform': metadata['transform']
    }
    
    try:
        with rasterio.open(filepath, 'w', **profile) as dst:
            dst.write(fire_state, 1)
        return filepath
    except Exception as e:
        print(f"❌ Failed to save frame {filepath}: {e}")
        raise e

def create_scenario_metadata(ignition_points: List[Tuple[float, float]],
                           simulation_hours: int,
                           weather_params: Dict,
                           start_time: datetime) -> Dict:
    """Create metadata for simulation scenario."""
    scenario_id = f"sim_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    metadata = {
        'scenario_id': scenario_id,
        'created_at': start_time.isoformat(),
        'simulation_hours': simulation_hours,
        'ignition_points': ignition_points,
        'weather_parameters': weather_params,
        'model_version': '1.0',
        'resolution_meters': 30
    }
    
    return metadata

def tensor_to_numpy(tensor: tf.Tensor) -> np.ndarray:
    """Safely convert TensorFlow tensor to numpy array."""
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    return tensor

def numpy_to_tensor(array: np.ndarray) -> tf.Tensor:
    """Convert numpy array to TensorFlow tensor."""
    return tf.constant(array, dtype=tf.float32)

def validate_grid_bounds(grid: np.ndarray, row: int, col: int) -> bool:
    """Check if coordinates are within grid bounds."""
    return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]

def calculate_fire_intensity(probability: float, fuel_load: float, wind_speed: float) -> float:
    """Calculate fire intensity based on probability, fuel, and weather."""
    base_intensity = probability * fuel_load
    wind_boost = 1.0 + (wind_speed / 50.0) * 0.5  # Max 25% boost from wind
    return np.clip(base_intensity * wind_boost, 0.0, 1.0)

def calculate_slope_and_aspect_tf(dem: np.ndarray, resolution: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate slope and aspect from DEM using TensorFlow (enhanced from duplicate folder)
    
    Args:
        dem: Digital elevation model
        resolution: Pixel resolution in meters
        
    Returns:
        Tuple of (slope_degrees, aspect_degrees)
    """
    # Convert to TensorFlow tensor
    dem_tf = tf.constant(dem, dtype=tf.float32)
    dem_tf = dem_tf[tf.newaxis, :, :, tf.newaxis]  # Add batch and channel dims
    
    # Calculate gradients
    grad_y, grad_x = tf.image.image_gradients(dem_tf)
    
    # Remove extra dimensions
    grad_x = grad_x[0, :, :, 0] / resolution
    grad_y = grad_y[0, :, :, 0] / resolution
    
    # Calculate slope in radians then convert to degrees
    slope_rad = tf.atan(tf.sqrt(grad_x**2 + grad_y**2))
    slope_deg = slope_rad * 180.0 / np.pi
    
    # Calculate aspect (direction of steepest descent)
    aspect_rad = tf.atan2(-grad_y, -grad_x)  # Negative for geographic convention
    aspect_deg = aspect_rad * 180.0 / np.pi
    
    # Convert aspect to 0-360 range
    aspect_deg = tf.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)
    
    return slope_deg.numpy(), aspect_deg.numpy()

def resize_array_tf(array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize array using TensorFlow for GPU acceleration (from duplicate folder)
    
    Args:
        array: Input array to resize
        target_shape: Target (height, width)
        
    Returns:
        Resized array
    """
    if array.shape == target_shape:
        return array
        
    # Use TensorFlow for resizing
    tf_array = tf.constant(array[np.newaxis, :, :, np.newaxis])
    resized = tf.image.resize(tf_array, target_shape, method='bilinear')
    return resized[0, :, :, 0].numpy()

def create_fire_animation_data(simulation_frames: List[np.ndarray], 
                              metadata: Dict) -> Dict:
    """
    Prepare data for web animation (from duplicate folder)
    
    Args:
        simulation_frames: List of fire state arrays
        metadata: Geospatial metadata
        
    Returns:
        Dictionary with animation data
    """
    animation_data = {
        'frames': [],
        'bounds': metadata.get('bounds'),
        'shape': simulation_frames[0].shape if simulation_frames else None,
        'frame_count': len(simulation_frames)
    }
    
    for i, frame in enumerate(simulation_frames):
        # Convert to format suitable for web display
        frame_data = {
            'time_step': i,
            'fire_pixels': np.where(frame > 0.1),  # Locations with fire
            'fire_intensity': frame[frame > 0.1].tolist(),  # Intensity values
            'max_intensity': float(np.max(frame)),
            'total_burned_area': float(np.sum(frame > 0.1))  # Number of burning pixels
        }
        animation_data['frames'].append(frame_data)
    
    return animation_data

print("✅ CA Engine utilities loaded successfully")
