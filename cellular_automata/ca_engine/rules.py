# ====================
# Fire Spread Rules for Cellular Automata
# ====================
"""
Fire spread rules and physics for the cellular automata simulation.
Implements simplified fire behavior models optimized for the Uttarakhand region.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, List
from .config import *

class FireSpreadRules:
    """
    Fire spread rules implementation for cellular automata.
    Handles neighborhood analysis, environmental factors, and spread calculations.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        if use_gpu:
            self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        else:
            self.device = '/CPU:0'
            
        print(f"✅ FireSpreadRules initialized on {self.device}")
    
    def calculate_spread_probability(self, 
                                   current_state: tf.Tensor,
                                   base_probability: tf.Tensor,
                                   environmental_layers: Dict[str, tf.Tensor],
                                   weather_params: Dict[str, float],
                                   time_step: int) -> tf.Tensor:
        """
        Calculate fire spread probability for each cell based on:
        - Current fire state
        - Base fire probability from ML model
        - Environmental factors (elevation, fuel, barriers)
        - Weather conditions (wind, moisture)
        """
        
        with tf.device(self.device):
            # Get grid dimensions
            height, width = tf.shape(current_state)[0], tf.shape(current_state)[1]
            
            # Initialize spread probability grid
            spread_prob = tf.zeros_like(current_state, dtype=tf.float32)
            
            # Neighborhood influence calculation
            neighbor_influence = self._calculate_neighborhood_influence(
                current_state, weather_params
            )
            
            # Environmental modifiers
            env_modifier = self._calculate_environmental_modifier(
                environmental_layers, weather_params
            )
            
            # Temporal decay factor (fire intensity decreases over time)
            temporal_factor = self._calculate_temporal_factor(current_state, time_step)
            
            # Combine all factors
            spread_prob = (
                base_probability * 
                neighbor_influence * 
                env_modifier * 
                temporal_factor
            )
            
            # Apply barriers (roads, water bodies, settlements)
            if 'barriers' in environmental_layers:
                barrier_mask = environmental_layers['barriers']
                spread_prob = spread_prob * (1.0 - barrier_mask * BARRIER_RESISTANCE)
            
            # Ensure probabilities are in valid range
            spread_prob = tf.clip_by_value(spread_prob, 0.0, 1.0)
            
            return spread_prob
    
    def _calculate_neighborhood_influence(self, 
                                        current_state: tf.Tensor,
                                        weather_params: Dict[str, float]) -> tf.Tensor:
        """
        Calculate influence from neighboring burning cells.
        Considers wind direction and distance weighting.
        """
        
        # Wind parameters
        wind_direction = weather_params.get('wind_direction', 0)  # degrees
        wind_speed = weather_params.get('wind_speed', 10)  # km/h
        
        # Create convolution kernel for neighborhood analysis
        kernel = self._create_wind_influenced_kernel(wind_direction, wind_speed)
        
        # Expand dimensions for convolution
        current_expanded = tf.expand_dims(tf.expand_dims(current_state, 0), -1)
        kernel_expanded = tf.expand_dims(tf.expand_dims(kernel, -1), -1)
        
        # Apply convolution to calculate neighborhood influence
        neighbor_influence = tf.nn.conv2d(
            current_expanded, 
            kernel_expanded, 
            strides=[1, 1, 1, 1], 
            padding='SAME'
        )
        
        # Remove extra dimensions
        neighbor_influence = tf.squeeze(neighbor_influence)
        
        # Normalize by maximum possible influence
        max_influence = tf.reduce_max(kernel)
        neighbor_influence = neighbor_influence / max_influence
        
        return tf.clip_by_value(neighbor_influence, 0.0, 1.0)
    
    def _create_wind_influenced_kernel(self, 
                                     wind_direction: float, 
                                     wind_speed: float) -> tf.Tensor:
        """
        Create a 3x3 convolution kernel influenced by wind direction.
        Higher weights in the downwind direction.
        """
        
        # Base Moore neighborhood kernel
        base_kernel = tf.constant([
            [0.707, 1.0, 0.707],  # Diagonal neighbors have distance weight
            [1.0,   0.0, 1.0],    # Direct neighbors, center is 0
            [0.707, 1.0, 0.707]
        ], dtype=tf.float32)
        
        # Wind influence matrix
        wind_rad = tf.constant(wind_direction * np.pi / 180.0)
        wind_vector = tf.stack([tf.sin(wind_rad), tf.cos(wind_rad)])
        
        # Calculate wind influence for each kernel position
        positions = tf.constant([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],  [0, 0],  [0, 1],
            [1, -1],  [1, 0],  [1, 1]
        ], dtype=tf.float32)
        
        wind_influences = tf.TensorArray(tf.float32, size=9)
        
        for i in tf.range(9):
            if i == 4:  # Center position
                wind_influences = wind_influences.write(i, 0.0)
            else:
                pos = positions[i]
                pos_norm = tf.norm(pos)
                if pos_norm > 0:
                    pos_normalized = pos / pos_norm
                    alignment = tf.reduce_sum(wind_vector * pos_normalized)
                    wind_factor = 1.0 + (wind_speed / 50.0) * alignment
                    wind_influences = wind_influences.write(i, tf.maximum(wind_factor, 0.1))
                else:
                    wind_influences = wind_influences.write(i, 1.0)
        
        wind_matrix = tf.reshape(wind_influences.stack(), [3, 3])
        
        # Combine base kernel with wind influence
        final_kernel = base_kernel * wind_matrix
        
        # Normalize kernel
        kernel_sum = tf.reduce_sum(final_kernel)
        if kernel_sum > 0:
            final_kernel = final_kernel / kernel_sum
        
        return final_kernel
    
    def _calculate_environmental_modifier(self, 
                                        environmental_layers: Dict[str, tf.Tensor],
                                        weather_params: Dict[str, float]) -> tf.Tensor:
        """
        Calculate environmental modification factor based on:
        - Elevation/slope effects
        - Fuel load (vegetation type)
        - Moisture content
        - Settlement protection
        """
        
        modifier = tf.ones_like(environmental_layers['elevation'], dtype=tf.float32)
        
        # Slope influence (fire spreads faster uphill)
        if 'slope' in environmental_layers:
            slope_modifier = 1.0 + (environmental_layers['slope'] / 90.0) * SLOPE_INFLUENCE_FACTOR
            modifier = modifier * slope_modifier
        
        # Fuel load influence
        if 'fuel_load' in environmental_layers:
            fuel_modifier = environmental_layers['fuel_load'] * FUEL_LOAD_MULTIPLIER
            modifier = modifier * fuel_modifier
        
        # Moisture damping effect
        moisture = weather_params.get('relative_humidity', 50) / 100.0
        moisture_modifier = 1.0 - (moisture * MOISTURE_DAMPING_FACTOR)
        modifier = modifier * moisture_modifier
        
        # Temperature boost
        temperature = weather_params.get('temperature', 25)  # Celsius
        temp_modifier = 1.0 + ((temperature - 20) / 50.0) * 0.3  # 30% boost at 70°C
        modifier = modifier * temp_modifier
        
        return tf.clip_by_value(modifier, 0.1, 3.0)
    
    def _calculate_temporal_factor(self, 
                                 current_state: tf.Tensor, 
                                 time_step: int) -> tf.Tensor:
        """
        Calculate temporal decay factor.
        Fire intensity decreases over time without fuel replenishment.
        """
        
        # Simple temporal decay - fires burn out gradually
        decay_rate = 0.05  # 5% intensity loss per hour
        temporal_factor = tf.maximum(
            1.0 - (time_step * decay_rate),
            0.3  # Minimum factor to prevent complete extinction
        )
        
        return tf.ones_like(current_state) * temporal_factor
    
    def update_fire_state(self, 
                         current_state: tf.Tensor,
                         spread_probability: tf.Tensor,
                         ignition_mask: tf.Tensor = None) -> tf.Tensor:
        """
        Update fire state based on spread probabilities.
        
        Args:
            current_state: Current fire intensity grid (0-1)
            spread_probability: Calculated spread probabilities
            ignition_mask: New ignition points (optional)
            
        Returns:
            Updated fire state grid
        """
        
        with tf.device(self.device):
            # Start with current state
            new_state = tf.identity(current_state)
            
            # Add new ignitions if provided
            if ignition_mask is not None:
                new_state = tf.maximum(new_state, ignition_mask)
            
            # Apply spread probabilities
            # Fire spreads to cells based on calculated probabilities
            spread_mask = tf.greater(spread_probability, SPREAD_THRESHOLD)
            random_values = tf.random.uniform(tf.shape(spread_probability))
            
            # Probabilistic spreading
            new_fires = tf.logical_and(
                spread_mask,
                tf.less(random_values, spread_probability)
            )
            
            # Update state where new fires occur
            new_fire_intensity = spread_probability * 0.8  # New fires start at 80% intensity
            new_state = tf.where(
                new_fires,
                tf.maximum(new_state, new_fire_intensity),
                new_state
            )
            
            # Apply burning threshold
            new_state = tf.where(
                tf.greater(new_state, BURNING_THRESHOLD),
                new_state,
                tf.zeros_like(new_state)
            )
            
            # Gradual intensity decay for existing fires
            decay_factor = 0.95  # 5% decay per time step
            new_state = new_state * decay_factor
            
            return tf.clip_by_value(new_state, 0.0, 1.0)
    
    def apply_suppression_effects(self, 
                                 fire_state: tf.Tensor,
                                 environmental_layers: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Apply fire suppression effects near settlements and roads.
        Simplified suppression model.
        """
        
        if 'barriers' not in environmental_layers:
            return fire_state
        
        # Reduce fire intensity near barriers (represents suppression efforts)
        barriers = environmental_layers['barriers']
        
        # Create suppression zone around barriers
        suppression_kernel = tf.constant([
            [0.1, 0.3, 0.1],
            [0.3, 1.0, 0.3],
            [0.1, 0.3, 0.1]
        ], dtype=tf.float32)
        
        # Expand dimensions for convolution
        barriers_expanded = tf.expand_dims(tf.expand_dims(barriers, 0), -1)
        kernel_expanded = tf.expand_dims(tf.expand_dims(suppression_kernel, -1), -1)
        
        # Calculate suppression zones
        suppression_zones = tf.nn.conv2d(
            barriers_expanded,
            kernel_expanded,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        
        suppression_zones = tf.squeeze(suppression_zones)
        suppression_zones = tf.clip_by_value(suppression_zones, 0.0, 1.0)
        
        # Apply suppression effect
        suppression_factor = 1.0 - (suppression_zones * SETTLEMENT_PROTECTION)
        suppressed_state = fire_state * suppression_factor
        
        return suppressed_state
    
    def calculate_fire_statistics(self, fire_state: tf.Tensor) -> Dict[str, float]:
        """Calculate fire spread statistics for monitoring."""
        
        total_cells = tf.cast(tf.size(fire_state), tf.float32)
        burning_cells = tf.reduce_sum(tf.cast(tf.greater(fire_state, IGNITION_THRESHOLD), tf.float32))
        high_intensity_cells = tf.reduce_sum(tf.cast(tf.greater(fire_state, BURNING_THRESHOLD), tf.float32))
        
        stats = {
            'total_burning_cells': int(burning_cells.numpy()),
            'high_intensity_cells': int(high_intensity_cells.numpy()),
            'burned_area_km2': float(burning_cells.numpy() * (CELL_SIZE * CELL_SIZE) / 1e6),
            'fire_perimeter_km': float(burning_cells.numpy() * CELL_SIZE / 1000.0),
            'max_intensity': float(tf.reduce_max(fire_state).numpy()),
            'mean_intensity': float(tf.reduce_mean(fire_state).numpy())
        }
        
        return stats

print("✅ Fire spread rules module loaded successfully")

class SimplifiedFireRules:
    """
    Simplified fire spread rules for rapid prototyping (from duplicate folder)
    Enhanced with numpy-based operations for faster development iterations
    """
    
    def __init__(self, config_dict: Dict = None):
        # Use basic configuration if none provided
        self.config = {
            'base_spread_rate': 0.1,
            'wind_influence': 0.3,
            'slope_influence': 0.2,
            'fuel_influence': 0.4
        }
        if config_dict:
            self.config.update(config_dict)
    
    def simple_spread(self,
                     fire_state: np.ndarray,
                     probability_map: np.ndarray,
                     wind_direction: float = 0.0,
                     wind_speed: float = 10.0) -> np.ndarray:
        """
        Simple fire spread using numpy operations
        
        Args:
            fire_state: Current fire state
            probability_map: Fire probability map
            wind_direction: Wind direction in degrees
            wind_speed: Wind speed in km/h
            
        Returns:
            Updated fire state
        """
        from scipy import ndimage
        
        # Create simple kernel with wind bias
        kernel = self._create_wind_biased_kernel(wind_direction, wind_speed)
        
        # Calculate neighbor influence
        neighbor_influence = ndimage.convolve(fire_state, kernel, mode='constant')
        
        # Combine with probability map
        spread_probability = (
            self.config['base_spread_rate'] * 
            probability_map * 
            (1.0 + neighbor_influence)
        )
        
        # Apply stochastic spread
        random_values = np.random.random(fire_state.shape)
        new_ignitions = (random_values < spread_probability).astype(np.float32)
        
        # Keep existing fire and add new ignitions
        new_state = np.maximum(fire_state, new_ignitions)
        
        return new_state
    
    def _create_wind_biased_kernel(self, wind_direction: float, wind_speed: float) -> np.ndarray:
        """Create wind-biased convolution kernel"""
        
        # Base kernel
        kernel = np.array([
            [0.1, 0.2, 0.1],
            [0.2, 0.0, 0.2],
            [0.1, 0.2, 0.1]
        ])
        
        # Apply wind bias (simplified)
        wind_factor = min(wind_speed / 20.0, 1.0) * self.config['wind_influence']
        
        # Adjust kernel based on wind direction quadrant
        if 0 <= wind_direction < 90:  # NE quadrant
            kernel[0, 2] *= (1.0 + wind_factor)  # Boost NE
            kernel[2, 0] *= (1.0 - wind_factor * 0.5)  # Reduce SW
        elif 90 <= wind_direction < 180:  # SE quadrant
            kernel[2, 2] *= (1.0 + wind_factor)  # Boost SE
            kernel[0, 0] *= (1.0 - wind_factor * 0.5)  # Reduce NW
        elif 180 <= wind_direction < 270:  # SW quadrant
            kernel[2, 0] *= (1.0 + wind_factor)  # Boost SW
            kernel[0, 2] *= (1.0 - wind_factor * 0.5)  # Reduce NE
        else:  # NW quadrant
            kernel[0, 0] *= (1.0 + wind_factor)  # Boost NW
            kernel[2, 2] *= (1.0 - wind_factor * 0.5)  # Reduce SE
        
        return kernel
    
    def apply_lulc_effects(self, 
                          fire_state: np.ndarray,
                          lulc_map: np.ndarray,
                          lulc_behavior: Dict = None) -> np.ndarray:
        """
        Apply land use/land cover effects to fire spread
        
        Args:
            fire_state: Current fire state
            lulc_map: Land use land cover map
            lulc_behavior: LULC fire behavior dictionary
            
        Returns:
            Modified fire state with LULC effects
        """
        if lulc_behavior is None:
            # Use default from config if available
            from .config import LULC_FIRE_BEHAVIOR
            lulc_behavior = LULC_FIRE_BEHAVIOR
        
        # Create flammability map
        flammability_map = np.ones_like(fire_state)
        
        for lulc_class, behavior in lulc_behavior.items():
            mask = (lulc_map == lulc_class)
            flammability_map[mask] = behavior['flammability']
        
        # Apply flammability effects
        modified_state = fire_state * flammability_map
        
        return modified_state
