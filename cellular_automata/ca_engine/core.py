# ====================
# Cellular Automata Core Engine
# ====================
"""
Main cellular automata engine for forest fire spread simulation.
Integrates ML predictions, environmental data, and fire spread rules.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os
import json

from . import config
from .rules import FireSpreadRules
from .utils import (
    setup_tensorflow_gpu, load_probability_map, load_environmental_layers,
    create_ignition_points, save_simulation_frame, create_scenario_metadata,
    tensor_to_numpy, numpy_to_tensor
)

class ForestFireCA:
    """
    Forest Fire Cellular Automata Simulation Engine.
    
    Simulates fire spread using:
    - ML-generated daily fire probability maps
    - Environmental layers (DEM, LULC, GHSL)
    - Weather parameters
    - User-defined ignition points
    """
    
    def __init__(self, use_gpu: bool = True, config_override: Dict = None):
        """
        Initialize the CA engine.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            config_override: Optional configuration overrides
        """
        self.config = config
        if config_override:
            for key, value in config_override.items():
                setattr(self.config, key, value)
        
        # Setup TensorFlow
        self.use_gpu = use_gpu and setup_tensorflow_gpu()
        
        # Initialize fire spread rules
        self.fire_rules = FireSpreadRules(use_gpu=self.use_gpu)
        
        # Simulation state
        self.current_state = None
        self.environmental_layers = None
        self.metadata = None
        self.scenario_metadata = None
        
        print(f"✅ ForestFireCA initialized with GPU: {self.use_gpu}")
    
    def load_base_probability_map(self, probability_map_path: str) -> bool:
        """
        Load the base fire probability map from ML model output.
        
        Args:
            probability_map_path: Path to .tif file with fire probabilities
            
        Returns:
            Success status
        """
        try:
            # Load probability map
            prob_array, metadata = load_probability_map(probability_map_path)
            self.base_probability = numpy_to_tensor(prob_array)
            self.metadata = metadata
            
            # Load corresponding environmental layers
            self.environmental_layers = load_environmental_layers(metadata, self.config)
            
            # Convert to tensors
            for key, array in self.environmental_layers.items():
                self.environmental_layers[key] = numpy_to_tensor(array)
            
            print(f"✅ Base probability map loaded: {prob_array.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load probability map: {e}")
            return False
    
    def initialize_simulation(self, 
                            ignition_points: List[Tuple[float, float]],
                            weather_params: Dict[str, float],
                            simulation_hours: int = 6) -> str:
        """
        Initialize a new fire simulation scenario.
        
        Args:
            ignition_points: List of (longitude, latitude) tuples for ignition
            weather_params: Weather conditions (wind_direction, wind_speed, etc.)
            simulation_hours: Total hours to simulate
            
        Returns:
            Scenario ID for tracking
        """
        if self.base_probability is None:
            raise ValueError("Base probability map must be loaded first")
        
        # Create scenario metadata
        start_time = datetime.now()
        self.scenario_metadata = create_scenario_metadata(
            ignition_points, simulation_hours, weather_params, start_time
        )
        
        # Initialize fire state grid
        grid_shape = tensor_to_numpy(self.base_probability).shape
        self.current_state = tf.zeros(grid_shape, dtype=tf.float32)
        
        # Set ignition points
        ignition_mask = create_ignition_points(
            grid_shape, ignition_points, self.metadata['transform']
        )
        self.current_state = tf.maximum(self.current_state, numpy_to_tensor(ignition_mask))
        
        # Store simulation parameters
        self.weather_params = weather_params
        self.simulation_hours = simulation_hours
        self.current_hour = 0
        
        scenario_id = self.scenario_metadata['scenario_id']
        print(f"✅ Simulation initialized: {scenario_id}")
        return scenario_id
    
    def step_simulation(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Perform one time step of the simulation.
        
        Returns:
            Current fire state array and statistics
        """
        if self.current_state is None:
            raise ValueError("Simulation must be initialized first")
        
        # Calculate spread probabilities
        spread_prob = self.fire_rules.calculate_spread_probability(
            self.current_state,
            self.base_probability,
            self.environmental_layers,
            self.weather_params,
            self.current_hour
        )
        
        # Update fire state
        self.current_state = self.fire_rules.update_fire_state(
            self.current_state,
            spread_prob
        )
        
        # Apply suppression effects
        self.current_state = self.fire_rules.apply_suppression_effects(
            self.current_state,
            self.environmental_layers
        )
        
        # Calculate statistics
        stats = self.fire_rules.calculate_fire_statistics(self.current_state)
        stats['simulation_hour'] = self.current_hour
        
        self.current_hour += 1
        
        # Convert to numpy for output
        fire_state_np = tensor_to_numpy(self.current_state)
        
        print(f"⏰ Hour {self.current_hour}: {stats['total_burning_cells']} cells burning")
        return fire_state_np, stats
    
    def run_full_simulation(self, 
                          ignition_points: List[Tuple[float, float]],
                          weather_params: Dict[str, float],
                          simulation_hours: int = 6,
                          save_frames: bool = True,
                          output_dir: str = None) -> Dict:
        """
        Run complete fire simulation and save results.
        
        Args:
            ignition_points: List of ignition coordinates
            weather_params: Weather conditions
            simulation_hours: Total simulation time
            save_frames: Whether to save hourly frames
            output_dir: Output directory for results
            
        Returns:
            Complete simulation results
        """
        # Initialize simulation
        scenario_id = self.initialize_simulation(
            ignition_points, weather_params, simulation_hours
        )
        
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.join(self.config.DATA_PATHS['output_dir'], scenario_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Store results
        results = {
            'scenario_id': scenario_id,
            'metadata': self.scenario_metadata,
            'hourly_states': [],
            'hourly_statistics': [],
            'frame_paths': []
        }
        
        # Save initial state
        initial_state = tensor_to_numpy(self.current_state)
        if save_frames:
            frame_path = save_simulation_frame(
                initial_state, 0, self.metadata, output_dir, scenario_id
            )
            results['frame_paths'].append(frame_path)
        
        results['hourly_states'].append(initial_state.tolist())
        
        # Run simulation steps
        print(f"🔥 Starting {simulation_hours}-hour simulation...")
        
        for hour in range(1, simulation_hours + 1):
            # Perform simulation step
            fire_state, stats = self.step_simulation()
            
            # Store results
            results['hourly_states'].append(fire_state.tolist())
            results['hourly_statistics'].append(stats)
            
            # Save frame if requested
            if save_frames:
                frame_path = save_simulation_frame(
                    fire_state, hour, self.metadata, output_dir, scenario_id
                )
                results['frame_paths'].append(frame_path)
            
            # Check for simulation completion
            if stats['total_burning_cells'] == 0:
                print(f"🔥 Fire extinguished at hour {hour}")
                break
        
        # Save complete results
        results_path = os.path.join(output_dir, f"{scenario_id}_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'scenario_id': results['scenario_id'],
            'metadata': results['metadata'],
            'hourly_statistics': results['hourly_statistics'],
            'frame_paths': results['frame_paths'],
            'total_hours_simulated': len(results['hourly_statistics'])
        }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Simulation complete: {len(results['hourly_statistics'])} hours")
        print(f"📁 Results saved to: {output_dir}")
        
        return results
    
    def get_current_state(self) -> Optional[np.ndarray]:
        """Get current fire state as numpy array."""
        if self.current_state is not None:
            return tensor_to_numpy(self.current_state)
        return None
    
    def add_ignition_point(self, longitude: float, latitude: float) -> bool:
        """
        Add a new ignition point during simulation.
        
        Args:
            longitude, latitude: Geographic coordinates
            
        Returns:
            Success status
        """
        if self.current_state is None or self.metadata is None:
            return False
        
        try:
            grid_shape = tensor_to_numpy(self.current_state).shape
            ignition_mask = create_ignition_points(
                grid_shape, [(longitude, latitude)], self.metadata['transform']
            )
            
            # Add to current state
            self.current_state = tf.maximum(
                self.current_state, 
                numpy_to_tensor(ignition_mask)
            )
            
            print(f"🔥 Added ignition point at ({longitude:.4f}, {latitude:.4f})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add ignition point: {e}")
            return False
    
    def reset_simulation(self):
        """Reset simulation state."""
        self.current_state = None
        self.scenario_metadata = None
        self.current_hour = 0
        print("🔄 Simulation reset")
    
    def get_simulation_info(self) -> Dict:
        """Get current simulation information."""
        if self.scenario_metadata is None:
            return {}
        
        info = {
            'scenario_id': self.scenario_metadata['scenario_id'],
            'current_hour': self.current_hour,
            'total_hours': self.simulation_hours,
            'ignition_points': self.scenario_metadata['ignition_points'],
            'weather_params': self.weather_params
        }
        
        if self.current_state is not None:
            stats = self.fire_rules.calculate_fire_statistics(self.current_state)
            info.update(stats)
        
        return info

# Convenience function for quick simulations
def run_quick_simulation(probability_map_path: str,
                        ignition_points: List[Tuple[float, float]],
                        weather_params: Dict[str, float] = None,
                        simulation_hours: int = 6,
                        output_dir: str = None) -> Dict:
    """
    Run a quick fire simulation with default parameters.
    
    Args:
        probability_map_path: Path to ML probability map
        ignition_points: List of ignition coordinates
        weather_params: Weather conditions (optional)
        simulation_hours: Simulation duration
        output_dir: Output directory
        
    Returns:
        Simulation results
    """
    # Default weather parameters
    if weather_params is None:
        weather_params = {
            'wind_direction': 45,  # Northeast
            'wind_speed': 15,      # 15 km/h
            'temperature': 30,     # 30°C
            'relative_humidity': 40 # 40%
        }
    
    # Create CA engine
    ca_engine = ForestFireCA(use_gpu=True)
    
    # Load probability map
    if not ca_engine.load_base_probability_map(probability_map_path):
        raise ValueError(f"Failed to load probability map: {probability_map_path}")
    
    # Run simulation
    results = ca_engine.run_full_simulation(
        ignition_points=ignition_points,
        weather_params=weather_params,
        simulation_hours=simulation_hours,
        save_frames=True,
        output_dir=output_dir
    )
    
    return results

print("✅ Cellular Automata core engine loaded successfully")
