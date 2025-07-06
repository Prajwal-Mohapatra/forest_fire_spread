# ====================
# CA Engine Package Initialization
# ====================
"""
Forest Fire Cellular Automata Engine
=====================================

A TensorFlow-based cellular automata simulation engine for forest fire spread prediction.
Integrates with ML-generated fire probability maps and environmental data.

Main Components:
- core.py: Main simulation engine
- rules.py: Fire spread rules and physics
- utils.py: Utility functions for data handling
- config.py: Configuration parameters

Usage:
    from cellular_automata.ca_engine import ForestFireCA, run_quick_simulation
    
    # Quick simulation
    results = run_quick_simulation(
        probability_map_path="path/to/probability.tif",
        ignition_points=[(longitude, latitude)],
        simulation_hours=6
    )
    
    # Advanced usage
    ca_engine = ForestFireCA(use_gpu=True)
    ca_engine.load_base_probability_map("probability.tif")
    scenario_id = ca_engine.initialize_simulation(ignition_points, weather_params)
    fire_state, stats = ca_engine.step_simulation()
"""

from .core import ForestFireCA, run_quick_simulation
from .rules import FireSpreadRules
from .utils import (
    setup_tensorflow_gpu, 
    load_probability_map, 
    create_ignition_points,
    save_simulation_frame
)
from . import config

__version__ = "1.0.0"
__author__ = "Forest Fire Prediction Team"

# Package metadata
__all__ = [
    "ForestFireCA",
    "run_quick_simulation", 
    "FireSpreadRules",
    "config",
    "setup_tensorflow_gpu",
    "load_probability_map",
    "create_ignition_points",
    "save_simulation_frame"
]

# Configuration check
try:
    import tensorflow as tf
    import numpy as np
    import rasterio
    
    # Check TensorFlow GPU availability
    gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0
    
    print(f"✅ CA Engine package loaded successfully")
    print(f"📊 TensorFlow version: {tf.__version__}")
    print(f"🖥️  GPU available: {gpu_available}")
    print(f"📏 Default cell size: {config.CELL_SIZE}m")
    print(f"⏰ Default time step: {config.TIME_STEP}s")
    
except ImportError as e:
    print(f"⚠️ Missing dependencies: {e}")
    print("Please install required packages: tensorflow, numpy, rasterio")

# Example usage documentation
EXAMPLE_USAGE = """
Example Usage:
==============

1. Quick Simulation:
   -----------------
   from cellular_automata.ca_engine import run_quick_simulation
   
   results = run_quick_simulation(
       probability_map_path="data/fire_probability_2016_05_15.tif",
       ignition_points=[(77.5, 30.2)],  # longitude, latitude
       weather_params={
           'wind_direction': 45,    # degrees
           'wind_speed': 20,        # km/h  
           'temperature': 35,       # Celsius
           'relative_humidity': 30  # percent
       },
       simulation_hours=6
   )

2. Step-by-step Simulation:
   ------------------------
   from cellular_automata.ca_engine import ForestFireCA
   
   # Initialize engine
   ca = ForestFireCA(use_gpu=True)
   ca.load_base_probability_map("probability.tif")
   
   # Setup simulation
   scenario_id = ca.initialize_simulation(
       ignition_points=[(77.5, 30.2)],
       weather_params={'wind_direction': 45, 'wind_speed': 20}
   )
   
   # Run simulation steps
   for hour in range(6):
       fire_state, stats = ca.step_simulation()
       print(f"Hour {hour}: {stats['total_burning_cells']} cells burning")

3. Interactive Simulation:
   -----------------------
   # Add ignition points during simulation
   ca.add_ignition_point(77.6, 30.3)
   
   # Get current state
   current_fire = ca.get_current_state()
   
   # Get simulation info
   info = ca.get_simulation_info()
"""

def print_example_usage():
    """Print example usage instructions."""
    print(EXAMPLE_USAGE)
