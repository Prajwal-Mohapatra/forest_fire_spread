# ====================
# CA Engine Configuration
# ====================
"""
Configuration parameters for the forest fire cellular automata simulation.
All parameters are optimized for the Uttarakhand region at 30m resolution.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict

# Simulation Parameters
CELL_SIZE = 30  # meters - matches ML model resolution
TIME_STEP = 3600  # seconds (1 hour)
MAX_SIMULATION_HOURS = 12
DEFAULT_SIMULATION_HOURS = [1, 2, 3, 6, 12]

# Fire Spread Parameters
BASE_SPREAD_RATE = 0.1  # base probability of fire spreading per hour
WIND_INFLUENCE_FACTOR = 2.0  # multiplier for wind direction
SLOPE_INFLUENCE_FACTOR = 1.5  # multiplier for uphill spread
MOISTURE_DAMPING_FACTOR = 0.3  # reduction factor for high moisture

# Probability Thresholds
IGNITION_THRESHOLD = 0.1  # minimum probability to sustain fire
SPREAD_THRESHOLD = 0.05  # minimum probability to spread fire
BURNING_THRESHOLD = 0.3  # probability above which cell is actively burning

# Environmental Parameters
FUEL_LOAD_MULTIPLIER = 1.2  # LULC-based fuel availability
BARRIER_RESISTANCE = 0.9  # resistance from roads/water bodies
SETTLEMENT_PROTECTION = 0.95  # fire suppression near settlements

# Wind Direction Mapping (meteorological convention)
WIND_DIRECTIONS = {
    0: (0, 1),    # North
    45: (1, 1),   # Northeast
    90: (1, 0),   # East
    135: (1, -1), # Southeast
    180: (0, -1), # South
    225: (-1, -1),# Southwest
    270: (-1, 0), # West
    315: (-1, 1)  # Northwest
}

# Neighborhood Configuration (Moore neighborhood with distance weighting)
NEIGHBOR_OFFSETS = [
    (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
    (0, -1, 1.0),                   (0, 1, 1.0),
    (1, -1, 1.414),  (1, 0, 1.0),  (1, 1, 1.414)
]

# TensorFlow/GPU Configuration
USE_GPU = True
MEMORY_GROWTH = True
BATCH_PROCESSING = True
CHUNK_SIZE = 1000  # for large grid processing

# Output Configuration
OUTPUT_FORMAT = "GTiff"
ANIMATION_FPS = 2
CONFIDENCE_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Visualization Parameters
FIRE_COLORMAP = {
    'no_fire': (0, 0, 0, 0),        # transparent
    'low_risk': (255, 255, 0, 128),  # yellow
    'medium_risk': (255, 165, 0, 180), # orange
    'high_risk': (255, 69, 0, 220),   # red-orange
    'active_fire': (220, 20, 60, 255) # crimson
}

# Data Paths (relative to project root)
DATA_PATHS = {
    'ml_predictions': 'working_forest_fire_ml/fire_pred_model/outputs/predictions/',
    'dem_data': 'dataset collection/',
    'weather_data': 'dataset collection/',
    'lulc_data': 'dataset collection/',
    'ghsl_data': 'dataset collection/',
    'output_dir': 'cellular_automata/outputs/'
}

# Error Handling
MISSING_DATA_FILL = 0.0
ERROR_THRESHOLD = 0.1  # acceptable error rate
MAX_RETRY_ATTEMPTS = 3

# Performance Monitoring
ENABLE_PROFILING = True
LOG_LEVEL = "INFO"
PROGRESS_UPDATE_INTERVAL = 100  # cells processed

# Model Integration
ML_MODEL_PATH = "working_forest_fire_ml/fire_pred_model/outputs/final_model.h5"
PREDICTION_CACHE_SIZE = 10  # number of daily predictions to cache

# Enhanced Configuration Classes (merged from duplicate folder)
@dataclass
class AdvancedCAConfig:
    """Advanced configuration class for Cellular Automata simulation"""
    
    # Spatial parameters
    resolution: float = 30.0  # meters per pixel
    
    # Temporal parameters
    time_step: float = 1.0  # hours per simulation step
    max_simulation_hours: int = 12  # maximum simulation duration
    
    # Fire spread parameters
    base_spread_rate: float = 0.1  # base probability of spread per hour
    wind_influence: float = 0.3  # wind influence factor (0-1)
    slope_influence: float = 0.2  # slope influence factor (0-1)
    fuel_influence: float = 0.4  # fuel/vegetation influence factor (0-1)
    
    # Weather constants (simplified - daily values)
    default_wind_speed: float = 5.0  # km/h
    default_wind_direction: float = 45.0  # degrees (0=North, 90=East)
    default_humidity: float = 40.0  # percentage
    default_temperature: float = 25.0  # celsius
    
    # Barrier effects
    urban_barrier_strength: float = 0.9  # how much urban areas block fire (0-1)
    water_barrier_strength: float = 1.0  # how much water blocks fire (0-1)
    road_barrier_strength: float = 0.3  # how much roads block fire (0-1)
    
    # Fire states
    UNBURNED = 0
    BURNING = 1
    BURNED = 2
    
    # Neighborhood type
    neighborhood_type: str = "moore"  # "moore" (8-cell) or "neumann" (4-cell)
    
    # Performance settings
    use_gpu: bool = True
    batch_processing: bool = True
    chunk_size: int = 1024  # for processing large areas in chunks

# Enhanced Land use classification for fire behavior (from duplicate folder)
LULC_FIRE_BEHAVIOR = {
    # High fire risk
    10: {"flammability": 0.9, "spread_rate": 1.2, "name": "Grassland"},
    20: {"flammability": 0.8, "spread_rate": 1.0, "name": "Shrubland"},
    30: {"flammability": 0.7, "spread_rate": 0.9, "name": "Deciduous Forest"},
    40: {"flammability": 0.6, "spread_rate": 0.8, "name": "Evergreen Forest"},
    
    # Medium fire risk
    50: {"flammability": 0.4, "spread_rate": 0.6, "name": "Mixed Forest"},
    60: {"flammability": 0.3, "spread_rate": 0.5, "name": "Agricultural"},
    
    # Low fire risk
    70: {"flammability": 0.1, "spread_rate": 0.2, "name": "Urban/Built"},
    80: {"flammability": 0.0, "spread_rate": 0.0, "name": "Water"},
    90: {"flammability": 0.0, "spread_rate": 0.0, "name": "Barren/Rock"},
    
    # Default for unknown classes
    0: {"flammability": 0.5, "spread_rate": 0.7, "name": "Unknown"}
}

# Slope effect on fire spread (enhanced function from duplicate folder)
def get_slope_factor(slope_degrees: float) -> float:
    """
    Calculate slope factor for fire spread
    Fire spreads faster uphill, slower downhill
    """
    if slope_degrees <= 0:
        return 0.8  # slightly slower on flat/downhill
    elif slope_degrees <= 15:
        return 1.0 + (slope_degrees / 15) * 0.5  # gradual increase
    elif slope_degrees <= 30:
        return 1.5 + (slope_degrees - 15) / 15 * 1.0  # steeper increase
    else:
        return 2.5  # maximum multiplier for very steep slopes

# Simulation scenarios (from duplicate folder)
SIMULATION_SCENARIOS = {
    "quick_demo": {
        "simulation_hours": 3,
        "output_frequency": 1,
        "description": "Quick 3-hour simulation for demo"
    },
    "short_term": {
        "simulation_hours": 6,
        "output_frequency": 1,
        "description": "Short-term 6-hour prediction"
    },
    "extended": {
        "simulation_hours": 12,
        "output_frequency": 2,
        "description": "Extended 12-hour simulation"
    },
    "detailed": {
        "simulation_hours": 24,
        "output_frequency": 4,
        "description": "Detailed 24-hour simulation"
    }
}

# Default enhanced configuration instance
DEFAULT_ADVANCED_CONFIG = AdvancedCAConfig()
