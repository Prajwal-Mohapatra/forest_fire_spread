# ====================
# CA Engine Configuration
# ====================
"""
Configuration parameters for the forest fire cellular automata simulation.
All parameters are optimized for the Uttarakhand region at 30m resolution.
"""

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
