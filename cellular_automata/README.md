# Forest Fire Cellular Automata Engine

A TensorFlow-based cellular automata simulation engine for modeling forest fire spread in the Uttarakhand region. This engine integrates with machine learning-generated fire probability maps to provide realistic fire spread simulations.

## Features

- **GPU-Accelerated**: TensorFlow-based implementation with CUDA support
- **ML Integration**: Uses ResUNet-A generated probability maps as input
- **Environmental Modeling**: Incorporates elevation, vegetation, and barrier data
- **Weather-Aware**: Considers wind direction, speed, temperature, and humidity
- **Interactive**: Supports real-time ignition point addition
- **Scalable**: Optimized for full Uttarakhand state at 30m resolution

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify TensorFlow GPU setup (optional)
python -c "import tensorflow as tf; print(f'GPU Available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"
```

### Basic Usage

```python
from cellular_automata.ca_engine import run_quick_simulation

# Run a 6-hour fire simulation
results = run_quick_simulation(
    probability_map_path="data/fire_probability_2016_05_15.tif",
    ignition_points=[(77.5, 30.2)],  # longitude, latitude
    weather_params={
        'wind_direction': 45,    # degrees (meteorological convention)
        'wind_speed': 20,        # km/h
        'temperature': 35,       # Celsius
        'relative_humidity': 30  # percent
    },
    simulation_hours=6
)

print(f"Simulation ID: {results['scenario_id']}")
print(f"Total hours simulated: {results['total_hours_simulated']}")
```

### Advanced Usage

```python
from cellular_automata.ca_engine import ForestFireCA

# Initialize CA engine
ca = ForestFireCA(use_gpu=True)

# Load base probability map from ML model
ca.load_base_probability_map("data/fire_probability_2016_05_15.tif")

# Initialize simulation
scenario_id = ca.initialize_simulation(
    ignition_points=[(77.5, 30.2), (77.6, 30.3)],
    weather_params={
        'wind_direction': 225,  # Southwest wind
        'wind_speed': 15,
        'temperature': 30,
        'relative_humidity': 40
    },
    simulation_hours=12
)

# Run step-by-step simulation
for hour in range(12):
    fire_state, stats = ca.step_simulation()

    print(f"Hour {hour + 1}:")
    print(f"  - Burning cells: {stats['total_burning_cells']}")
    print(f"  - Burned area: {stats['burned_area_km2']:.2f} km²")
    print(f"  - Max intensity: {stats['max_intensity']:.3f}")

    # Add new ignition point at hour 3
    if hour == 2:
        ca.add_ignition_point(77.7, 30.4)
```

## Architecture

### Core Components

1. **`core.py`** - Main simulation engine

   - `ForestFireCA`: Primary simulation class
   - `run_quick_simulation()`: Convenience function

2. **`rules.py`** - Fire spread physics

   - `FireSpreadRules`: TensorFlow-based spread calculations
   - Neighborhood analysis with wind influence
   - Environmental factor integration

3. **`utils.py`** - Data handling utilities

   - GeoTIFF loading and saving
   - Coordinate transformations
   - GPU configuration

4. **`config.py`** - Configuration parameters
   - Simulation parameters
   - Fire behavior constants
   - Output settings

### Data Flow

```
ML Probability Map (.tif) → CA Engine → Hourly Fire States
                     ↑                      ↓
            Environmental Data         GeoTIFF Outputs
            Weather Parameters         Animation Frames
            Ignition Points           Statistics & Metadata
```

## Input Requirements

### ML Probability Maps

- **Format**: GeoTIFF (.tif)
- **Values**: 0.0-1.0 (fire probability)
- **Resolution**: 30m
- **Projection**: Any (automatically handled)
- **Bands**: Single band with probability values

### Environmental Data (Synthetic)

- **Elevation**: DEM-derived elevation and slope
- **Fuel Load**: LULC-derived vegetation density
- **Barriers**: GHSL-derived roads, water, settlements

### Weather Parameters

```python
weather_params = {
    'wind_direction': 45,      # degrees (0=North, 90=East, 180=South, 270=West)
    'wind_speed': 20,          # km/h
    'temperature': 35,         # Celsius
    'relative_humidity': 30    # percent
}
```

### Ignition Points

```python
ignition_points = [
    (77.5, 30.2),  # longitude, latitude
    (77.6, 30.3),  # multiple ignition points supported
]
```

## Output

### Simulation Results

- **Hourly Fire States**: 2D arrays with fire intensity (0-1)
- **GeoTIFF Frames**: Georeferenced fire maps for each hour
- **Statistics**: Burned area, fire perimeter, intensity metrics
- **Metadata**: Scenario details, parameters, timestamps

### File Structure

```
outputs/
└── sim_20240706_143022/
    ├── fire_spread_sim_20240706_143022_hour_00.tif
    ├── fire_spread_sim_20240706_143022_hour_01.tif
    ├── ...
    ├── fire_spread_sim_20240706_143022_hour_06.tif
    └── sim_20240706_143022_results.json
```

## Performance

### GPU Acceleration

- **Memory**: Automatically manages GPU memory growth
- **Batch Processing**: Optimized for large grids
- **Chunking**: Processes large areas in manageable chunks

### Typical Performance

- **Grid Size**: 3000×3000 cells (90km × 90km at 30m resolution)
- **Processing**: ~2-5 seconds per hour on RTX 3080
- **Memory**: ~4-8GB GPU memory for full Uttarakhand

## Configuration

### Simulation Parameters

```python
# In config.py
CELL_SIZE = 30              # meters
TIME_STEP = 3600           # seconds (1 hour)
BASE_SPREAD_RATE = 0.1     # base spread probability
WIND_INFLUENCE_FACTOR = 2.0 # wind effect multiplier
SLOPE_INFLUENCE_FACTOR = 1.5 # uphill spread boost
```

### Thresholds

```python
IGNITION_THRESHOLD = 0.1    # minimum fire probability
SPREAD_THRESHOLD = 0.05     # minimum spread probability
BURNING_THRESHOLD = 0.3     # active fire threshold
```

## Integration with ML Pipeline

### Expected ML Model Outputs

The CA engine expects daily fire probability maps generated by the ResUNet-A model:

```python
# From forest_fire_ml/predict.py
prediction = predict_fire_probability(
    model_path="final_model.h5",
    input_tif_path="stacked_data_2016_05_15.tif",
    output_path="fire_probability_2016_05_15.tif"
)
```

### Integration Example

```python
# Generate ML prediction
from forest_fire_ml..predict import predict_fire_probability

# Predict fire probability for specific date
prob_map_path = predict_fire_probability(
    model_path="forest_fire_ml/outputs/final_model.h5",
    input_tif_path="data/stacked_2016_05_15.tif",
    output_path="outputs/probability_2016_05_15.tif"
)

# Run CA simulation
from cellular_automata.ca_engine import run_quick_simulation

results = run_quick_simulation(
    probability_map_path=prob_map_path,
    ignition_points=[(77.5, 30.2)],
    simulation_hours=6
)
```

## Web Interface Integration

The CA engine is designed to integrate with web applications:

```python
# Flask API endpoint example
@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    data = request.json

    results = run_quick_simulation(
        probability_map_path=data['probability_map'],
        ignition_points=data['ignition_points'],
        weather_params=data['weather'],
        simulation_hours=data['hours']
    )

    return jsonify({
        'scenario_id': results['scenario_id'],
        'frame_paths': results['frame_paths'],
        'statistics': results['hourly_statistics']
    })
```

## Development Status

- ✅ **Core Engine**: Complete TensorFlow implementation
- ✅ **Fire Physics**: Simplified spread rules with wind/slope effects
- ✅ **Data Integration**: ML probability map loading
- ✅ **Output Generation**: GeoTIFF frames and JSON metadata
- 🔄 **Environmental Data**: Currently using synthetic data
- 🔄 **Web Integration**: API endpoints for web interface
- 📋 **Future**: Rothermel physics, real environmental data

## Dependencies

See `requirements.txt` for complete list. Key dependencies:

- TensorFlow ≥2.8.0 (with GPU support)
- NumPy ≥1.21.0
- Rasterio ≥1.3.0 (geospatial I/O)
- OpenCV ≥4.5.0 (image processing)

## Contributing

This engine is part of the Uttarakhand Forest Fire Prediction project. For integration with the ML pipeline and web interface, coordinate with the respective development teams.

## License

Part of the Forest Fire Spread Prediction System for Uttarakhand region.
