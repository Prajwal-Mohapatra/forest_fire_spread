# ====================
# Web API for Fire Simulation
# ====================
"""
Flask-based web API for the forest fire simulation system.
Provides endpoints for running simulations and retrieving results.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import uuid

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "cellular_automata"))

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(project_root, "cellular_automata", "uploads")
app.config['OUTPUT_FOLDER'] = os.path.join(project_root, "cellular_automata", "outputs")

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Store active simulations
active_simulations = {}

# Import enhanced integration from the migrated CA system
try:
    from integration.ml_ca_bridge import MLCABridge
    ML_BRIDGE_AVAILABLE = True
except ImportError:
    print("⚠️ MLCABridge not available, using fallback simulation")
    ML_BRIDGE_AVAILABLE = False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        import tensorflow as tf
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'tensorflow_version': tf.__version__,
            'gpu_available': gpu_available,
            'message': 'Fire simulation API is running'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    """
    Run a fire simulation.
    
    Expected JSON payload:
    {
        "ignition_points": [[lon, lat], ...],
        "weather_params": {
            "wind_direction": 45,
            "wind_speed": 15,
            "temperature": 30,
            "relative_humidity": 40
        },
        "simulation_hours": 6,
        "date": "2016-05-15",
        "use_ml_prediction": true
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['ignition_points', 'weather_params']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract parameters
        ignition_points = data['ignition_points']
        weather_params = data['weather_params']
        simulation_hours = data.get('simulation_hours', 6)
        simulation_date = data.get('date', '2016-05-15')
        use_ml_prediction = data.get('use_ml_prediction', True)
        
        # Generate unique simulation ID
        simulation_id = str(uuid.uuid4())[:8]
        
        # Store simulation info
        active_simulations[simulation_id] = {
            'status': 'running',
            'started_at': datetime.now().isoformat(),
            'parameters': data,
            'results': None
        }
        
        # Run simulation based on mode
        if use_ml_prediction:
            results = run_ml_ca_simulation(
                ignition_points, weather_params, simulation_hours, simulation_date, simulation_id
            )
        else:
            results = run_synthetic_simulation(
                ignition_points, weather_params, simulation_hours, simulation_id
            )
        
        if results:
            active_simulations[simulation_id]['status'] = 'completed'
            active_simulations[simulation_id]['results'] = results
            active_simulations[simulation_id]['completed_at'] = datetime.now().isoformat()
            
            return jsonify({
                'simulation_id': simulation_id,
                'status': 'completed',
                'results': {
                    'scenario_id': results.get('scenario_id'),
                    'total_hours': results.get('total_hours_simulated'),
                    'frame_paths': results.get('frame_paths', []),
                    'statistics': results.get('hourly_statistics', [])
                }
            })
        else:
            active_simulations[simulation_id]['status'] = 'failed'
            return jsonify({
                'simulation_id': simulation_id,
                'status': 'failed',
                'error': 'Simulation execution failed'
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulation/<simulation_id>/status', methods=['GET'])
def get_simulation_status(simulation_id):
    """Get simulation status and results."""
    if simulation_id not in active_simulations:
        return jsonify({'error': 'Simulation not found'}), 404
    
    sim_info = active_simulations[simulation_id]
    
    response = {
        'simulation_id': simulation_id,
        'status': sim_info['status'],
        'started_at': sim_info['started_at']
    }
    
    if sim_info['status'] == 'completed':
        response['completed_at'] = sim_info.get('completed_at')
        response['results'] = sim_info.get('results')
    
    return jsonify(response)

@app.route('/api/simulation/<simulation_id>/frame/<int:hour>', methods=['GET'])
def get_simulation_frame(simulation_id, hour):
    """Get a specific simulation frame as GeoTIFF."""
    if simulation_id not in active_simulations:
        return jsonify({'error': 'Simulation not found'}), 404
    
    sim_info = active_simulations[simulation_id]
    if sim_info['status'] != 'completed':
        return jsonify({'error': 'Simulation not completed'}), 400
    
    results = sim_info.get('results')
    if not results or 'frame_paths' not in results:
        return jsonify({'error': 'No frames available'}), 404
    
    frame_paths = results['frame_paths']
    if hour >= len(frame_paths):
        return jsonify({'error': f'Hour {hour} not available'}), 404
    
    frame_path = frame_paths[hour]
    if not os.path.exists(frame_path):
        return jsonify({'error': 'Frame file not found'}), 404
    
    return send_file(frame_path, as_attachment=True)

@app.route('/api/simulation/<simulation_id>/animation', methods=['GET'])
def get_simulation_animation(simulation_id):
    """Get simulation animation data as JSON."""
    if simulation_id not in active_simulations:
        return jsonify({'error': 'Simulation not found'}), 404
    
    sim_info = active_simulations[simulation_id]
    if sim_info['status'] != 'completed':
        return jsonify({'error': 'Simulation not completed'}), 400
    
    results = sim_info.get('results')
    if not results:
        return jsonify({'error': 'No results available'}), 404
    
    # Package animation data
    animation_data = {
        'simulation_id': simulation_id,
        'scenario_id': results.get('scenario_id'),
        'total_hours': results.get('total_hours_simulated'),
        'hourly_statistics': results.get('hourly_statistics', []),
        'frame_urls': [
            f'/api/simulation/{simulation_id}/frame/{i}' 
            for i in range(len(results.get('frame_paths', [])))
        ],
        'parameters': sim_info.get('parameters', {})
    }
    
    return jsonify(animation_data)

@app.route('/api/available_dates', methods=['GET'])
def get_available_dates():
    """Get list of dates with available ML predictions."""
    # Enhanced version with better formatting from duplicate
    try:
        if ML_BRIDGE_AVAILABLE:
            bridge = MLCABridge()
            dates = bridge.get_available_dates()
        else:
            # Fallback list for demo
            dates = ['2016_04_01', '2016_04_15', '2016_05_01', 
                    '2016_05_15', '2016_05_20', '2016_05_25']
        
        # Convert to more readable format (from duplicate folder)
        formatted_dates = []
        for date_str in dates:
            try:
                # Convert 2016_04_15 to readable format
                year, month, day = date_str.split('_')
                date_obj = datetime(int(year), int(month), int(day))
                formatted_dates.append({
                    'value': date_str,
                    'label': date_obj.strftime('%B %d, %Y'),
                    'iso': date_obj.isoformat(),
                    'short': date_obj.strftime('%m/%d/%Y')
                })
            except:
                # If parsing fails, use original format
                formatted_dates.append({
                    'value': date_str,
                    'label': date_str.replace('_', '-'),
                    'iso': date_str,
                    'short': date_str
                })
        
        return jsonify({
            'available_dates': formatted_dates,
            'date_range': {
                'start': '2016-04-01',
                'end': '2016-05-29'
            },
            'total_count': len(formatted_dates)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulation-cache/<simulation_id>', methods=['GET'])
def get_simulation_cache(simulation_id):
    """Get cached simulation results (from duplicate folder)."""
    if simulation_id not in active_simulations:
        return jsonify({'error': 'Simulation not found'}), 404
    
    sim_info = active_simulations[simulation_id]
    
    # Create cache-friendly response
    cached_data = {
        'simulation_id': simulation_id,
        'status': sim_info['status'],
        'parameters': sim_info.get('parameters', {}),
        'started_at': sim_info.get('started_at'),
        'completed_at': sim_info.get('completed_at'),
        'results_summary': None
    }
    
    if sim_info['status'] == 'completed' and sim_info.get('results'):
        results = sim_info['results']
        cached_data['results_summary'] = {
            'total_hours': results.get('total_hours_simulated'),
            'scenario_id': results.get('scenario_id'),
            'frame_count': len(results.get('frame_paths', [])),
            'final_statistics': results.get('hourly_statistics', [])[-1] if results.get('hourly_statistics') else None
        }
    
    return jsonify({
        'success': True,
        'cached_data': cached_data
    })

@app.route('/api/multiple-scenarios', methods=['POST'])
def run_multiple_scenarios():
    """Run multiple fire scenarios for comparison (from duplicate folder)."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['scenarios']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        scenarios = data['scenarios']
        base_weather = data.get('weather_params', {
            'wind_direction': 45,
            'wind_speed': 15,
            'temperature': 30,
            'relative_humidity': 40
        })
        
        # Validate scenarios format
        for i, scenario in enumerate(scenarios):
            if 'ignition_points' not in scenario:
                return jsonify({'error': f'Scenario {i+1} missing ignition_points'}), 400
        
        print(f"🎭 Running {len(scenarios)} scenarios for comparison")
        
        scenario_results = {}
        
        for i, scenario_config in enumerate(scenarios):
            scenario_id = f"comparison_{i+1}_{datetime.now().strftime('%H%M%S')}"
            
            # Use scenario-specific weather or base weather
            weather_params = scenario_config.get('weather_params', base_weather)
            ignition_points = scenario_config['ignition_points']
            simulation_hours = scenario_config.get('simulation_hours', 6)
            
            # Run individual scenario
            results = run_synthetic_simulation(
                ignition_points, weather_params, simulation_hours, scenario_id
            )
            
            if results:
                scenario_results[f"scenario_{i+1}"] = {
                    'config': scenario_config,
                    'results': results,
                    'scenario_id': scenario_id
                }
        
        # Create comparison summary
        comparison_summary = {
            'total_scenarios': len(scenario_results),
            'scenario_comparison': []
        }
        
        for scenario_id, scenario_data in scenario_results.items():
            stats = scenario_data['results'].get('hourly_statistics', [])
            final_stats = stats[-1] if stats else {}
            
            scenario_summary = {
                'scenario_id': scenario_id,
                'total_burned_area_km2': final_stats.get('burned_area_km2', 0),
                'max_fire_intensity': final_stats.get('max_intensity', 0),
                'total_burning_cells': final_stats.get('total_burning_cells', 0),
                'ignition_points_count': len(scenario_data['config']['ignition_points'])
            }
            comparison_summary['scenario_comparison'].append(scenario_summary)
        
        return jsonify({
            'success': True,
            'scenarios': scenario_results,
            'summary': comparison_summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Multiple scenarios failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_api_config():
    """Get API configuration for frontend (from duplicate folder)."""
    return jsonify({
        'api_version': '1.0.0',
        'max_simulation_hours': 24,
        'max_ignition_points': 10,
        'supported_formats': ['GeoTIFF', 'JSON'],
        'default_weather': {
            'wind_speed': 15.0,
            'wind_direction': 45.0,
            'temperature': 30.0,
            'relative_humidity': 40.0
        },
        'resolution_meters': 30,
        'coordinate_system': 'Geographic (WGS84)',
        'demo_scenarios': [
            'dehradun_fire', 'rishikesh_fire', 'nainital_fire'
        ],
        'features': {
            'ml_prediction': ML_BRIDGE_AVAILABLE,
            'synthetic_simulation': True,
            'multiple_scenarios': True,
            'animation_export': True,
            'real_time_stats': True
        }
    })

@app.route('/api/export-results/<simulation_id>', methods=['GET'])
def export_simulation_results(simulation_id):
    """Export simulation results as downloadable file (from duplicate folder)."""
    try:
        if simulation_id not in active_simulations:
            return jsonify({'error': 'Simulation not found'}), 404
        
        sim_info = active_simulations[simulation_id]
        
        if sim_info['status'] != 'completed':
            return jsonify({'error': 'Simulation not completed'}), 400
        
        results = sim_info.get('results')
        if not results:
            return jsonify({'error': 'No results available'}), 404
        
        # Create export package
        export_data = {
            'simulation_metadata': {
                'simulation_id': simulation_id,
                'scenario_id': results.get('scenario_id'),
                'created_at': sim_info.get('started_at'),
                'completed_at': sim_info.get('completed_at'),
                'parameters': sim_info.get('parameters', {})
            },
            'simulation_results': {
                'total_hours_simulated': results.get('total_hours_simulated'),
                'hourly_statistics': results.get('hourly_statistics', []),
                'frame_count': len(results.get('frame_paths', []))
            },
            'export_info': {
                'exported_at': datetime.now().isoformat(),
                'format': 'JSON',
                'version': '1.0.0'
            }
        }
        
        # For now, return JSON. In full implementation, create ZIP with GeoTIFFs
        return jsonify({
            'success': True,
            'export_format': 'JSON',
            'export_data': export_data,
            'download_url': f'/api/simulation/{simulation_id}/download'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Utility functions for the API

def validate_coordinates(x, y, max_x=400, max_y=400):
    """Validate ignition point coordinates (from duplicate folder)."""
    if not (0 <= x < max_x and 0 <= y < max_y):
        raise ValueError(f"Coordinates ({x}, {y}) out of bounds. Max: ({max_x}, {max_y})")
    return True

def cache_simulation_results(simulation_id, results):
    """Cache simulation results for faster retrieval (from duplicate folder)."""
    try:
        cache_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"simulation_{simulation_id}.json")
        
        # Create cache-friendly data structure
        cache_data = {
            'simulation_id': simulation_id,
            'cached_at': datetime.now().isoformat(),
            'results_summary': {
                'scenario_id': results.get('scenario_id'),
                'total_hours': results.get('total_hours_simulated'),
                'frame_count': len(results.get('frame_paths', [])),
                'statistics': results.get('hourly_statistics', [])
            }
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"💾 Cached simulation results for {simulation_id}")
        
    except Exception as e:
        print(f"⚠️ Failed to cache results: {str(e)}")

def run_ml_ca_simulation(ignition_points, weather_params, simulation_hours, date, simulation_id):
    """Run simulation using ML predictions."""
    try:
        from integration.ml_ca_bridge import MLCABridge
        
        bridge = MLCABridge()
        
        # For demo, use synthetic data if real ML data not available
        input_data_path = os.path.join(
            project_root, "dataset collection", f"stacked_data_{date.replace('-', '_')}.tif"
        )
        
        if not os.path.exists(input_data_path):
            print(f"⚠️ ML input data not found, using synthetic simulation")
            return run_synthetic_simulation(ignition_points, weather_params, simulation_hours, simulation_id)
        
        results = bridge.run_complete_pipeline(
            input_data_path=input_data_path,
            ignition_points=ignition_points,
            weather_params=weather_params,
            simulation_hours=simulation_hours,
            scenario_name=f"web_sim_{simulation_id}"
        )
        
        return results.get('ca_results') if results else None
        
    except Exception as e:
        print(f"❌ ML-CA simulation error: {e}")
        return None

def run_synthetic_simulation(ignition_points, weather_params, simulation_hours, simulation_id):
    """Run simulation using synthetic probability map."""
    try:
        from ca_engine import run_quick_simulation
        from test_ca_engine import create_synthetic_probability_map
        
        # Create synthetic probability map
        prob_map_path = os.path.join(
            app.config['OUTPUT_FOLDER'], f"synthetic_{simulation_id}.tif"
        )
        
        if not create_synthetic_probability_map(prob_map_path, width=300, height=300):
            return None
        
        # Run CA simulation
        results = run_quick_simulation(
            probability_map_path=prob_map_path,
            ignition_points=ignition_points,
            weather_params=weather_params,
            simulation_hours=simulation_hours,
            output_dir=os.path.join(app.config['OUTPUT_FOLDER'], f"web_sim_{simulation_id}")
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Synthetic simulation error: {e}")
        return None

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🔥 Starting Fire Simulation Web API")
    print("=" * 50)
    print(f"📊 Output directory: {app.config['OUTPUT_FOLDER']}")
    print(f"📁 Upload directory: {app.config['UPLOAD_FOLDER']}")
    
    # Run in development mode
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        threaded=True
    )
