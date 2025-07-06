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
    # For demo purposes, return a fixed list of dates
    # In full implementation, this would scan the actual data directory
    available_dates = [
        '2016-04-01', '2016-04-15', '2016-05-01', 
        '2016-05-15', '2016-05-20', '2016-05-25'
    ]
    
    return jsonify({
        'available_dates': available_dates,
        'date_range': {
            'start': '2016-04-01',
            'end': '2016-05-29'
        }
    })

@app.route('/api/demo/scenarios', methods=['GET'])
def get_demo_scenarios():
    """Get predefined demo scenarios."""
    demo_scenarios = [
        {
            'id': 'dehradun_fire',
            'name': 'Dehradun Valley Fire',
            'date': '2016-05-15',
            'location': 'Dehradun',
            'ignition_points': [[78.0322, 30.3165]],
            'weather': {
                'wind_direction': 225,
                'wind_speed': 18,
                'temperature': 32,
                'relative_humidity': 35
            },
            'description': 'Simulated fire spread in Dehradun valley during hot, dry conditions'
        },
        {
            'id': 'rishikesh_fire',
            'name': 'Rishikesh Forest Fire',
            'date': '2016-05-20',
            'location': 'Rishikesh',
            'ignition_points': [[78.2676, 30.0869]],
            'weather': {
                'wind_direction': 45,
                'wind_speed': 22,
                'temperature': 35,
                'relative_humidity': 25
            },
            'description': 'Forest fire near Rishikesh with northeast winds'
        },
        {
            'id': 'nainital_fire',
            'name': 'Nainital Hill Fire',
            'date': '2016-05-25',
            'location': 'Nainital',
            'ignition_points': [[79.4633, 29.3803]],
            'weather': {
                'wind_direction': 180,
                'wind_speed': 12,
                'temperature': 28,
                'relative_humidity': 45
            },
            'description': 'Hill fire in Nainital region with moderate conditions'
        }
    ]
    
    return jsonify({'demo_scenarios': demo_scenarios})

@app.route('/api/demo/run/<scenario_id>', methods=['POST'])
def run_demo_scenario(scenario_id):
    """Run a predefined demo scenario."""
    demo_scenarios = {
        'dehradun_fire': {
            'ignition_points': [[78.0322, 30.3165]],
            'weather_params': {
                'wind_direction': 225,
                'wind_speed': 18,
                'temperature': 32,
                'relative_humidity': 35
            },
            'date': '2016-05-15'
        },
        'rishikesh_fire': {
            'ignition_points': [[78.2676, 30.0869]],
            'weather_params': {
                'wind_direction': 45,
                'wind_speed': 22,
                'temperature': 35,
                'relative_humidity': 25
            },
            'date': '2016-05-20'
        },
        'nainital_fire': {
            'ignition_points': [[79.4633, 29.3803]],
            'weather_params': {
                'wind_direction': 180,
                'wind_speed': 12,
                'temperature': 28,
                'relative_humidity': 45
            },
            'date': '2016-05-25'
        }
    }
    
    if scenario_id not in demo_scenarios:
        return jsonify({'error': 'Demo scenario not found'}), 404
    
    scenario = demo_scenarios[scenario_id]
    
    # Add default parameters
    scenario['simulation_hours'] = 6
    scenario['use_ml_prediction'] = False  # Use synthetic for demo
    
    # Run simulation using the main simulation endpoint
    return run_simulation_with_data(scenario)

def run_simulation_with_data(data):
    """Helper function to run simulation with provided data."""
    try:
        ignition_points = data['ignition_points']
        weather_params = data['weather_params']
        simulation_hours = data.get('simulation_hours', 6)
        
        # Generate unique simulation ID
        simulation_id = str(uuid.uuid4())[:8]
        
        # Run synthetic simulation for demo
        results = run_synthetic_simulation(
            ignition_points, weather_params, simulation_hours, simulation_id
        )
        
        if results:
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
            return jsonify({
                'simulation_id': simulation_id,
                'status': 'failed',
                'error': 'Demo simulation failed'
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
