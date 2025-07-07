# ====================
# ML-CA Integration Bridge
# ====================
"""
Integration bridge between ML fire prediction model and CA simulation engine.
Handles data flow from ResUNet-A model outputs to CA simulation inputs.
"""

import os
import sys
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "forest_fire_ml", "fire_pred_model"))

class MLCABridge:
    """
    Bridge between ML model predictions and CA simulation.
    Handles prediction generation, data validation, and simulation orchestration.
    """
    
    def __init__(self, 
                 ml_model_path: str = None,
                 ca_output_dir: str = None):
        """
        Initialize the ML-CA bridge.
        
        Args:
            ml_model_path: Path to trained ML model
            ca_output_dir: Directory for CA simulation outputs
        """
        
        # Set default paths - use working implementation only
        if ml_model_path is None:
            possible_model_paths = [
                os.path.join(project_root, "forest_fire_ml", "fire_pred_model", "outputs", "final_model.h5"),
                os.path.join(project_root, "outputs", "final_model.h5")
            ]
            
            ml_model_path = None
            for path in possible_model_paths:
                if os.path.exists(path):
                    ml_model_path = path
                    break
            
            # If no model found, use the first path as default (for error reporting)
            if ml_model_path is None:
                ml_model_path = possible_model_paths[0]
        
        if ca_output_dir is None:
            ca_output_dir = os.path.join(project_root, "cellular_automata", "outputs")
        
        self.ml_model_path = ml_model_path
        self.ca_output_dir = ca_output_dir
        
        # Verify paths
        self.ml_available = os.path.exists(ml_model_path)
        os.makedirs(ca_output_dir, exist_ok=True)
        
        print(f"🌉 ML-CA Bridge initialized")
        print(f"   ML Model: {'✅ Available' if self.ml_available else '❌ Missing'} ({ml_model_path})")
        print(f"   CA Output: {ca_output_dir}")
    
    def generate_ml_prediction(self, 
                             input_tif_path: str, 
                             output_path: str = None,
                             date_str: str = None) -> Optional[str]:
        """
        Generate fire probability prediction using ML model.
        
        Args:
            input_tif_path: Path to stacked input data
            output_path: Path for probability map output
            date_str: Date string for naming (YYYY_MM_DD)
            
        Returns:
            Path to generated probability map or None if failed
        """
        
        if not self.ml_available:
            print("❌ ML model not available, cannot generate predictions")
            return None
        
        if not os.path.exists(input_tif_path):
            print(f"❌ Input data not found: {input_tif_path}")
            return None
        
        # Set output path
        if output_path is None:
            if date_str is None:
                date_str = datetime.now().strftime("%Y_%m_%d")
            output_path = os.path.join(
                self.ca_output_dir, f"probability_map_{date_str}.tif"
            )
        
        try:
            # Import ML prediction function from working implementation
            # The path is already added in the module initialization above
            from predict import predict_fire_probability
            
            print(f"🔮 Generating ML prediction...")
            print(f"   Input: {input_tif_path}")
            print(f"   Output: {output_path}")
            
            # Generate prediction
            prediction = predict_fire_probability(
                model_path=self.ml_model_path,
                input_tif_path=input_tif_path,
                output_path=output_path,
                patch_size=256,
                overlap=64
            )
            
            if prediction is not None and os.path.exists(output_path):
                print(f"✅ ML prediction generated: {output_path}")
                return output_path
            else:
                print("❌ ML prediction failed")
                return None
                
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            return None
    
    def run_ca_simulation(self, 
                         probability_map_path: str,
                         ignition_points: List[Tuple[float, float]],
                         weather_params: Dict[str, float],
                         simulation_hours: int = 6,
                         scenario_name: str = None) -> Optional[Dict]:
        """
        Run CA simulation using probability map.
        
        Args:
            probability_map_path: Path to ML-generated probability map
            ignition_points: List of (longitude, latitude) ignition points
            weather_params: Weather conditions
            simulation_hours: Duration of simulation
            scenario_name: Optional scenario name
            
        Returns:
            Simulation results or None if failed
        """
        
        if not os.path.exists(probability_map_path):
            print(f"❌ Probability map not found: {probability_map_path}")
            return None
        
        try:
            # Import CA engine
            from cellular_automata.ca_engine import run_quick_simulation
            
            print(f"🔥 Running CA simulation...")
            print(f"   Probability map: {probability_map_path}")
            print(f"   Ignition points: {len(ignition_points)}")
            print(f"   Duration: {simulation_hours} hours")
            
            # Set output directory
            if scenario_name is None:
                scenario_name = f"ml_ca_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            output_dir = os.path.join(self.ca_output_dir, scenario_name)
            
            # Run simulation
            results = run_quick_simulation(
                probability_map_path=probability_map_path,
                ignition_points=ignition_points,
                weather_params=weather_params,
                simulation_hours=simulation_hours,
                output_dir=output_dir
            )
            
            if results:
                print(f"✅ CA simulation completed: {results['scenario_id']}")
                return results
            else:
                print("❌ CA simulation failed")
                return None
                
        except Exception as e:
            print(f"❌ CA simulation error: {e}")
            return None
    
    def run_complete_pipeline(self,
                            input_data_path: str,
                            ignition_points: List[Tuple[float, float]],
                            weather_params: Dict[str, float],
                            simulation_hours: int = 6,
                            date_str: str = None,
                            scenario_name: str = None) -> Optional[Dict]:
        """
        Run complete ML→CA pipeline.
        
        Args:
            input_data_path: Path to stacked input data for ML model
            ignition_points: Fire ignition points
            weather_params: Weather conditions
            simulation_hours: Simulation duration
            date_str: Date string for naming
            scenario_name: Scenario name
            
        Returns:
            Complete pipeline results
        """
        
        print("🚀 Running complete ML→CA pipeline")
        print("=" * 50)
        
        # Step 1: Generate ML prediction
        print("\n📊 Step 1: ML Prediction Generation")
        probability_map_path = self.generate_ml_prediction(
            input_tif_path=input_data_path,
            date_str=date_str
        )
        
        if probability_map_path is None:
            return None
        
        # Step 2: Run CA simulation  
        print("\n🔥 Step 2: CA Simulation")
        ca_results = self.run_ca_simulation(
            probability_map_path=probability_map_path,
            ignition_points=ignition_points,
            weather_params=weather_params,
            simulation_hours=simulation_hours,
            scenario_name=scenario_name
        )
        
        if ca_results is None:
            return None
        
        # Step 3: Package complete results
        print("\n📦 Step 3: Results Packaging")
        complete_results = {
            'pipeline_id': f"ml_ca_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'input_data_path': input_data_path,
            'probability_map_path': probability_map_path,
            'ml_model_path': self.ml_model_path,
            'ca_results': ca_results,
            'simulation_parameters': {
                'ignition_points': ignition_points,
                'weather_params': weather_params,
                'simulation_hours': simulation_hours
            },
            'created_at': datetime.now().isoformat()
        }
        
        # Save complete results
        results_path = os.path.join(
            self.ca_output_dir, 
            f"{complete_results['pipeline_id']}_complete_results.json"
        )
        
        with open(results_path, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        print(f"✅ Complete pipeline results saved: {results_path}")
        
        return complete_results
    
    def validate_data_consistency(self, 
                                 input_data_path: str, 
                                 probability_map_path: str) -> bool:
        """
        Validate data consistency between input and probability map.
        
        Args:
            input_data_path: Original stacked data
            probability_map_path: Generated probability map
            
        Returns:
            True if data is consistent
        """
        
        try:
            import rasterio
            
            # Check input data
            with rasterio.open(input_data_path) as src_input:
                input_shape = (src_input.height, src_input.width)
                input_bounds = src_input.bounds
                input_crs = src_input.crs
            
            # Check probability map
            with rasterio.open(probability_map_path) as src_prob:
                prob_shape = (src_prob.height, src_prob.width)
                prob_bounds = src_prob.bounds
                prob_crs = src_prob.crs
                prob_data = src_prob.read(1)
            
            # Validate shape
            if input_shape != prob_shape:
                print(f"⚠️  Shape mismatch: input {input_shape} vs probability {prob_shape}")
                return False
            
            # Validate bounds (allow small differences due to floating point)
            bounds_diff = max(abs(a - b) for a, b in zip(input_bounds, prob_bounds))
            if bounds_diff > 0.001:
                print(f"⚠️  Bounds mismatch: difference {bounds_diff}")
                return False
            
            # Validate CRS
            if input_crs != prob_crs:
                print(f"⚠️  CRS mismatch: {input_crs} vs {prob_crs}")
                return False
            
            # Validate probability values
            if not (0 <= prob_data.min() <= prob_data.max() <= 1):
                print(f"⚠️  Invalid probability range: [{prob_data.min()}, {prob_data.max()}]")
                return False
            
            print("✅ Data consistency validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Data validation error: {e}")
            return False
    
    def create_demo_scenario(self, 
                           demo_date: str = "2016_05_15",
                           demo_location: str = "dehradun") -> Optional[Dict]:
        """
        Create a demo scenario for testing and presentation.
        
        Args:
            demo_date: Date for demo scenario
            demo_location: Location name for ignition points
            
        Returns:
            Demo scenario results
        """
        
        print(f"🎬 Creating demo scenario: {demo_location} on {demo_date}")
        
        # Demo ignition points (Dehradun area)
        demo_ignition_points = {
            'dehradun': [(78.0322, 30.3165)],  # Dehradun city
            'rishikesh': [(78.2676, 30.0869)], # Rishikesh
            'haridwar': [(78.1642, 29.9457)],  # Haridwar
            'nainital': [(79.4633, 29.3803)]   # Nainital
        }
        
        # Demo weather conditions (typical for May in Uttarakhand)
        demo_weather = {
            'wind_direction': 225,  # Southwest (typical pre-monsoon)
            'wind_speed': 18,       # 18 km/h
            'temperature': 32,      # 32°C (hot)
            'relative_humidity': 35 # 35% (dry)
        }
        
        ignition_points = demo_ignition_points.get(demo_location, demo_ignition_points['dehradun'])
        
        # Check for existing input data
        input_data_path = os.path.join(
            project_root, "dataset collection", f"stacked_data_{demo_date}.tif"
        )
        
        if not os.path.exists(input_data_path):
            print(f"⚠️  Demo input data not found: {input_data_path}")
            print("   Using synthetic data for demo...")
            
            # Create synthetic input data for demo
            return self._create_synthetic_demo(demo_date, demo_location, ignition_points, demo_weather)
        
        # Run with real data
        scenario_name = f"demo_{demo_location}_{demo_date}"
        
        results = self.run_complete_pipeline(
            input_data_path=input_data_path,
            ignition_points=ignition_points,
            weather_params=demo_weather,
            simulation_hours=6,
            date_str=demo_date,
            scenario_name=scenario_name
        )
        
        return results
    
    def _create_synthetic_demo(self, 
                              demo_date: str, 
                              demo_location: str,
                              ignition_points: List[Tuple[float, float]],
                              weather_params: Dict[str, float]) -> Optional[Dict]:
        """Create synthetic demo scenario."""
        
        try:
            # Create synthetic probability map
            from cellular_automata.test_ca_engine import create_synthetic_probability_map
            
            prob_map_path = os.path.join(
                self.ca_output_dir, f"synthetic_demo_{demo_location}_{demo_date}.tif"
            )
            
            if create_synthetic_probability_map(prob_map_path, width=400, height=400):
                scenario_name = f"synthetic_demo_{demo_location}_{demo_date}"
                
                ca_results = self.run_ca_simulation(
                    probability_map_path=prob_map_path,
                    ignition_points=ignition_points,
                    weather_params=weather_params,
                    simulation_hours=6,
                    scenario_name=scenario_name
                )
                
                if ca_results:
                    # Wrap in the expected complete results structure
                    complete_results = {
                        'pipeline_id': f"synthetic_demo_{demo_location}_{demo_date}_{datetime.now().strftime('%H%M%S')}",
                        'input_data_path': 'synthetic',
                        'probability_map_path': prob_map_path,
                        'ml_model_path': 'synthetic',
                        'ca_results': ca_results,
                        'simulation_parameters': {
                            'ignition_points': ignition_points,
                            'weather_params': weather_params,
                            'simulation_hours': 6
                        },
                        'created_at': datetime.now().isoformat()
                    }
                    return complete_results
                else:
                    return None
            
        except Exception as e:
            print(f"❌ Synthetic demo creation failed: {e}")
        
        return None

# Convenience functions for quick access
def quick_ml_ca_simulation(input_data_path: str,
                          ignition_lat: float,
                          ignition_lon: float,
                          wind_direction: float = 45,
                          wind_speed: float = 15,
                          simulation_hours: int = 6) -> Optional[Dict]:
    """
    Quick ML→CA simulation with minimal parameters.
    
    Args:
        input_data_path: Path to stacked input data
        ignition_lat, ignition_lon: Ignition point coordinates
        wind_direction: Wind direction in degrees
        wind_speed: Wind speed in km/h
        simulation_hours: Simulation duration
        
    Returns:
        Simulation results
    """
    
    bridge = MLCABridge()
    
    weather_params = {
        'wind_direction': wind_direction,
        'wind_speed': wind_speed,
        'temperature': 30,
        'relative_humidity': 40
    }
    
    return bridge.run_complete_pipeline(
        input_data_path=input_data_path,
        ignition_points=[(ignition_lon, ignition_lat)],
        weather_params=weather_params,
        simulation_hours=simulation_hours
    )

def create_demo_scenarios() -> List[Dict]:
    """Create multiple demo scenarios for presentation."""
    
    bridge = MLCABridge()
    
    demo_configs = [
        ('dehradun', '2016_05_15'),
        ('rishikesh', '2016_05_20'), 
        ('nainital', '2016_05_25')
    ]
    
    results = []
    
    for location, date in demo_configs:
        print(f"\n🎬 Creating demo: {location} on {date}")
        result = bridge.create_demo_scenario(date, location)
        if result:
            results.append(result)
    
    return results

if __name__ == "__main__":
    # Test the ML-CA bridge
    print("🌉 Testing ML-CA Integration Bridge")
    print("=" * 60)
    
    bridge = MLCABridge()
    
    # Create a demo scenario
    demo_result = bridge.create_demo_scenario()
    
    if demo_result:
        print("\n✅ Demo scenario created successfully!")
        ca_results = demo_result.get('ca_results', {})
        print(f"   Scenario ID: {ca_results.get('scenario_id', 'N/A')}")
        print(f"   Hours simulated: {ca_results.get('total_hours_simulated', 'N/A')}")
    else:
        print("\n❌ Demo scenario creation failed")
    
    print(f"\n🎯 ML-CA Bridge test completed")
