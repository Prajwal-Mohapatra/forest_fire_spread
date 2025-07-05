"""
Model versioning and metadata tracking utilities for forest fire prediction.
"""
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
import tensorflow as tf

class ModelVersionManager:
    """Manages model versions, metadata, and checkpoints."""
    
    def __init__(self, base_dir: str = "outputs"):
        self.base_dir = base_dir
        self.versions_dir = os.path.join(base_dir, "versions")
        self.metadata_file = os.path.join(self.versions_dir, "version_registry.json")
        
        os.makedirs(self.versions_dir, exist_ok=True)
        self._load_registry()
    
    def _load_registry(self):
        """Load existing version registry or create new one."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "versions": {},
                "latest_version": None,
                "creation_date": datetime.now().isoformat()
            }
    
    def _save_registry(self):
        """Save version registry to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def _generate_version_id(self, metadata: Dict[str, Any]) -> str:
        """Generate unique version ID based on configuration hash."""
        config_str = json.dumps(metadata.get('config', {}), sort_keys=True)
        hash_obj = hashlib.md5(config_str.encode())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}_{hash_obj.hexdigest()[:8]}"
    
    def save_model_version(self, 
                          model: tf.keras.Model,
                          metadata: Dict[str, Any],
                          version_id: Optional[str] = None) -> str:
        """
        Save a new model version with comprehensive metadata.
        
        Args:
            model: Trained Keras model
            metadata: Dictionary containing training configuration, metrics, etc.
            version_id: Optional custom version ID
            
        Returns:
            str: Version ID of saved model
        """
        if version_id is None:
            version_id = self._generate_version_id(metadata)
        
        version_dir = os.path.join(self.versions_dir, version_id)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model weights and architecture
        model_path = os.path.join(version_dir, "model.h5")
        model.save_weights(model_path)
        
        # Save model architecture
        architecture_path = os.path.join(version_dir, "architecture.json")
        with open(architecture_path, 'w') as f:
            f.write(model.to_json())
        
        # Enhance metadata with system info
        enhanced_metadata = {
            **metadata,
            "version_id": version_id,
            "save_timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "architecture_path": architecture_path,
            "total_params": model.count_params(),
            "trainable_params": sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            "tensorflow_version": tf.__version__,
        }
        
        # Save detailed metadata
        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2, default=str)
        
        # Update registry
        self.registry["versions"][version_id] = enhanced_metadata
        self.registry["latest_version"] = version_id
        self._save_registry()
        
        print(f"✅ Model version {version_id} saved successfully")
        print(f"📁 Location: {version_dir}")
        return version_id
    
    def load_model_version(self, version_id: str = None) -> tuple:
        """
        Load a specific model version.
        
        Args:
            version_id: Version to load (defaults to latest)
            
        Returns:
            tuple: (model, metadata)
        """
        if version_id is None:
            version_id = self.registry.get("latest_version")
            if version_id is None:
                raise ValueError("No model versions found")
        
        if version_id not in self.registry["versions"]:
            raise ValueError(f"Version {version_id} not found")
        
        metadata = self.registry["versions"][version_id]
        version_dir = os.path.join(self.versions_dir, version_id)
        
        # Load architecture
        architecture_path = os.path.join(version_dir, "architecture.json")
        with open(architecture_path, 'r') as f:
            model = tf.keras.models.model_from_json(f.read())
        
        # Load weights
        model_path = os.path.join(version_dir, "model.h5")
        model.load_weights(model_path)
        
        print(f"✅ Loaded model version {version_id}")
        return model, metadata
    
    def list_versions(self) -> Dict[str, Any]:
        """List all available model versions."""
        return self.registry["versions"]
    
    def compare_versions(self, version_ids: list) -> Dict[str, Any]:
        """Compare metrics across multiple versions."""
        comparison = {}
        
        for vid in version_ids:
            if vid in self.registry["versions"]:
                metadata = self.registry["versions"][vid]
                comparison[vid] = {
                    "save_timestamp": metadata.get("save_timestamp"),
                    "metrics": metadata.get("metrics", {}),
                    "config": metadata.get("config", {}),
                    "total_params": metadata.get("total_params")
                }
        
        return comparison
    
    def delete_version(self, version_id: str, confirm: bool = False):
        """Delete a specific model version."""
        if not confirm:
            print("⚠️ Use confirm=True to actually delete the version")
            return
        
        if version_id not in self.registry["versions"]:
            print(f"Version {version_id} not found")
            return
        
        version_dir = os.path.join(self.versions_dir, version_id)
        
        # Remove files
        import shutil
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
        
        # Update registry
        del self.registry["versions"][version_id]
        if self.registry["latest_version"] == version_id:
            # Set new latest version
            remaining_versions = list(self.registry["versions"].keys())
            self.registry["latest_version"] = remaining_versions[-1] if remaining_versions else None
        
        self._save_registry()
        print(f"🗑️ Version {version_id} deleted successfully")
    
    def get_best_version(self, metric: str = "val_iou_score") -> Optional[str]:
        """Find the version with the best performance on a given metric."""
        best_version = None
        best_score = -1
        
        for version_id, metadata in self.registry["versions"].items():
            metrics = metadata.get("metrics", {})
            if metric in metrics:
                score = metrics[metric]
                if isinstance(score, list):
                    score = max(score)  # Take best epoch score
                
                if score > best_score:
                    best_score = score
                    best_version = version_id
        
        return best_version
    
    def print_version_summary(self):
        """Print a summary of all model versions."""
        print("\n" + "="*70)
        print("MODEL VERSION REGISTRY")
        print("="*70)
        
        if not self.registry["versions"]:
            print("No model versions found.")
            return
        
        print(f"Total Versions: {len(self.registry['versions'])}")
        print(f"Latest Version: {self.registry.get('latest_version', 'None')}")
        print(f"Registry Created: {self.registry.get('creation_date', 'Unknown')}")
        
        print("\nVERSION DETAILS:")
        print("-" * 70)
        
        for version_id, metadata in self.registry["versions"].items():
            print(f"🔹 {version_id}")
            print(f"   Created: {metadata.get('save_timestamp', 'Unknown')}")
            print(f"   Parameters: {metadata.get('total_params', 'Unknown'):,}")
            
            metrics = metadata.get("metrics", {})
            if metrics:
                print(f"   Best Metrics: IoU={metrics.get('val_iou_score', [0])[-1]:.4f}, "
                      f"Dice={metrics.get('val_dice_coef', [0])[-1]:.4f}")
            
            config = metadata.get("config", {})
            if config:
                print(f"   Loss: {config.get('loss_function', 'Unknown')}")
                print(f"   Optimizer: {config.get('optimizer', 'Unknown')}")
            print()
        
        print("="*70)
