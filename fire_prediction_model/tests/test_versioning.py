"""
Unit tests for model versioning and metadata tracking.
Tests version creation, metadata saving/loading, and model management.
"""

import unittest
import tempfile
import shutil
import os
import json
import sys
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.versioning import ModelVersionManager, create_model_version, save_model_metadata
except ImportError:
    # Skip if versioning module not available
    raise unittest.SkipTest("Versioning module not available")


class TestModelVersionManager(unittest.TestCase):
    """Test ModelVersionManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.version_manager = ModelVersionManager(base_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test version manager initialization."""
        self.assertEqual(self.version_manager.base_dir, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_create_version_basic(self):
        """Test basic version creation."""
        version_info = {
            'model_name': 'ResUNet-A',
            'architecture': 'ResUNet-A with ASPP',
            'dataset': 'forest_fire_2016',
            'performance': {'iou': 0.85, 'dice': 0.90}
        }
        
        version_id = self.version_manager.create_version(
            model_name='test_model',
            version_info=version_info
        )
        
        # Check version ID format
        self.assertIsInstance(version_id, str)
        self.assertTrue(len(version_id) > 0)
        
        # Check that version directory was created
        version_dir = os.path.join(self.temp_dir, version_id)
        self.assertTrue(os.path.exists(version_dir))
        
        # Check that metadata file was created
        metadata_file = os.path.join(version_dir, 'metadata.json')
        self.assertTrue(os.path.exists(metadata_file))
    
    def test_create_version_with_model(self):
        """Test version creation with model saving."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.save = MagicMock()
        
        version_info = {
            'model_name': 'ResUNet-A',
            'epochs': 30,
            'batch_size': 8
        }
        
        version_id = self.version_manager.create_version(
            model_name='test_model',
            version_info=version_info,
            model=mock_model
        )
        
        # Check that model.save was called
        mock_model.save.assert_called_once()
        
        # Check directory structure
        version_dir = os.path.join(self.temp_dir, version_id)
        self.assertTrue(os.path.exists(version_dir))
    
    def test_save_metadata(self):
        """Test metadata saving."""
        version_id = 'test_version_001'
        version_dir = os.path.join(self.temp_dir, version_id)
        os.makedirs(version_dir)
        
        metadata = {
            'model_name': 'TestModel',
            'created_at': datetime.now().isoformat(),
            'performance': {'accuracy': 0.95}
        }
        
        self.version_manager.save_metadata(version_id, metadata)
        
        # Check that metadata file exists
        metadata_file = os.path.join(version_dir, 'metadata.json')
        self.assertTrue(os.path.exists(metadata_file))
        
        # Check metadata content
        with open(metadata_file, 'r') as f:
            saved_metadata = json.load(f)
        
        self.assertEqual(saved_metadata['model_name'], 'TestModel')
        self.assertIn('performance', saved_metadata)
    
    def test_load_metadata(self):
        """Test metadata loading."""
        version_id = 'test_version_002'
        version_dir = os.path.join(self.temp_dir, version_id)
        os.makedirs(version_dir)
        
        # Create metadata
        metadata = {
            'model_name': 'LoadTestModel',
            'version': '1.0.0',
            'metrics': {'loss': 0.1, 'accuracy': 0.9}
        }
        
        metadata_file = os.path.join(version_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Load metadata
        loaded_metadata = self.version_manager.load_metadata(version_id)
        
        self.assertEqual(loaded_metadata['model_name'], 'LoadTestModel')
        self.assertEqual(loaded_metadata['version'], '1.0.0')
        self.assertIn('metrics', loaded_metadata)
    
    def test_list_versions(self):
        """Test listing available versions."""
        # Create multiple versions
        version_ids = []
        for i in range(3):
            version_info = {'model_name': f'model_{i}'}
            version_id = self.version_manager.create_version(
                model_name=f'test_model_{i}',
                version_info=version_info
            )
            version_ids.append(version_id)
        
        # List versions
        versions = self.version_manager.list_versions()
        
        self.assertEqual(len(versions), 3)
        for version_id in version_ids:
            self.assertIn(version_id, versions)
    
    def test_get_latest_version(self):
        """Test getting latest version."""
        # Create versions with timestamps
        version_ids = []
        for i in range(3):
            version_info = {
                'model_name': f'model_{i}',
                'created_at': datetime.now().isoformat()
            }
            version_id = self.version_manager.create_version(
                model_name=f'test_model_{i}',
                version_info=version_info
            )
            version_ids.append(version_id)
        
        # Get latest version
        latest_version = self.version_manager.get_latest_version()
        
        self.assertIsNotNone(latest_version)
        self.assertIn(latest_version, version_ids)
    
    def test_delete_version(self):
        """Test version deletion."""
        # Create version
        version_info = {'model_name': 'DeleteTestModel'}
        version_id = self.version_manager.create_version(
            model_name='delete_test',
            version_info=version_info
        )
        
        # Verify it exists
        version_dir = os.path.join(self.temp_dir, version_id)
        self.assertTrue(os.path.exists(version_dir))
        
        # Delete version
        self.version_manager.delete_version(version_id)
        
        # Verify it's deleted
        self.assertFalse(os.path.exists(version_dir))
    
    def test_compare_versions(self):
        """Test version comparison."""
        # Create two versions
        version_info_1 = {
            'model_name': 'Model_v1',
            'performance': {'iou': 0.8, 'dice': 0.85}
        }
        version_id_1 = self.version_manager.create_version(
            model_name='compare_test_1',
            version_info=version_info_1
        )
        
        version_info_2 = {
            'model_name': 'Model_v2',
            'performance': {'iou': 0.85, 'dice': 0.90}
        }
        version_id_2 = self.version_manager.create_version(
            model_name='compare_test_2',
            version_info=version_info_2
        )
        
        # Compare versions
        comparison = self.version_manager.compare_versions(version_id_1, version_id_2)
        
        self.assertIsInstance(comparison, dict)
        self.assertIn('version_1', comparison)
        self.assertIn('version_2', comparison)
        self.assertIn('differences', comparison)


class TestVersioningFunctions(unittest.TestCase):
    """Test standalone versioning functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_model_version_basic(self):
        """Test basic model version creation function."""
        config = {
            'model': {'input_shape': [256, 256, 9], 'num_classes': 1},
            'training': {'epochs': 30, 'batch_size': 8}
        }
        
        training_history = {
            'loss': [0.5, 0.3, 0.2],
            'val_loss': [0.6, 0.4, 0.3],
            'iou_score': [0.7, 0.8, 0.85]
        }
        
        metrics = {
            'iou': 0.85,
            'dice': 0.90,
            'f1': 0.88
        }
        
        version_id = create_model_version(
            base_dir=self.temp_dir,
            model_name='test_function_model',
            config=config,
            training_history=training_history,
            metrics=metrics
        )
        
        # Check return value
        self.assertIsInstance(version_id, str)
        
        # Check directory creation
        version_dir = os.path.join(self.temp_dir, version_id)
        self.assertTrue(os.path.exists(version_dir))
        
        # Check metadata file
        metadata_file = os.path.join(version_dir, 'metadata.json')
        self.assertTrue(os.path.exists(metadata_file))
    
    @patch('platform.system')
    @patch('platform.processor')
    @patch('psutil.virtual_memory')
    def test_save_model_metadata_with_system_info(self, mock_memory, mock_processor, mock_system):
        """Test metadata saving with system information."""
        # Mock system information
        mock_system.return_value = 'Linux'
        mock_processor.return_value = 'x86_64'
        mock_memory.return_value = MagicMock(total=17179869184)  # 16GB
        
        version_dir = os.path.join(self.temp_dir, 'test_metadata')
        os.makedirs(version_dir)
        
        metadata = {
            'model_name': 'SystemInfoTest',
            'performance': {'accuracy': 0.95}
        }
        
        save_model_metadata(version_dir, metadata)
        
        # Check metadata file
        metadata_file = os.path.join(version_dir, 'metadata.json')
        self.assertTrue(os.path.exists(metadata_file))
        
        # Load and check content
        with open(metadata_file, 'r') as f:
            saved_metadata = json.load(f)
        
        self.assertIn('system_info', saved_metadata)
        self.assertIn('os', saved_metadata['system_info'])
        self.assertIn('processor', saved_metadata['system_info'])
        self.assertIn('memory_gb', saved_metadata['system_info'])
    
    def test_version_id_format(self):
        """Test version ID format consistency."""
        config = {'model': {'input_shape': [256, 256, 9]}}
        
        # Create multiple versions
        version_ids = []
        for i in range(5):
            version_id = create_model_version(
                base_dir=self.temp_dir,
                model_name=f'format_test_{i}',
                config=config
            )
            version_ids.append(version_id)
        
        # Check format consistency
        for version_id in version_ids:
            # Should be string
            self.assertIsInstance(version_id, str)
            
            # Should have reasonable length
            self.assertGreater(len(version_id), 10)
            self.assertLess(len(version_id), 50)
            
            # Should be unique
            other_ids = [vid for vid in version_ids if vid != version_id]
            self.assertNotIn(version_id, other_ids)


class TestVersioningIntegration(unittest.TestCase):
    """Integration tests for versioning system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.version_manager = ModelVersionManager(base_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_workflow(self):
        """Test complete versioning workflow."""
        # Create initial version
        version_info_1 = {
            'model_name': 'ResUNet-A',
            'architecture': 'ResUNet-A with ASPP',
            'epochs': 25,
            'performance': {'iou': 0.80, 'dice': 0.85}
        }
        
        version_id_1 = self.version_manager.create_version(
            model_name='workflow_test',
            version_info=version_info_1
        )
        
        # Create improved version
        version_info_2 = {
            'model_name': 'ResUNet-A-Enhanced',
            'architecture': 'ResUNet-A with Enhanced ASPP',
            'epochs': 30,
            'performance': {'iou': 0.85, 'dice': 0.90}
        }
        
        version_id_2 = self.version_manager.create_version(
            model_name='workflow_test_v2',
            version_info=version_info_2
        )
        
        # List versions
        versions = self.version_manager.list_versions()
        self.assertEqual(len(versions), 2)
        self.assertIn(version_id_1, versions)
        self.assertIn(version_id_2, versions)
        
        # Get latest version
        latest = self.version_manager.get_latest_version()
        self.assertEqual(latest, version_id_2)
        
        # Compare versions
        comparison = self.version_manager.compare_versions(version_id_1, version_id_2)
        self.assertIn('differences', comparison)
        
        # Load metadata
        metadata_1 = self.version_manager.load_metadata(version_id_1)
        metadata_2 = self.version_manager.load_metadata(version_id_2)
        
        self.assertEqual(metadata_1['epochs'], 25)
        self.assertEqual(metadata_2['epochs'], 30)
        
        # Clean up one version
        self.version_manager.delete_version(version_id_1)
        
        versions_after = self.version_manager.list_versions()
        self.assertEqual(len(versions_after), 1)
        self.assertNotIn(version_id_1, versions_after)
        self.assertIn(version_id_2, versions_after)
    
    def test_error_handling(self):
        """Test error handling in versioning system."""
        # Test loading non-existent version
        with self.assertRaises(FileNotFoundError):
            self.version_manager.load_metadata('non_existent_version')
        
        # Test deleting non-existent version
        with self.assertRaises(FileNotFoundError):
            self.version_manager.delete_version('non_existent_version')
        
        # Test comparing with non-existent version
        version_info = {'model_name': 'ErrorTestModel'}
        version_id = self.version_manager.create_version(
            model_name='error_test',
            version_info=version_info
        )
        
        with self.assertRaises(FileNotFoundError):
            self.version_manager.compare_versions(version_id, 'non_existent_version')


if __name__ == '__main__':
    unittest.main()
