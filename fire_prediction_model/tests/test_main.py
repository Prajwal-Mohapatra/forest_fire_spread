"""
Unit tests for the forest fire prediction model.
Tests critical functions including metrics, preprocessing, model components, and utilities.
"""

import unittest
import numpy as np
import tensorflow as tf
import tempfile
import os
import sys
import yaml
from unittest.mock import Mock, patch

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import (
    iou_score, dice_coef, precision_score, recall_score, f1_score,
    focal_loss, combined_loss, comprehensive_evaluation, print_evaluation_report,
    compute_class_weights
)
from dataset.preprocess import normalize_patch
from model.resunet_a import (
    build_resunet_a, residual_block, atrous_spatial_pyramid_pooling, 
    improved_attention_gate
)
from utils.versioning import ModelVersionManager


class TestMetrics(unittest.TestCase):
    """Test suite for metric functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample binary segmentation data
        self.y_true = tf.constant([
            [[[1], [0], [1], [0]],
             [[0], [1], [0], [1]],
             [[1], [1], [0], [0]],
             [[0], [0], [1], [1]]]
        ], dtype=tf.float32)
        
        self.y_pred = tf.constant([
            [[[0.9], [0.1], [0.8], [0.2]],
             [[0.1], [0.9], [0.3], [0.7]],
             [[0.8], [0.9], [0.2], [0.1]],
             [[0.1], [0.2], [0.8], [0.9]]]
        ], dtype=tf.float32)
        
        # Ground truth binary masks for testing
        self.y_true_binary = tf.constant([
            [[[1], [0], [1], [0]],
             [[0], [1], [0], [1]],
             [[1], [1], [0], [0]],
             [[0], [0], [1], [1]]]
        ], dtype=tf.float32)
        
        self.y_pred_binary = tf.cast(self.y_pred > 0.5, tf.float32)
    
    def test_iou_score(self):
        """Test IoU score calculation."""
        iou = iou_score(self.y_true, self.y_pred)
        self.assertIsInstance(iou.numpy(), (float, np.float32, np.float64))
        self.assertGreaterEqual(iou.numpy(), 0.0)
        self.assertLessEqual(iou.numpy(), 1.0)
        
        # Test perfect prediction
        perfect_iou = iou_score(self.y_true, self.y_true)
        self.assertAlmostEqual(perfect_iou.numpy(), 1.0, places=5)
    
    def test_dice_coefficient(self):
        """Test Dice coefficient calculation."""
        dice = dice_coef(self.y_true, self.y_pred)
        self.assertIsInstance(dice.numpy(), (float, np.float32, np.float64))
        self.assertGreaterEqual(dice.numpy(), 0.0)
        self.assertLessEqual(dice.numpy(), 1.0)
        
        # Test perfect prediction
        perfect_dice = dice_coef(self.y_true, self.y_true)
        self.assertAlmostEqual(perfect_dice.numpy(), 1.0, places=5)
    
    def test_precision_score(self):
        """Test precision score calculation."""
        precision = precision_score(self.y_true, self.y_pred)
        self.assertIsInstance(precision.numpy(), (float, np.float32, np.float64))
        self.assertGreaterEqual(precision.numpy(), 0.0)
        self.assertLessEqual(precision.numpy(), 1.0)
    
    def test_recall_score(self):
        """Test recall score calculation."""
        recall = recall_score(self.y_true, self.y_pred)
        self.assertIsInstance(recall.numpy(), (float, np.float32, np.float64))
        self.assertGreaterEqual(recall.numpy(), 0.0)
        self.assertLessEqual(recall.numpy(), 1.0)
    
    def test_f1_score(self):
        """Test F1 score calculation."""
        f1 = f1_score(self.y_true, self.y_pred)
        self.assertIsInstance(f1.numpy(), (float, np.float32, np.float64))
        self.assertGreaterEqual(f1.numpy(), 0.0)
        self.assertLessEqual(f1.numpy(), 1.0)
    
    def test_focal_loss(self):
        """Test focal loss function."""
        loss_fn = focal_loss(alpha=0.25, gamma=2.0)
        loss = loss_fn(self.y_true, self.y_pred)
        self.assertIsInstance(loss.numpy(), (float, np.float32, np.float64))
        self.assertGreaterEqual(loss.numpy(), 0.0)
    
    def test_combined_loss(self):
        """Test combined focal + dice loss function."""
        loss_fn = combined_loss(alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.5)
        loss = loss_fn(self.y_true, self.y_pred)
        self.assertIsInstance(loss.numpy(), (float, np.float32, np.float64))
        self.assertGreaterEqual(loss.numpy(), 0.0)


class TestPreprocessing(unittest.TestCase):
    """Test suite for preprocessing functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample multi-band image data
        self.sample_patch = np.random.rand(256, 256, 9).astype(np.float32)
        self.sample_patch_denormalized = self.sample_patch * 1000 + 500  # Simulate real world values
    
    def test_normalize_patch_basic(self):
        """Test basic normalization functionality."""
        normalized = normalize_patch(self.sample_patch_denormalized)
        
        # Check shape preservation
        self.assertEqual(normalized.shape, self.sample_patch_denormalized.shape)
        
        # Check that values are in reasonable range (0-1 for min-max normalization)
        self.assertGreaterEqual(normalized.min(), 0.0)
        self.assertLessEqual(normalized.max(), 1.0)
    
    def test_normalize_patch_edge_cases(self):
        """Test normalization with edge cases."""
        # Test with all zeros
        zero_patch = np.zeros((256, 256, 9), dtype=np.float32)
        normalized_zero = normalize_patch(zero_patch)
        self.assertEqual(normalized_zero.shape, zero_patch.shape)
        
        # Test with constant values
        constant_patch = np.ones((256, 256, 9), dtype=np.float32) * 100
        normalized_constant = normalize_patch(constant_patch)
        self.assertEqual(normalized_constant.shape, constant_patch.shape)
    
    def test_normalize_patch_per_channel(self):
        """Test per-channel normalization."""
        # Create patch with different value ranges per channel
        test_patch = np.random.rand(256, 256, 9)
        for i in range(9):
            test_patch[:, :, i] = test_patch[:, :, i] * (i + 1) * 1000
        
        normalized = normalize_patch(test_patch)
        
        # Each channel should be normalized independently
        for i in range(9):
            channel_min = normalized[:, :, i].min()
            channel_max = normalized[:, :, i].max()
            self.assertGreaterEqual(channel_min, 0.0)
            self.assertLessEqual(channel_max, 1.0)


class TestModelArchitecture(unittest.TestCase):
    """Test suite for model architecture components."""
    
    def setUp(self):
        """Set up test parameters."""
        self.input_shape = (256, 256, 9)
        self.num_classes = 1
    
    def test_resunet_a_model_creation(self):
        """Test ResUNet-A model creation."""
        model = build_resunet_a(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            use_enhanced_aspp=False
        )
        
        # Check model properties
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None,) + self.input_shape)
        self.assertEqual(model.output_shape, (None, 256, 256, 1))
        
        # Check that model has reasonable number of parameters
        total_params = model.count_params()
        self.assertGreater(total_params, 1000)  # Should have substantial parameters
        self.assertLess(total_params, 100_000_000)  # But not excessive
    
    def test_resunet_a_with_enhanced_aspp(self):
        """Test ResUNet-A model with enhanced ASPP."""
        model = build_resunet_a(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            use_enhanced_aspp=True
        )
        
        self.assertIsInstance(model, tf.keras.Model)
        
        # Both enhanced and basic models should have substantial parameters
        model_basic = build_resunet_a(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            use_enhanced_aspp=False
        )
        
        self.assertGreater(model.count_params(), 1000)
        self.assertGreater(model_basic.count_params(), 1000)
    
    def test_attention_gate(self):
        """Test Attention Gate component."""
        # Create test inputs
        g = tf.random.normal((1, 64, 64, 256))  # Gating signal
        x = tf.random.normal((1, 64, 64, 256))  # Feature map
        
        output = improved_attention_gate(x, g, inter_channels=128)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check that attention weights are applied (output should be different from input)
        self.assertFalse(tf.reduce_all(tf.equal(output, x)))
    
    def test_aspp_module(self):
        """Test ASPP module."""
        # Create test input
        x = tf.random.normal((1, 32, 32, 512))
        
        output = atrous_spatial_pyramid_pooling(
            x,
            output_filters=256,
            rates=[6, 12, 18],
            use_global_pooling=True
        )
        
        # Check output shape
        self.assertEqual(output.shape, (1, 32, 32, 256))
    
    def test_model_prediction(self):
        """Test model prediction with sample input."""
        model = build_resunet_a(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        
        # Create sample input
        sample_input = np.random.rand(1, *self.input_shape)
        
        # Test prediction
        prediction = model.predict(sample_input, verbose=0)
        
        # Check prediction shape and range
        self.assertEqual(prediction.shape, (1, 256, 256, 1))
        self.assertTrue(np.all(prediction >= 0))
        self.assertTrue(np.all(prediction <= 1))


class TestVersioning(unittest.TestCase):
    """Test suite for model versioning system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.version_manager = ModelVersionManager(base_dir=self.temp_dir)
        
        # Create a simple test model
        self.test_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.test_metadata = {
            'config': {'test': True},
            'metrics': {'accuracy': 0.95, 'loss': 0.1},
            'optimizer': 'Adam',
            'learning_rate': 0.001
        }
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_model_version(self):
        """Test saving a model version."""
        version_id = self.version_manager.save_model_version(
            self.test_model, 
            self.test_metadata
        )
        
        # Check that version ID is generated
        self.assertIsInstance(version_id, str)
        self.assertTrue(version_id.startswith('v_'))
        
        # Check that files are created
        version_dir = os.path.join(self.temp_dir, "versions", version_id)
        self.assertTrue(os.path.exists(version_dir))
        self.assertTrue(os.path.exists(os.path.join(version_dir, "model.h5")))
        self.assertTrue(os.path.exists(os.path.join(version_dir, "architecture.json")))
        self.assertTrue(os.path.exists(os.path.join(version_dir, "metadata.json")))
    
    def test_load_model_version(self):
        """Test loading a model version."""
        # Save a version first
        version_id = self.version_manager.save_model_version(
            self.test_model, 
            self.test_metadata
        )
        
        # Load the version
        loaded_model, loaded_metadata = self.version_manager.load_model_version(version_id)
        
        # Check that model is loaded correctly
        self.assertIsInstance(loaded_model, tf.keras.Model)
        self.assertEqual(loaded_model.input_shape, self.test_model.input_shape)
        self.assertEqual(loaded_model.output_shape, self.test_model.output_shape)
        
        # Check metadata
        self.assertIn('config', loaded_metadata)
        self.assertIn('metrics', loaded_metadata)
    
    def test_list_versions(self):
        """Test listing model versions."""
        # Initially should be empty
        versions = self.version_manager.list_versions()
        initial_count = len(versions)
        
        # Save a version
        self.version_manager.save_model_version(self.test_model, self.test_metadata)
        
        # Check that version is listed
        versions = self.version_manager.list_versions()
        self.assertEqual(len(versions), initial_count + 1)
    
    def test_get_best_version(self):
        """Test getting the best version by metric."""
        # Save multiple versions with different metrics
        metadata1 = {**self.test_metadata, 'metrics': {'val_iou_score': 0.8}}
        metadata2 = {**self.test_metadata, 'metrics': {'val_iou_score': 0.9}}
        metadata3 = {**self.test_metadata, 'metrics': {'val_iou_score': 0.7}}
        
        v1 = self.version_manager.save_model_version(self.test_model, metadata1)
        v2 = self.version_manager.save_model_version(self.test_model, metadata2)
        v3 = self.version_manager.save_model_version(self.test_model, metadata3)
        
        # Get best version
        best_version = self.version_manager.get_best_version('val_iou_score')
        self.assertEqual(best_version, v2)


class TestConfiguration(unittest.TestCase):
    """Test suite for configuration handling."""
    
    def test_config_loading(self):
        """Test loading configuration file."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'config.yaml'
        )
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            self.assertIn('model', config)
            self.assertIn('training', config)
            self.assertIn('data', config)
            
            # Check required parameters
            self.assertIn('input_shape', config['model'])
            self.assertIn('batch_size', config['training'])
            self.assertIn('learning_rate', config['training'])
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # This would be expanded based on specific validation needs
        sample_config = {
            'model': {'input_shape': [256, 256, 9]},
            'training': {'batch_size': 8, 'learning_rate': 0.001}
        }
        
        # Basic validation checks
        self.assertIsInstance(sample_config['model']['input_shape'], list)
        self.assertGreater(sample_config['training']['batch_size'], 0)
        self.assertGreater(sample_config['training']['learning_rate'], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_prediction_pipeline(self):
        """Test the complete prediction pipeline."""
        # Create a minimal model
        model = build_resunet_a(
            input_shape=(256, 256, 9),
            num_classes=1,
            use_enhanced_aspp=False
        )
        
        # Create sample input
        sample_input = np.random.rand(256, 256, 9)
        normalized_input = normalize_patch(sample_input)
        
        # Add batch dimension
        batch_input = np.expand_dims(normalized_input, axis=0)
        
        # Test prediction
        prediction = model.predict(batch_input, verbose=0)
        
        # Validate prediction
        self.assertEqual(prediction.shape, (1, 256, 256, 1))
        self.assertTrue(np.all(prediction >= 0))
        self.assertTrue(np.all(prediction <= 1))
        
        # Test with multiple samples
        batch_input_multi = np.random.rand(4, 256, 256, 9)
        prediction_multi = model.predict(batch_input_multi, verbose=0)
        self.assertEqual(prediction_multi.shape, (4, 256, 256, 1))


if __name__ == '__main__':
    # Configure test environment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    
    # Create test suite
    unittest.main(verbosity=2)
