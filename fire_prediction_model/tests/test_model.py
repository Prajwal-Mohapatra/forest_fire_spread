"""
Unit tests for model architecture components.
Tests ResUNet-A model, ASPP module, attention gates, and model building.
"""

import unittest
import numpy as np
import tensorflow as tf
import sys
import os
import tempfile
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.resunet_a import ResUNetA, ASPPBlock


class TestASPPBlock(unittest.TestCase):
    """Test ASPP (Atrous Spatial Pyramid Pooling) block."""
    
    def setUp(self):
        """Set up test data."""
        tf.random.set_seed(42)
        
        # Create test input
        self.test_input = tf.random.normal((2, 32, 32, 256))
        
        # ASPP configuration
        self.aspp_config = {
            'output_filters': 256,
            'dilation_rates': [6, 12, 18],
            'use_global_pooling': True,
            'dropout_rate': 0.1
        }
    
    def test_aspp_block_creation(self):
        """Test ASPP block creation."""
        aspp = ASPPBlock(**self.aspp_config)
        
        # Check that it's a tf.keras.layers.Layer
        self.assertIsInstance(aspp, tf.keras.layers.Layer)
        
        # Check configuration
        self.assertEqual(aspp.output_filters, 256)
        self.assertEqual(aspp.dilation_rates, [6, 12, 18])
        self.assertTrue(aspp.use_global_pooling)
        self.assertEqual(aspp.dropout_rate, 0.1)
    
    def test_aspp_block_call(self):
        """Test ASPP block forward pass."""
        aspp = ASPPBlock(**self.aspp_config)
        
        # Build the layer
        output = aspp(self.test_input)
        
        # Check output shape
        expected_shape = (2, 32, 32, 256)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output type
        self.assertEqual(output.dtype, tf.float32)
    
    def test_aspp_block_without_global_pooling(self):
        """Test ASPP block without global pooling."""
        config = self.aspp_config.copy()
        config['use_global_pooling'] = False
        
        aspp = ASPPBlock(**config)
        output = aspp(self.test_input)
        
        # Should still work
        expected_shape = (2, 32, 32, 256)
        self.assertEqual(output.shape, expected_shape)
    
    def test_aspp_block_different_rates(self):
        """Test ASPP block with different dilation rates."""
        config = self.aspp_config.copy()
        config['dilation_rates'] = [3, 6, 9, 12]
        
        aspp = ASPPBlock(**config)
        output = aspp(self.test_input)
        
        # Should work with different number of rates
        expected_shape = (2, 32, 32, 256)
        self.assertEqual(output.shape, expected_shape)
    
    def test_aspp_block_training_mode(self):
        """Test ASPP block in training vs inference mode."""
        aspp = ASPPBlock(**self.aspp_config)
        
        # Training mode
        output_train = aspp(self.test_input, training=True)
        
        # Inference mode
        output_infer = aspp(self.test_input, training=False)
        
        # Shapes should be the same
        self.assertEqual(output_train.shape, output_infer.shape)
        
        # Values might be different due to dropout
        # (though with small probability they could be the same)
        self.assertEqual(output_train.dtype, output_infer.dtype)


class TestResUNetA(unittest.TestCase):
    """Test ResUNet-A model."""
    
    def setUp(self):
        """Set up test data."""
        tf.random.set_seed(42)
        
        # Model configuration
        self.model_config = {
            'input_shape': [64, 64, 9],
            'num_classes': 1,
            'use_enhanced_aspp': True
        }
        
        self.aspp_config = {
            'output_filters': 256,
            'dilation_rates': [6, 12, 18],
            'use_global_pooling': True,
            'dropout_rate': 0.1
        }
        
        self.attention_config = {
            'inter_channels_ratio': 0.5
        }
        
        # Create test input
        self.test_input = tf.random.normal((2, 64, 64, 9))
    
    def test_model_creation(self):
        """Test model creation."""
        model = ResUNetA(
            input_shape=self.model_config['input_shape'],
            num_classes=self.model_config['num_classes'],
            use_enhanced_aspp=self.model_config['use_enhanced_aspp'],
            aspp_config=self.aspp_config,
            attention_config=self.attention_config
        )
        
        # Check that it's a tf.keras.Model
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape
        expected_input_shape = (None, 64, 64, 9)
        self.assertEqual(model.input_shape, expected_input_shape)
    
    def test_model_call(self):
        """Test model forward pass."""
        model = ResUNetA(
            input_shape=self.model_config['input_shape'],
            num_classes=self.model_config['num_classes'],
            use_enhanced_aspp=self.model_config['use_enhanced_aspp'],
            aspp_config=self.aspp_config,
            attention_config=self.attention_config
        )
        
        # Forward pass
        output = model(self.test_input)
        
        # Check output shape
        expected_shape = (2, 64, 64, 1)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output type
        self.assertEqual(output.dtype, tf.float32)
        
        # Check output range (should be sigmoid output, so 0-1)
        self.assertTrue(tf.reduce_all(output >= 0.0))
        self.assertTrue(tf.reduce_all(output <= 1.0))
    
    def test_model_without_aspp(self):
        """Test model without enhanced ASPP."""
        model = ResUNetA(
            input_shape=self.model_config['input_shape'],
            num_classes=self.model_config['num_classes'],
            use_enhanced_aspp=False,
            aspp_config=self.aspp_config,
            attention_config=self.attention_config
        )
        
        output = model(self.test_input)
        
        # Should still work
        expected_shape = (2, 64, 64, 1)
        self.assertEqual(output.shape, expected_shape)
    
    def test_model_different_input_shapes(self):
        """Test model with different input shapes."""
        # Test smaller input
        small_input = tf.random.normal((1, 32, 32, 9))
        model_small = ResUNetA(
            input_shape=[32, 32, 9],
            num_classes=1,
            use_enhanced_aspp=True,
            aspp_config=self.aspp_config,
            attention_config=self.attention_config
        )
        
        output_small = model_small(small_input)
        self.assertEqual(output_small.shape, (1, 32, 32, 1))
        
        # Test larger input
        large_input = tf.random.normal((1, 128, 128, 9))
        model_large = ResUNetA(
            input_shape=[128, 128, 9],
            num_classes=1,
            use_enhanced_aspp=True,
            aspp_config=self.aspp_config,
            attention_config=self.attention_config
        )
        
        output_large = model_large(large_input)
        self.assertEqual(output_large.shape, (1, 128, 128, 1))
    
    def test_model_different_channels(self):
        """Test model with different number of input channels."""
        # Test with 3 channels (RGB)
        rgb_input = tf.random.normal((1, 64, 64, 3))
        model_rgb = ResUNetA(
            input_shape=[64, 64, 3],
            num_classes=1,
            use_enhanced_aspp=True,
            aspp_config=self.aspp_config,
            attention_config=self.attention_config
        )
        
        output_rgb = model_rgb(rgb_input)
        self.assertEqual(output_rgb.shape, (1, 64, 64, 1))
        
        # Test with 12 channels
        multi_input = tf.random.normal((1, 64, 64, 12))
        model_multi = ResUNetA(
            input_shape=[64, 64, 12],
            num_classes=1,
            use_enhanced_aspp=True,
            aspp_config=self.aspp_config,
            attention_config=self.attention_config
        )
        
        output_multi = model_multi(multi_input)
        self.assertEqual(output_multi.shape, (1, 64, 64, 1))
    
    def test_model_multiclass(self):
        """Test model with multiple classes."""
        model_multiclass = ResUNetA(
            input_shape=[64, 64, 9],
            num_classes=3,
            use_enhanced_aspp=True,
            aspp_config=self.aspp_config,
            attention_config=self.attention_config
        )
        
        output = model_multiclass(self.test_input)
        
        # Check output shape
        expected_shape = (2, 64, 64, 3)
        self.assertEqual(output.shape, expected_shape)
        
        # For multiclass, output should be softmax probabilities
        # Sum across classes should be approximately 1
        class_sums = tf.reduce_sum(output, axis=-1)
        self.assertTrue(tf.reduce_all(tf.abs(class_sums - 1.0) < 0.01))
    
    def test_model_summary(self):
        """Test model summary generation."""
        model = ResUNetA(
            input_shape=self.model_config['input_shape'],
            num_classes=self.model_config['num_classes'],
            use_enhanced_aspp=self.model_config['use_enhanced_aspp'],
            aspp_config=self.aspp_config,
            attention_config=self.attention_config
        )
        
        # Build model
        _ = model(self.test_input)
        
        # Should be able to get summary without errors
        try:
            summary = model.summary()
            # If summary() returns None, that's fine
        except Exception as e:
            self.fail(f"Model summary failed: {e}")
    
    def test_model_trainable_parameters(self):
        """Test that model has trainable parameters."""
        model = ResUNetA(
            input_shape=self.model_config['input_shape'],
            num_classes=self.model_config['num_classes'],
            use_enhanced_aspp=self.model_config['use_enhanced_aspp'],
            aspp_config=self.aspp_config,
            attention_config=self.attention_config
        )
        
        # Build model
        _ = model(self.test_input)
        
        # Check trainable parameters
        trainable_params = model.trainable_variables
        self.assertGreater(len(trainable_params), 0)
        
        # Check total parameter count
        total_params = model.count_params()
        self.assertGreater(total_params, 1000)  # Should have substantial parameters


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components."""
    
    def setUp(self):
        """Set up test configuration."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test config file
        self.config = {
            'model': {
                'input_shape': [64, 64, 9],
                'num_classes': 1,
                'use_enhanced_aspp': True
            },
            'aspp': {
                'output_filters': 256,
                'dilation_rates': [6, 12, 18],
                'use_global_pooling': True,
                'dropout_rate': 0.1
            },
            'attention': {
                'inter_channels_ratio': 0.5
            }
        }
        
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_model_with_config(self):
        """Test model creation with config file."""
        # Load config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        model = ResUNetA(
            input_shape=config['model']['input_shape'],
            num_classes=config['model']['num_classes'],
            use_enhanced_aspp=config['model']['use_enhanced_aspp'],
            aspp_config=config['aspp'],
            attention_config=config['attention']
        )
        
        # Test forward pass
        test_input = tf.random.normal((1, 64, 64, 9))
        output = model(test_input)
        
        self.assertEqual(output.shape, (1, 64, 64, 1))
    
    def test_model_compilation(self):
        """Test model compilation with different optimizers and losses."""
        model = ResUNetA(
            input_shape=[64, 64, 9],
            num_classes=1,
            use_enhanced_aspp=True,
            aspp_config=self.config['aspp'],
            attention_config=self.config['attention']
        )
        
        # Test compilation with different optimizers
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        
        # Should compile without errors
        self.assertTrue(model.compiled)
    
    def test_model_training_step(self):
        """Test single training step."""
        model = ResUNetA(
            input_shape=[32, 32, 9],
            num_classes=1,
            use_enhanced_aspp=True,
            aspp_config=self.config['aspp'],
            attention_config=self.config['attention']
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        
        # Create test data
        x_train = tf.random.normal((2, 32, 32, 9))
        y_train = tf.random.uniform((2, 32, 32, 1), 0, 1)
        
        # Single training step
        try:
            history = model.fit(x_train, y_train, epochs=1, verbose=0)
            
            # Check that loss was computed
            self.assertIn('loss', history.history)
            self.assertGreater(len(history.history['loss']), 0)
            
        except Exception as e:
            self.fail(f"Training step failed: {e}")


if __name__ == '__main__':
    unittest.main()
