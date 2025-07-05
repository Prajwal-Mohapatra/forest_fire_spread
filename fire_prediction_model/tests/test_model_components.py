"""
Comprehensive unit tests for the forest fire prediction model.
"""
import unittest
import numpy as np
import tensorflow as tf
import os
import sys

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.resunet_a import (
    residual_block, 
    atrous_spatial_pyramid_pooling, 
    improved_attention_gate, 
    build_resunet_a
)
from utils.metrics import (
    iou_score, 
    dice_coef, 
    focal_loss, 
    combined_loss,
    comprehensive_evaluation,
    precision_score,
    recall_score,
    f1_score
)
from dataset.preprocess import normalize_patch

class TestModelArchitecture(unittest.TestCase):
    """Test ResUNet-A model architecture components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_shape = (256, 256, 9)
        self.batch_size = 2
        
    def test_residual_block(self):
        """Test residual block functionality."""
        # Test basic residual block
        input_tensor = tf.random.normal((self.batch_size, 64, 64, 32))
        output = residual_block(input_tensor, filters=32)
        
        self.assertEqual(output.shape, (self.batch_size, 64, 64, 32))
        
        # Test residual block with stride
        output_stride = residual_block(input_tensor, filters=64, stride=2)
        self.assertEqual(output_stride.shape, (self.batch_size, 32, 32, 64))
        
        # Test residual block with dilation
        output_dilated = residual_block(input_tensor, filters=32, dilation=2)
        self.assertEqual(output_dilated.shape, (self.batch_size, 64, 64, 32))
    
    def test_aspp_module(self):
        """Test ASPP module."""
        input_tensor = tf.random.normal((self.batch_size, 32, 32, 512))
        
        # Test basic ASPP
        aspp_output = atrous_spatial_pyramid_pooling(input_tensor, output_filters=256)
        self.assertEqual(aspp_output.shape, (self.batch_size, 32, 32, 256))
        
        # Test ASPP with custom rates
        aspp_custom = atrous_spatial_pyramid_pooling(
            input_tensor, 
            output_filters=128, 
            rates=[3, 6, 9],
            use_global_pooling=False
        )
        self.assertEqual(aspp_custom.shape, (self.batch_size, 32, 32, 128))
    
    def test_attention_gate(self):
        """Test improved attention gate."""
        x = tf.random.normal((self.batch_size, 64, 64, 256))
        g = tf.random.normal((self.batch_size, 32, 32, 512))
        
        attended_output = improved_attention_gate(x, g, inter_channels=128)
        self.assertEqual(attended_output.shape, (self.batch_size, 64, 64, 256))
        
        # Test with same spatial dimensions
        g_same = tf.random.normal((self.batch_size, 64, 64, 512))
        attended_same = improved_attention_gate(x, g_same, inter_channels=128)
        self.assertEqual(attended_same.shape, (self.batch_size, 64, 64, 256))
    
    def test_resunet_a_model(self):
        """Test complete ResUNet-A model."""
        # Test basic model
        model = build_resunet_a(input_shape=self.input_shape, num_classes=1)
        
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(len(model.outputs), 1)
        self.assertEqual(model.input_shape, (None, 256, 256, 9))
        self.assertEqual(model.output_shape, (None, 256, 256, 1))
        
        # Test with enhanced ASPP
        model_aspp = build_resunet_a(
            input_shape=self.input_shape, 
            num_classes=1, 
            use_enhanced_aspp=True
        )
        self.assertEqual(model_aspp.input_shape, (None, 256, 256, 9))
        
        # Test multiclass model
        model_multiclass = build_resunet_a(
            input_shape=self.input_shape, 
            num_classes=3
        )
        self.assertEqual(model_multiclass.output_shape, (None, 256, 256, 3))
    
    def test_model_inference(self):
        """Test model inference with random input."""
        model = build_resunet_a(input_shape=self.input_shape)
        
        # Test single prediction
        input_batch = tf.random.normal((1, 256, 256, 9))
        prediction = model(input_batch, training=False)
        
        self.assertEqual(prediction.shape, (1, 256, 256, 1))
        self.assertTrue(tf.reduce_all(prediction >= 0))
        self.assertTrue(tf.reduce_all(prediction <= 1))
        
        # Test batch prediction
        input_batch = tf.random.normal((4, 256, 256, 9))
        predictions = model(input_batch, training=False)
        self.assertEqual(predictions.shape, (4, 256, 256, 1))

class TestMetrics(unittest.TestCase):
    """Test metric functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic ground truth and predictions
        self.y_true = tf.constant([[[[1.0], [0.0]], [[0.0], [1.0]]]])
        self.y_pred = tf.constant([[[[0.8], [0.2]], [[0.3], [0.9]]]])
        
        # Numpy versions for sklearn metrics
        self.y_true_np = np.array([[1, 0], [0, 1]])
        self.y_pred_np = np.array([[0.8, 0.2], [0.3, 0.9]])
    
    def test_iou_score(self):
        """Test IoU score calculation."""
        iou = iou_score(self.y_true, self.y_pred)
        
        # IoU should be between 0 and 1
        self.assertGreaterEqual(iou, 0.0)
        self.assertLessEqual(iou, 1.0)
        
        # Test perfect prediction
        perfect_iou = iou_score(self.y_true, self.y_true)
        self.assertAlmostEqual(perfect_iou, 1.0, places=4)
    
    def test_dice_coefficient(self):
        """Test Dice coefficient calculation."""
        dice = dice_coef(self.y_true, self.y_pred)
        
        # Dice should be between 0 and 1
        self.assertGreaterEqual(dice, 0.0)
        self.assertLessEqual(dice, 1.0)
        
        # Test perfect prediction
        perfect_dice = dice_coef(self.y_true, self.y_true)
        self.assertAlmostEqual(perfect_dice, 1.0, places=4)
    
    def test_precision_recall_f1(self):
        """Test precision, recall, and F1 score."""
        precision = precision_score(self.y_true, self.y_pred)
        recall = recall_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred)
        
        # All metrics should be between 0 and 1
        for metric in [precision, recall, f1]:
            self.assertGreaterEqual(metric, 0.0)
            self.assertLessEqual(metric, 1.0)
        
        # F1 should be harmonic mean of precision and recall
        expected_f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        self.assertAlmostEqual(f1, expected_f1, places=4)
    
    def test_focal_loss(self):
        """Test focal loss calculation."""
        focal_loss_fn = focal_loss(alpha=0.25, gamma=2.0)
        loss = focal_loss_fn(self.y_true, self.y_pred)
        
        # Loss should be positive
        self.assertGreaterEqual(loss, 0.0)
        
        # Test with different parameters
        focal_loss_fn_alt = focal_loss(alpha=0.5, gamma=1.0)
        loss_alt = focal_loss_fn_alt(self.y_true, self.y_pred)
        self.assertGreaterEqual(loss_alt, 0.0)
    
    def test_combined_loss(self):
        """Test combined focal + dice loss."""
        combined_loss_fn = combined_loss(alpha=0.25, gamma=2.0)
        loss = combined_loss_fn(self.y_true, self.y_pred)
        
        # Loss should be positive
        self.assertGreaterEqual(loss, 0.0)
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation function."""
        metrics = comprehensive_evaluation(self.y_true_np, self.y_pred_np)
        
        # Check that all expected metrics are present
        expected_keys = [
            'confusion_matrix', 'precision', 'recall', 'f1_score',
            'iou', 'dice', 'accuracy', 'specificity', 'auc_roc', 'auc_pr'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check confusion matrix structure
        cm = metrics['confusion_matrix']
        self.assertIn('TP', cm)
        self.assertIn('TN', cm)
        self.assertIn('FP', cm)
        self.assertIn('FN', cm)
        
        # All metrics should be between 0 and 1 (except confusion matrix)
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions."""
    
    def test_normalize_patch(self):
        """Test patch normalization."""
        # Create test patch with known statistics
        patch = np.random.rand(256, 256, 9).astype(np.float32) * 1000
        
        normalized = normalize_patch(patch)
        
        # Check output shape
        self.assertEqual(normalized.shape, patch.shape)
        
        # Check normalization per channel
        for i in range(9):
            channel_data = normalized[:, :, i]
            # Should be approximately normalized (allowing for small numerical errors)
            self.assertAlmostEqual(np.mean(channel_data), 0.0, places=1)
            self.assertAlmostEqual(np.std(channel_data), 1.0, places=1)
    
    def test_normalize_patch_edge_cases(self):
        """Test normalization with edge cases."""
        # Test with constant values
        constant_patch = np.ones((256, 256, 9), dtype=np.float32) * 100
        normalized_constant = normalize_patch(constant_patch)
        
        # Should handle constant channels gracefully
        self.assertFalse(np.any(np.isnan(normalized_constant)))
        self.assertFalse(np.any(np.isinf(normalized_constant)))
        
        # Test with zeros
        zero_patch = np.zeros((256, 256, 9), dtype=np.float32)
        normalized_zero = normalize_patch(zero_patch)
        
        self.assertFalse(np.any(np.isnan(normalized_zero)))
        self.assertFalse(np.any(np.isinf(normalized_zero)))

class TestModelIntegration(unittest.TestCase):
    """Integration tests for the complete model pipeline."""
    
    def test_end_to_end_training_step(self):
        """Test a complete training step."""
        # Create model
        model = build_resunet_a(input_shape=(256, 256, 9))
        
        # Compile with focal loss
        focal_loss_fn = focal_loss(alpha=0.25, gamma=2.0)
        model.compile(
            optimizer='adam',
            loss=focal_loss_fn,
            metrics=[iou_score, dice_coef, precision_score, recall_score, f1_score]
        )
        
        # Create synthetic training data
        x_train = tf.random.normal((4, 256, 256, 9))
        y_train = tf.random.uniform((4, 256, 256, 1), maxval=2, dtype=tf.int32)
        y_train = tf.cast(y_train, tf.float32)
        
        # Test training step
        history = model.fit(x_train, y_train, epochs=1, verbose=0)
        
        # Check that loss and metrics are computed
        self.assertIn('loss', history.history)
        self.assertIn('iou_score', history.history)
        self.assertIn('dice_coef', history.history)
    
    def test_model_prediction_pipeline(self):
        """Test the complete prediction pipeline."""
        # Create model
        model = build_resunet_a(input_shape=(256, 256, 9))
        
        # Test prediction with normalized input
        input_data = np.random.rand(256, 256, 9).astype(np.float32)
        normalized_input = normalize_patch(input_data)
        
        # Predict
        prediction = model.predict(np.expand_dims(normalized_input, 0), verbose=0)
        
        # Check prediction shape and values
        self.assertEqual(prediction.shape, (1, 256, 256, 1))
        self.assertTrue(np.all(prediction >= 0))
        self.assertTrue(np.all(prediction <= 1))

def run_tests():
    """Run all tests and generate report."""
    print("🧪 Running comprehensive unit tests for forest fire prediction model...")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestModelArchitecture,
        TestMetrics,
        TestPreprocessing,
        TestModelIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"❌ {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"💥 {test}: {traceback}")
    
    if not result.failures and not result.errors:
        print("✅ All tests passed successfully!")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_tests()
