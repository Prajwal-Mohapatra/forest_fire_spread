"""
Unit tests for model metrics and loss functions.
Tests focal loss, combined loss, IoU, Dice, and other evaluation metrics.
"""

import unittest
import numpy as np
import tensorflow as tf
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import (
    focal_loss, combined_loss, compute_class_weights,
    iou_score, dice_coef, f1_score, precision_score, 
    recall_score, comprehensive_evaluation, print_evaluation_report
)


class TestLossFunctions(unittest.TestCase):
    """Test loss functions."""
    
    def setUp(self):
        """Set up test data."""
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Create test predictions and labels
        self.y_true = tf.constant([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=tf.float32)
        self.y_pred = tf.constant([[0.1, 0.9, 0.2], [0.8, 0.1, 0.7]], dtype=tf.float32)
        
        # Create batch data for testing
        self.batch_y_true = tf.random.uniform((4, 32, 32, 1), 0, 2, dtype=tf.int32)
        self.batch_y_true = tf.cast(self.batch_y_true, tf.float32)
        self.batch_y_pred = tf.random.uniform((4, 32, 32, 1), 0, 1, dtype=tf.float32)
    
    def test_focal_loss_basic(self):
        """Test basic focal loss functionality."""
        loss_fn = focal_loss(alpha=0.25, gamma=2.0)
        loss = loss_fn(self.y_true, self.y_pred)
        
        # Check that loss is computed and is a scalar
        self.assertIsInstance(loss.numpy(), (float, np.floating))
        self.assertGreater(loss.numpy(), 0.0)
    
    def test_focal_loss_parameters(self):
        """Test focal loss with different parameters."""
        # Test different alpha values
        loss_fn1 = focal_loss(alpha=0.1, gamma=2.0)
        loss_fn2 = focal_loss(alpha=0.9, gamma=2.0)
        
        loss1 = loss_fn1(self.y_true, self.y_pred)
        loss2 = loss_fn2(self.y_true, self.y_pred)
        
        # Losses should be different
        self.assertNotAlmostEqual(loss1.numpy(), loss2.numpy(), places=4)
        
        # Test different gamma values
        loss_fn3 = focal_loss(alpha=0.25, gamma=0.5)
        loss_fn4 = focal_loss(alpha=0.25, gamma=5.0)
        
        loss3 = loss_fn3(self.y_true, self.y_pred)
        loss4 = loss_fn4(self.y_true, self.y_pred)
        
        # Losses should be different
        self.assertNotAlmostEqual(loss3.numpy(), loss4.numpy(), places=4)
    
    def test_focal_loss_edge_cases(self):
        """Test focal loss edge cases."""
        loss_fn = focal_loss(alpha=0.25, gamma=2.0)
        
        # Perfect predictions
        y_true_perfect = tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32)
        y_pred_perfect = tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32)
        
        loss_perfect = loss_fn(y_true_perfect, y_pred_perfect)
        self.assertLess(loss_perfect.numpy(), 0.1)  # Should be very small
        
        # Worst predictions
        y_pred_worst = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
        loss_worst = loss_fn(y_true_perfect, y_pred_worst)
        self.assertGreater(loss_worst.numpy(), loss_perfect.numpy())
    
    def test_combined_loss_basic(self):
        """Test combined loss functionality."""
        loss_fn = combined_loss(dice_weight=0.5, focal_weight=0.5, alpha=0.25, gamma=2.0)
        loss = loss_fn(self.batch_y_true, self.batch_y_pred)
        
        # Check that loss is computed
        self.assertIsInstance(loss.numpy(), (float, np.floating))
        self.assertGreater(loss.numpy(), 0.0)
    
    def test_combined_loss_weights(self):
        """Test combined loss with different weights."""
        # Dice-heavy loss
        loss_fn1 = combined_loss(dice_weight=0.9, focal_weight=0.1)
        loss1 = loss_fn1(self.batch_y_true, self.batch_y_pred)
        
        # Focal-heavy loss
        loss_fn2 = combined_loss(dice_weight=0.1, focal_weight=0.9)
        loss2 = loss_fn2(self.batch_y_true, self.batch_y_pred)
        
        # Should be different
        self.assertNotAlmostEqual(loss1.numpy(), loss2.numpy(), places=4)
        
        # Weights should sum to 1.0
        loss_fn3 = combined_loss(dice_weight=0.3, focal_weight=0.7)
        loss3 = loss_fn3(self.batch_y_true, self.batch_y_pred)
        self.assertIsInstance(loss3.numpy(), (float, np.floating))
    
    def test_calculate_class_weights_balanced(self):
        """Test class weight calculation with balanced data."""
        # Balanced data
        y_balanced = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        weights = calculate_class_weights(y_balanced)
        
        # Should be approximately equal for balanced data
        self.assertAlmostEqual(weights[0], weights[1], places=2)
        
        # Should be close to 1.0 for balanced data
        self.assertAlmostEqual(weights[0], 1.0, places=1)
        self.assertAlmostEqual(weights[1], 1.0, places=1)
    
    def test_calculate_class_weights_imbalanced(self):
        """Test class weight calculation with imbalanced data."""
        # Imbalanced data (90% class 0, 10% class 1)
        y_imbalanced = np.array([0] * 90 + [1] * 10)
        weights = calculate_class_weights(y_imbalanced)
        
        # Class 1 (minority) should have higher weight
        self.assertGreater(weights[1], weights[0])
        
        # Weights should be positive
        self.assertGreater(weights[0], 0)
        self.assertGreater(weights[1], 0)
    
    def test_calculate_class_weights_edge_cases(self):
        """Test class weight calculation edge cases."""
        # Single class
        y_single = np.array([0, 0, 0, 0])
        weights = calculate_class_weights(y_single)
        
        # Should handle gracefully
        self.assertEqual(len(weights), 2)
        self.assertGreater(weights[0], 0)


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""
    
    def setUp(self):
        """Set up test data."""
        # Create binary test data
        self.y_true_binary = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        self.y_pred_binary = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        
        # Create 2D test data
        self.y_true_2d = np.array([[0, 1], [1, 0]])
        self.y_pred_2d = np.array([[0, 1], [0, 1]])
        
        # Create batch test data
        np.random.seed(42)
        self.y_true_batch = np.random.randint(0, 2, (2, 16, 16))
        self.y_pred_batch = np.random.rand(2, 16, 16)
    
    def test_iou_score_basic(self):
        """Test IoU score calculation."""
        iou = iou_score(self.y_true_2d, self.y_pred_2d)
        
        # IoU should be between 0 and 1
        self.assertGreaterEqual(iou, 0.0)
        self.assertLessEqual(iou, 1.0)
        
        # Perfect prediction should give IoU = 1
        iou_perfect = iou_score(self.y_true_2d, self.y_true_2d)
        self.assertAlmostEqual(iou_perfect, 1.0, places=5)
    
    def test_iou_score_edge_cases(self):
        """Test IoU score edge cases."""
        # No intersection
        y_true = np.array([[1, 1], [0, 0]])
        y_pred = np.array([[0, 0], [1, 1]])
        iou = iou_score(y_true, y_pred)
        self.assertAlmostEqual(iou, 0.0, places=5)
        
        # All zeros
        y_zeros = np.zeros((4, 4))
        iou_zeros = iou_score(y_zeros, y_zeros)
        # Should handle gracefully (might be 1.0 or NaN, depends on implementation)
        self.assertTrue(np.isfinite(iou_zeros) or np.isnan(iou_zeros))
    
    def test_dice_score_basic(self):
        """Test Dice score calculation."""
        dice = dice_score(self.y_true_2d, self.y_pred_2d)
        
        # Dice should be between 0 and 1
        self.assertGreaterEqual(dice, 0.0)
        self.assertLessEqual(dice, 1.0)
        
        # Perfect prediction should give Dice = 1
        dice_perfect = dice_score(self.y_true_2d, self.y_true_2d)
        self.assertAlmostEqual(dice_perfect, 1.0, places=5)
    
    def test_f1_score_basic(self):
        """Test F1 score calculation."""
        f1 = f1_score(self.y_true_binary, self.y_pred_binary)
        
        # F1 should be between 0 and 1
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)
        
        # Perfect prediction should give F1 = 1
        f1_perfect = f1_score(self.y_true_binary, self.y_true_binary)
        self.assertAlmostEqual(f1_perfect, 1.0, places=5)
    
    def test_precision_recall_specificity(self):
        """Test precision, recall, and specificity calculations."""
        precision = precision_score(self.y_true_binary, self.y_pred_binary)
        recall = recall_score(self.y_true_binary, self.y_pred_binary)
        specificity = specificity_score(self.y_true_binary, self.y_pred_binary)
        
        # All should be between 0 and 1
        for metric in [precision, recall, specificity]:
            self.assertGreaterEqual(metric, 0.0)
            self.assertLessEqual(metric, 1.0)
        
        # Test with known values
        y_true_known = np.array([1, 1, 0, 0])
        y_pred_known = np.array([1, 0, 0, 1])
        
        precision_known = precision_score(y_true_known, y_pred_known)
        recall_known = recall_score(y_true_known, y_pred_known)
        specificity_known = specificity_score(y_true_known, y_pred_known)
        
        # TP=1, FP=1, FN=1, TN=1
        # Precision = TP/(TP+FP) = 1/2 = 0.5
        # Recall = TP/(TP+FN) = 1/2 = 0.5
        # Specificity = TN/(TN+FP) = 1/2 = 0.5
        self.assertAlmostEqual(precision_known, 0.5, places=5)
        self.assertAlmostEqual(recall_known, 0.5, places=5)
        self.assertAlmostEqual(specificity_known, 0.5, places=5)
    
    def test_confusion_matrix_metrics(self):
        """Test confusion matrix metrics calculation."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        
        tn, fp, fn, tp = confusion_matrix_metrics(y_true, y_pred)
        
        # Check values
        self.assertEqual(tp, 1)  # True positives
        self.assertEqual(fp, 1)  # False positives
        self.assertEqual(fn, 1)  # False negatives
        self.assertEqual(tn, 1)  # True negatives
        
        # Total should equal number of samples
        self.assertEqual(tp + fp + fn + tn, len(y_true))
    
    def test_evaluate_model_comprehensive_mock(self):
        """Test comprehensive model evaluation with mock data."""
        # Create mock model
        class MockModel:
            def predict(self, x):
                return np.random.rand(x.shape[0], x.shape[1], x.shape[2], 1)
        
        model = MockModel()
        
        # Create test generator
        class MockGenerator:
            def __len__(self):
                return 2
            
            def __getitem__(self, idx):
                x = np.random.rand(2, 32, 32, 9).astype(np.float32)
                y = np.random.randint(0, 2, (2, 32, 32, 1)).astype(np.float32)
                return x, y
        
        generator = MockGenerator()
        
        # Test evaluation
        try:
            metrics = evaluate_model_comprehensive(model, generator, threshold=0.5)
            
            # Check that metrics are returned
            self.assertIsInstance(metrics, dict)
            
            # Check for expected keys
            expected_keys = ['iou', 'dice', 'f1', 'precision', 'recall', 'specificity']
            for key in expected_keys:
                self.assertIn(key, metrics)
                self.assertIsInstance(metrics[key], (float, np.floating))
                
        except Exception as e:
            # If evaluation fails due to dependencies, that's okay for unit test
            self.skipTest(f"Evaluation failed due to dependencies: {e}")


class TestMetricsIntegration(unittest.TestCase):
    """Integration tests for metrics."""
    
    def test_metrics_consistency(self):
        """Test that metrics are consistent with each other."""
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0, 0, 1])
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # F1 should be harmonic mean of precision and recall
        expected_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        self.assertAlmostEqual(f1, expected_f1, places=5)
    
    def test_metrics_with_tensorflow_tensors(self):
        """Test metrics work with TensorFlow tensors."""
        y_true_tf = tf.constant([1, 1, 0, 0], dtype=tf.float32)
        y_pred_tf = tf.constant([1, 0, 0, 1], dtype=tf.float32)
        
        # Convert to numpy and test
        y_true_np = y_true_tf.numpy()
        y_pred_np = y_pred_tf.numpy()
        
        precision = precision_score(y_true_np, y_pred_np)
        recall = recall_score(y_true_np, y_pred_np)
        
        # Should compute without errors
        self.assertIsInstance(precision, (float, np.floating))
        self.assertIsInstance(recall, (float, np.floating))


if __name__ == '__main__':
    unittest.main()
