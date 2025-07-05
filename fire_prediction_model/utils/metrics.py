import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc

def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred > 0.5, 'float32')
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(K.cast(y_pred > 0.5, 'float32'))
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def precision_score(y_true, y_pred, smooth=1e-6):
    """Precision metric for Keras training."""
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred > 0.5, 'float32')
    true_positives = K.sum(y_true * y_pred)
    predicted_positives = K.sum(y_pred)
    return (true_positives + smooth) / (predicted_positives + smooth)

def recall_score(y_true, y_pred, smooth=1e-6):
    """Recall metric for Keras training."""
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred > 0.5, 'float32')
    true_positives = K.sum(y_true * y_pred)
    possible_positives = K.sum(y_true)
    return (true_positives + smooth) / (possible_positives + smooth)

def f1_score(y_true, y_pred, smooth=1e-6):
    """F1 score metric for Keras training."""
    precision = precision_score(y_true, y_pred, smooth)
    recall = recall_score(y_true, y_pred, smooth)
    return 2 * ((precision * recall) / (precision + recall + smooth))

def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance.
    Args:
        alpha: Weighting factor for rare class (fire pixels)
        gamma: Focusing parameter to down-weight easy examples
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * K.pow((1 - p_t), gamma)
        
        # Binary cross entropy
        bce = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        
        # Apply focal weight
        focal = focal_weight * bce
        
        return K.mean(focal)
    
    return focal_loss_fixed

def combined_loss(alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.5):
    """
    Combined Focal + Dice loss for better segmentation performance.
    """
    def combined_loss_fixed(y_true, y_pred):
        # Focal loss component
        focal = focal_loss(alpha, gamma)(y_true, y_pred)
        
        # Dice loss component (1 - dice_coefficient)
        dice = 1 - dice_coef_continuous(y_true, y_pred)
        
        return focal_weight * focal + dice_weight * dice
    
    return combined_loss_fixed

def dice_coef_continuous(y_true, y_pred, smooth=1e-6):
    """
    Continuous dice coefficient for use in loss functions.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def compute_class_weights(train_generator, num_samples=10):
    """
    Compute class weights from training data.
    """
    fire_pixels = 0
    total_pixels = 0
    
    print("Computing class weights from training data...")
    for i in range(min(num_samples, len(train_generator))):
        _, masks = train_generator[i]
        fire_pixels += tf.reduce_sum(masks).numpy()
        total_pixels += tf.size(masks).numpy()
    
    fire_ratio = fire_pixels / total_pixels
    fire_weight = (1.0 / fire_ratio) / 2.0
    no_fire_weight = (1.0 / (1.0 - fire_ratio)) / 2.0
    
    print(f"Fire ratio: {fire_ratio:.6f}")
    print(f"Class weights - Fire: {fire_weight:.4f}, No-fire: {no_fire_weight:.4f}")
    
    return fire_weight, no_fire_weight

def comprehensive_evaluation(y_true, y_pred, threshold=0.5):
    """
    Compute comprehensive evaluation metrics including confusion matrix, 
    precision, recall, F1, IoU, Dice, and AUC.
    
    Args:
        y_true: Ground truth binary mask (numpy array)
        y_pred: Predicted probabilities (numpy array)
        threshold: Classification threshold
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    y_pred_binary = (y_pred_flat > threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_binary, labels=[0, 1]).ravel()
    
    # Basic metrics
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # Segmentation metrics
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    
    # Additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    
    # AUC metrics (if probabilities available)
    try:
        auc_roc = roc_auc_score(y_true_flat, y_pred_flat)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_flat, y_pred_flat)
        auc_pr = auc(recall_curve, precision_curve)
    except:
        auc_roc = 0.0
        auc_pr = 0.0
    
    return {
        'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn},
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'specificity': specificity,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr
    }

def print_evaluation_report(metrics_dict, title="Evaluation Report"):
    """Pretty print comprehensive evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    cm = metrics_dict['confusion_matrix']
    print(f"Confusion Matrix:")
    print(f"  True Pos: {cm['TP']:,8}  |  False Pos: {cm['FP']:,8}")
    print(f"  False Neg: {cm['FN']:,7}  |  True Neg:  {cm['TN']:,8}")
    
    print(f"\nSegmentation Metrics:")
    print(f"  IoU (Jaccard):     {metrics_dict['iou']:.4f}")
    print(f"  Dice Coefficient:  {metrics_dict['dice']:.4f}")
    
    print(f"\nClassification Metrics:")
    print(f"  Precision:         {metrics_dict['precision']:.4f}")
    print(f"  Recall:            {metrics_dict['recall']:.4f}")
    print(f"  F1-Score:          {metrics_dict['f1_score']:.4f}")
    print(f"  Accuracy:          {metrics_dict['accuracy']:.4f}")
    print(f"  Specificity:       {metrics_dict['specificity']:.4f}")
    
    print(f"\nAUC Metrics:")
    print(f"  AUC-ROC:           {metrics_dict['auc_roc']:.4f}")
    print(f"  AUC-PR:            {metrics_dict['auc_pr']:.4f}")
    print(f"{'='*50}")
