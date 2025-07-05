import tensorflow.keras.backend as K
import tensorflow as tf

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
