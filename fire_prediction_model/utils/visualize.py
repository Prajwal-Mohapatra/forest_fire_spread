import matplotlib.pyplot as plt
import numpy as np

def plot_prediction(image, mask, pred_mask, save_path=None):
    """
    Display image, true mask, predicted mask side-by-side
    image: (H, W, 3) or (H, W, C)
    mask, pred_mask: (H, W)
    """
    if image.shape[-1] > 3:
        image = image[:, :, :3]  # use RGB-like composite
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(image)
    ax[0].set_title("Input Image")
    
    ax[1].imshow(mask.squeeze(), cmap='Reds')
    ax[1].set_title("True Fire Mask")
    
    ax[2].imshow(pred_mask.squeeze(), cmap='Reds')
    ax[2].set_title("Predicted Fire Mask")
    
    for a in ax:
        a.axis('off')
        
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
