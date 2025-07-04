#=========== evaluate ==========
import os
import rasterio
import numpy as np
from sklearn.metrics import confusion_matrix
from glob import glob

def compute_metrics(gt_mask, pred_mask, threshold=0.5):
    gt = (gt_mask > 0).astype(np.uint8).flatten()
    pred = (pred_mask > threshold).astype(np.uint8).flatten()

    tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0, 1]).ravel()
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return iou, dice

def evaluate_predictions(pred_dir, gt_dir, output_log="outputs/eval_metrics.txt"):
    pred_files = sorted(glob(os.path.join(pred_dir, 'firemap_2016_05_*.tif')))
    results = []

    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        date_str = filename.split('_')[1] + '_' + filename.split('_')[2].replace('.tif', '')
        gt_path = os.path.join(gt_dir, f"stack_2016_{date_str}.tif")

        if not os.path.exists(gt_path):
            print(f"⚠️ Ground truth missing for {date_str}")
            continue

        with rasterio.open(pred_path) as p:
            pred_mask = p.read(1)

        with rasterio.open(gt_path) as gt:
            fire_mask = gt.read(10)  # 10th band = fire label

        iou, dice = compute_metrics(fire_mask, pred_mask)
        results.append((date_str, iou, dice))
        print(f"✅ {date_str} | IoU: {iou:.4f}, Dice: {dice:.4f}")

    # Save to log file
    with open(output_log, 'w') as f:
        for r in results:
            f.write(f"{r[0]}, IoU: {r[1]:.4f}, Dice: {r[2]:.4f}\n")
    print(f"\n📄 Evaluation summary saved to: {output_log}")
