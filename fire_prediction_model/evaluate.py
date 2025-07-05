#=========== Enhanced Evaluation Script ==========
import os
import rasterio
import numpy as np
import json
from datetime import datetime
from glob import glob
from utils.metrics import comprehensive_evaluation, print_evaluation_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_single_prediction(gt_mask, pred_mask, threshold=0.5):
    """Evaluate a single prediction with comprehensive metrics."""
    return comprehensive_evaluation(gt_mask, pred_mask, threshold)

def evaluate_predictions(pred_dir, gt_dir, output_dir="outputs/evaluation"):
    """
    Comprehensive evaluation of all predictions with detailed reporting.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pred_files = sorted(glob(os.path.join(pred_dir, 'firemap_2016_05_*.tif')))
    all_results = []
    cumulative_metrics = {}
    
    print(f"🔍 Evaluating {len(pred_files)} predictions...")
    
    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        date_str = filename.split('_')[1] + '_' + filename.split('_')[2].replace('.tif', '')
        gt_path = os.path.join(gt_dir, f"stack_2016_{date_str}.tif")

        if not os.path.exists(gt_path):
            print(f"⚠️ Ground truth missing for {date_str}")
            continue

        # Load data
        with rasterio.open(pred_path) as p:
            pred_mask = p.read(1)

        with rasterio.open(gt_path) as gt:
            fire_mask = gt.read(10)  # 10th band = fire label

        # Evaluate with comprehensive metrics
        metrics = evaluate_single_prediction(fire_mask, pred_mask)
        metrics['date'] = date_str
        metrics['filename'] = filename
        all_results.append(metrics)
        
        # Print individual results
        print(f"📊 {date_str}: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}, F1={metrics['f1_score']:.4f}")
    
    # Compute aggregate statistics
    aggregate_metrics = compute_aggregate_metrics(all_results)
    
    # Generate comprehensive report
    generate_evaluation_report(all_results, aggregate_metrics, output_dir)
    
    # Plot metrics trends
    plot_metrics_trends(all_results, output_dir)
    
    print(f"\n✅ Comprehensive evaluation completed!")
    print(f"📊 Mean IoU: {aggregate_metrics['mean_iou']:.4f} ± {aggregate_metrics['std_iou']:.4f}")
    print(f"📊 Mean Dice: {aggregate_metrics['mean_dice']:.4f} ± {aggregate_metrics['std_dice']:.4f}")
    print(f"📊 Mean F1: {aggregate_metrics['mean_f1']:.4f} ± {aggregate_metrics['std_f1']:.4f}")
    print(f"📄 Reports saved to: {output_dir}")

def compute_aggregate_metrics(results):
    """Compute mean and std of all metrics across predictions."""
    metrics_keys = ['iou', 'dice', 'f1_score', 'precision', 'recall', 'accuracy', 'auc_roc', 'auc_pr']
    aggregate = {}
    
    for key in metrics_keys:
        values = [r[key] for r in results]
        aggregate[f'mean_{key}'] = np.mean(values)
        aggregate[f'std_{key}'] = np.std(values)
        aggregate[f'min_{key}'] = np.min(values)
        aggregate[f'max_{key}'] = np.max(values)
    
    return aggregate

def generate_evaluation_report(results, aggregate_metrics, output_dir):
    """Generate detailed evaluation report in multiple formats."""
    
    # JSON report
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'total_samples': len(results),
        'individual_results': results,
        'aggregate_metrics': aggregate_metrics
    }
    
    with open(os.path.join(output_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Text report
    with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write("COMPREHENSIVE FIRE PREDICTION EVALUATION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples: {len(results)}\n\n")
        
        f.write("AGGREGATE METRICS:\n")
        f.write("-" * 30 + "\n")
        for metric in ['iou', 'dice', 'f1_score', 'precision', 'recall']:
            mean_val = aggregate_metrics[f'mean_{metric}']
            std_val = aggregate_metrics[f'std_{metric}']
            f.write(f"{metric.upper():12}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        f.write(f"\nDETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        for result in results:
            f.write(f"{result['date']}: IoU={result['iou']:.4f}, Dice={result['dice']:.4f}, F1={result['f1_score']:.4f}\n")

def plot_metrics_trends(results, output_dir):
    """Plot metrics trends over time."""
    dates = [r['date'] for r in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Fire Prediction Metrics Over Time', fontsize=16)
    
    metrics_to_plot = [
        ('iou', 'IoU Score'),
        ('dice', 'Dice Coefficient'), 
        ('f1_score', 'F1 Score'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('auc_roc', 'AUC-ROC')
    ]
    
    for idx, (metric, title) in enumerate(metrics_to_plot):
        row, col = idx // 3, idx % 3
        values = [r[metric] for r in results]
        
        axes[row, col].plot(range(len(dates)), values, 'o-', linewidth=2, markersize=6)
        axes[row, col].set_title(title)
        axes[row, col].set_ylabel(title)
        axes[row, col].set_xlabel('Sample Index')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_ylim(0, 1)
        
        # Add mean line
        mean_val = np.mean(values)
        axes[row, col].axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, 
                              label=f'Mean: {mean_val:.3f}')
        axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_trends.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    metrics_data = []
    metric_names = ['IoU', 'Dice', 'F1', 'Precision', 'Recall', 'AUC-ROC']
    
    for result in results:
        metrics_data.append([
            result['iou'], result['dice'], result['f1_score'],
            result['precision'], result['recall'], result['auc_roc']
        ])
    
    corr_matrix = np.corrcoef(np.array(metrics_data).T)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=metric_names, yticklabels=metric_names,
                fmt='.3f')
    plt.title('Metrics Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Example usage
    pred_dir = "outputs/predictions"
    gt_dir = "../fire-probability-prediction-map-unstacked-data/dataset_stacked"
    
    if os.path.exists(pred_dir) and os.path.exists(gt_dir):
        evaluate_predictions(pred_dir, gt_dir)
    else:
        print("⚠️ Prediction or ground truth directories not found.")
        print(f"Expected: {pred_dir} and {gt_dir}")
