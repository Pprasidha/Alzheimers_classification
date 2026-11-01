"""
Model Evaluation Script for Alzheimer's Stage Classification
Generates comprehensive clinical metrics and Confusion Matrix heatmap.
"""

import os
import argparse
import torch
import numpy as np

from models.cnn_model import AlzheimerCNN
from utils.dataset import load_dataset_splits, CLASS_NAMES, DISPLAY_NAMES
from utils.metrics import compute_metrics, plot_confusion_matrix


def evaluate_model(checkpoint_path="models/checkpoints/best_alzheimer_model.pth",
                   data_dir="data/mri_dataset",
                   output_cm_path="models/confusion_matrix.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating using device: {device}")

    # Load data
    _, _, test_loader = load_dataset_splits(data_dir, batch_size=16)

    # Load model
    model = AlzheimerCNN(in_channels=3, num_classes=4).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Run train.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    metrics, report, cm = compute_metrics(all_targets, all_preds, all_probs)

    print("\n" + "="*50)
    print("        TEST SET EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy:       {metrics['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy:      {metrics['balanced_accuracy']*100:.2f}%")
    print(f"Macro F1-Score:         {metrics['macro_f1']*100:.2f}%")
    print(f"Weighted Precision:     {metrics['weighted_precision']*100:.2f}%")
    print(f"Weighted Recall:        {metrics['weighted_recall']*100:.2f}%")
    print("\nDetailed Clinical Classification Report:\n")
    print(report)

    # Save Confusion Matrix
    os.makedirs(os.path.dirname(output_cm_path), exist_ok=True)
    plot_confusion_matrix(cm, save_path=output_cm_path, normalize=True)
    print(f"[OK] Saved normalized confusion matrix plot to '{output_cm_path}'")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Alzheimer MRI Model")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_alzheimer_model.pth")
    parser.add_argument("--data_dir", type=str, default="data/mri_dataset")
    parser.add_argument("--out_cm", type=str, default="models/confusion_matrix.png")
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.data_dir, args.out_cm)
