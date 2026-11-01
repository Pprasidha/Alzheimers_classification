"""
Evaluation Metrics & Visualization for Alzheimer's Stage Classification
Computes Confusion Matrix, Precision, Recall, F1-Score, and ROC-AUC.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, balanced_accuracy_score
from utils.dataset import DISPLAY_NAMES, CLASS_NAMES


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Computes a comprehensive suite of classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    macro_f1 = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[2]

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(b_acc),
        "weighted_precision": float(prec),
        "weighted_recall": float(rec),
        "weighted_f1": float(f1),
        "macro_f1": float(macro_f1)
    }

    report = classification_report(y_true, y_pred, target_names=DISPLAY_NAMES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return metrics, report, cm


def plot_confusion_matrix(cm, save_path=None, normalize=True):
    """
    Plots and saves an annotated confusion matrix heatmap.
    """
    plt.figure(figsize=(9, 7))
    if normalize:
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title("Normalized Confusion Matrix (%) - Alzheimer's Stages", fontsize=14, pad=15)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title("Confusion Matrix (Sample Counts) - Alzheimer's Stages", fontsize=14, pad=15)

    plt.ylabel('True Clinical Stage', fontsize=12)
    plt.xlabel('Predicted Stage', fontsize=12)
    plt.xticks(rotation=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
