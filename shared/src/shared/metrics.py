"""Metrics computation utilities."""

import numpy as np
from sklearn.metrics import (
    confusion_matrix as cm,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Label order.
        
    Returns:
        Confusion matrix.
    """
    return cm(y_true, y_pred, labels=labels)


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        average: Averaging method ('binary', 'macro', 'micro', 'weighted').
        
    Returns:
        Tuple of (precision, recall, f1).
    """
    if average not in ["binary", "macro", "micro", "weighted"]:
        average = "weighted"
    
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return float(prec), float(rec), float(f1)


def roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    multi_class: str = "ovr",
) -> float:
    """
    Compute ROC AUC score.
    Args:
        y_true: True labels.
        y_score: Target scores (probabilities).
        multi_class: Method for multiclass ('ovr', 'ovo').
    Returns:
        ROC AUC score.
    """
    try:
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            # Binary classification
            if y_score.ndim > 1 and y_score.shape[1] == 2:
                scores = y_score[:, 1]
            else:
                scores = y_score.ravel()
            return float(roc_auc_score(y_true, scores))
        else:
            # Multiclass
            return float(roc_auc_score(y_true, y_score, multi_class=multi_class))
    except Exception as e:
        print(f"roc_auc error: {e}")
        return 0.0


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Label names.
        
    Returns:
        Matplotlib figure.
    """
    import seaborn as sns
    
    cm_mat = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        xticklabels=labels if labels else "auto",
        yticklabels=labels if labels else "auto",
    )
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> plt.Figure:
    """
    Plot ROC curve(s).
    Args:
        y_true: True labels.
        y_score: Target scores (probabilities).
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        # Binary classification
        if y_score.ndim > 1 and y_score.shape[1] == 2:
            scores = y_score[:, 1]
        else:
            scores = y_score.ravel()
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc_value = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_value:.3f})')
    else:
        # Multiclass - plot one-vs-rest for each class
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc_value = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_value:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    return fig
