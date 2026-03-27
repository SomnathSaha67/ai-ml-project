"""Training script for MLP on datasets."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, Any

# Add paths
shared_path = Path(__file__).parent.parent.parent.parent / "shared" / "src"
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))

from shared.seed import set_global_seed
from shared.io import ensure_dir, save_json
from shared.plotting import set_plot_style, savefig
from shared.metrics import confusion_matrix, precision_recall_f1, roc_auc, plot_confusion_matrix, plot_roc_curve

from ..datasets import load_breast_cancer_dataset, load_iris_dataset
from .mlp_numpy import MLPNumpy


def train_mlp_on_dataset(
    dataset_name: str,
    hidden_sizes: list = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    seed: int = 42,
    save_dir: Path = None,
) -> Dict[str, Any]:
    """
    Train MLP on a dataset.
    
    Args:
        dataset_name: 'breast_cancer' or 'iris'
        hidden_sizes: Hidden layer sizes (if None, use defaults).
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Learning rate.
        seed: Random seed.
        save_dir: Directory to save results.
        
    Returns:
        Dictionary with results.
    """
    set_global_seed(seed)
    set_plot_style()
    
    if hidden_sizes is None:
        hidden_sizes = [64, 32]
    
    if save_dir is None:
        save_dir = Path(__file__).parent.parent.parent / "reports"
    
    save_dir = Path(save_dir)
    figures_dir = ensure_dir(save_dir / "figures")
    results_dir = ensure_dir(save_dir / "results")
    
    # Load dataset
    if dataset_name == 'breast_cancer':
        X_train, X_test, y_train, y_test = load_breast_cancer_dataset(seed=seed)
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        output_activation = 'sigmoid' if n_classes == 2 else 'softmax'
        output_size = 1 if n_classes == 2 else n_classes
    elif dataset_name == 'iris':
        X_train, X_test, y_train, y_test = load_iris_dataset(seed=seed)
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        output_activation = 'softmax'
        output_size = n_classes
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create and train MLP
    mlp = MLPNumpy(
        input_size=n_features,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        hidden_activation='relu',
        output_activation=output_activation,
        seed=seed,
    )
    
    print(f"\nTraining MLP on {dataset_name}...")
    print(f"Dataset shape: {X_train.shape}")
    print(f"Number of classes: {n_classes}")
    
    history = mlp.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_name='adam',
    )
    
    # Save history
    history_to_save = {
        k: [float(v) for v in vals] for k, vals in history.items()
    }
    history_path = results_dir / f"mlp_history_{dataset_name}.json"
    save_json(history_path, history_to_save)
    print(f"Saved history to {history_path}")
    
    # Plot loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if history['val_loss']:
        ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'MLP Training Loss - {dataset_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    loss_path = figures_dir / f"p1_mlp_loss_{dataset_name}.png"
    savefig(fig, loss_path)
    print(f"Saved loss plot to {loss_path}")
    
    # Plot accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_accuracy'], label='Train Accuracy', linewidth=2)
    if history['val_accuracy']:
        ax.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'MLP Training Accuracy - {dataset_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    acc_path = figures_dir / f"p1_mlp_accuracy_{dataset_name}.png"
    savefig(fig, acc_path)
    print(f"Saved accuracy plot to {acc_path}")
    
    # Evaluate on test set
    y_pred = mlp.predict(X_test)
    y_pred_proba = mlp.predict_proba(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig = plot_confusion_matrix(y_test, y_pred)
    cm_path = figures_dir / f"p1_confusion_matrix_{dataset_name}.png"
    savefig(fig, cm_path)
    print(f"Saved confusion matrix to {cm_path}")
    
    # Metrics
    prec, rec, f1 = precision_recall_f1(y_test, y_pred, average='weighted')
    roc_auc_score = roc_auc(y_test, y_pred_proba)
    
    # ROC plot
    fig = plot_roc_curve(y_test, y_pred_proba)
    roc_path = figures_dir / f"p1_roc_{dataset_name}.png"
    savefig(fig, roc_path)
    print(f"Saved ROC curve to {roc_path}")
    
    # Save metrics
    metrics = {
        'dataset': dataset_name,
        'n_features': int(n_features),
        'n_classes': int(n_classes),
        'hidden_sizes': hidden_sizes,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'test_accuracy': float(np.mean(y_pred == y_test)),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(roc_auc_score),
        'confusion_matrix': cm.tolist(),
    }
    
    metrics_path = results_dir / f"mlp_metrics_{dataset_name}.json"
    save_json(metrics_path, metrics)
    print(f"Saved metrics to {metrics_path}")
    
    return {
        'mlp': mlp,
        'metrics': metrics,
        'history': history,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }
