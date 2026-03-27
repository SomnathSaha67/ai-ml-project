"""
Run all Project 1 experiments.

Main entrypoint for MLP, feature extraction, and clustering.
"""

import sys
from pathlib import Path
import argparse

# Add paths
project_root = Path(__file__).parent.parent.parent.parent.parent
shared_path = project_root / "shared" / "src"
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))

from shared.seed import set_global_seed
from shared.tsne import run_tsne
from shared.plotting import set_plot_style
from shared.io import ensure_dir, save_json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..models.train_mlp import train_mlp_on_dataset
from ..clustering.choose_k import choose_k_for_clustering
from ..features.python_features import demonstrate_feature_extraction
from ..features.weka_bridge import demonstrate_weka_bridge


def run_all_experiments(seed: int = 42):
    """
    Run all Project 1 experiments.
    
    Args:
        seed: Random seed for reproducibility.
    """
    set_global_seed(seed)
    set_plot_style()
    
    project_dir = Path(__file__).parent.parent.parent
    reports_dir = ensure_dir(project_dir / "reports")
    figures_dir = ensure_dir(reports_dir / "figures")
    results_dir = ensure_dir(reports_dir / "results")
    
    print("\n" + "="*60)
    print("PROJECT 1: MLP + FEATURE EXTRACTION + CLUSTERING")
    print("="*60)
    
    # 1. Train MLP on Breast Cancer
    print("\n\n[1/5] Training MLP on Breast Cancer dataset...")
    bc_results = train_mlp_on_dataset(
        'breast_cancer',
        hidden_sizes=[64, 32],
        epochs=100,
        batch_size=32,
        learning_rate=0.01,
        seed=seed,
        save_dir=reports_dir,
    )
    
    # 2. Train MLP on Iris
    print("\n[2/5] Training MLP on Iris dataset...")
    iris_results = train_mlp_on_dataset(
        'iris',
        hidden_sizes=[64, 32],
        epochs=100,
        batch_size=16,
        learning_rate=0.01,
        seed=seed,
        save_dir=reports_dir,
    )
    
    # 3. Feature Extraction
    print("\n[3/5] Demonstrating feature extraction techniques...")
    features = demonstrate_feature_extraction(save_dir=figures_dir)
    
    # 4. Weka Bridge
    print("\n[4/5] Demonstrating Weka bridge...")
    demonstrate_weka_bridge()
    
    # 5. K-Means Clustering on combined features
    print("\n[5/5] K-Means clustering with validity indices...")
    
    # Use Breast Cancer data for clustering
    X_train = bc_results['X_train']
    X_test = bc_results['X_test']
    X_combined = np.vstack([X_train, X_test])
    
    # Standardize
    X_mean = X_combined.mean(axis=0)
    X_std = X_combined.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_normalized = (X_combined - X_mean) / X_std
    
    # Choose K
    k_results = choose_k_for_clustering(
        X_normalized,
        k_range=range(2, 11),
        seed=seed,
        save_dir=reports_dir,
    )
    
    # Use best K from silhouette score
    best_k = k_results['best_k_silhouette']
    print(f"\n\nUsing K={best_k} (best by Silhouette score)")
    
    # Get best labels
    best_labels = k_results['labels_dict'][best_k]
    
    # t-SNE visualization
    print("Computing t-SNE visualization...")
    from shared.clustering_validity import silhouette as compute_silhouette
    
    X_tsne = run_tsne(X_normalized, seed=seed, perplexity=30)
    
    # Plot t-SNE colored by clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=best_labels, 
                         cmap='viridis', s=100, alpha=0.7, edgecolors='k')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f't-SNE Visualization - K={best_k} Clusters')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    from shared.plotting import savefig
    tsne_path = figures_dir / "p1_tsne_clusters.png"
    savefig(fig, tsne_path)
    print(f"Saved t-SNE visualization to {tsne_path}")
    
    # Compute silhouette score
    silhouette_score = compute_silhouette(X_normalized, best_labels)
    print(f"Silhouette Score (K={best_k}): {silhouette_score:.4f}")
    
    # Save clustering results
    clustering_metrics = {
        'best_k': int(best_k),
        'silhouette_score': float(silhouette_score),
        'n_samples': int(X_normalized.shape[0]),
        'n_features': int(X_normalized.shape[1]),
    }
    clustering_path = results_dir / "clustering_metrics.json"
    save_json(clustering_path, clustering_metrics)
    print(f"Saved clustering metrics to {clustering_path}")
    
    print("\n" + "="*60)
    print("PROJECT 1 EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {reports_dir}")
    print(f"Figures saved to: {figures_dir}")
    print(f"Metrics saved to: {results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Project 1 experiments')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    run_all_experiments(seed=args.seed)
