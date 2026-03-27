"""Choose optimal K for K-Means using validity indices."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

# Add paths
shared_path = Path(__file__).parent.parent.parent.parent / "shared" / "src"
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))

from shared.io import ensure_dir, save_json, save_csv
from shared.plotting import set_plot_style, savefig
from shared.clustering_validity import (
    silhouette, calinski_harabasz, davies_bouldin, evaluate_k_range
)

from .kmeans_numpy import KMeansNumpy


def choose_k_for_clustering(
    X: np.ndarray,
    k_range: range = range(2, 11),
    seed: int = 42,
    save_dir: Path = None,
) -> Dict:
    """
    Choose optimal K using validity indices.
    
    Args:
        X: Input data.
        k_range: Range of K values to test.
        seed: Random seed.
        save_dir: Directory to save results.
        
    Returns:
        Dictionary with best K and evaluation results.
    """
    set_plot_style()
    
    if save_dir is None:
        save_dir = Path(__file__).parent.parent.parent / "reports"
    
    save_dir = Path(save_dir)
    figures_dir = ensure_dir(save_dir / "figures")
    results_dir = ensure_dir(save_dir / "results")
    
    print(f"\nEvaluating K-Means for K in {list(k_range)}...")
    
    # Fit K-Means for each K
    labels_dict = {}
    results = []
    
    for k in k_range:
        kmeans = KMeansNumpy(n_clusters=k, init='kmeans++', seed=seed)
        kmeans.fit(X)
        labels_dict[k] = kmeans.labels
        
        # Compute validity indices
        sil = silhouette(X, kmeans.labels)
        ch = calinski_harabasz(X, kmeans.labels)
        db = davies_bouldin(X, kmeans.labels)
        inertia = kmeans.get_inertia()
        
        results.append({
            'K': k,
            'Silhouette': sil,
            'Calinski-Harabasz': ch,
            'Davies-Bouldin': db,
            'Inertia': inertia,
        })
        
        print(f"K={k}: Silhouette={sil:.4f}, CH={ch:.2f}, DB={db:.4f}")
    
    results_df = pd.DataFrame(results)
    
    # Save results
    csv_path = results_dir / "choose_k_results.csv"
    save_csv(csv_path, results_df)
    print(f"Saved results to {csv_path}")
    
    json_path = results_dir / "choose_k_results.json"
    save_json(json_path, results_df.to_dict(orient='records'))
    
    # Find best K
    best_k_silhouette = results_df.loc[results_df['Silhouette'].idxmax(), 'K']
    best_k_ch = results_df.loc[results_df['Calinski-Harabasz'].idxmax(), 'K']
    best_k_db = results_df.loc[results_df['Davies-Bouldin'].idxmin(), 'K']
    
    print(f"\nBest K by Silhouette: {best_k_silhouette}")
    print(f"Best K by Calinski-Harabasz: {best_k_ch}")
    print(f"Best K by Davies-Bouldin: {best_k_db}")
    
    # Plot indices
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Silhouette
    axes[0, 0].plot(results_df['K'], results_df['Silhouette'], marker='o', linewidth=2)
    axes[0, 0].axvline(best_k_silhouette, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('K')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].set_title('Silhouette Score vs K')
    axes[0, 0].grid(alpha=0.3)
    
    # Calinski-Harabasz
    axes[0, 1].plot(results_df['K'], results_df['Calinski-Harabasz'], marker='o', linewidth=2)
    axes[0, 1].axvline(best_k_ch, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('K')
    axes[0, 1].set_ylabel('Calinski-Harabasz Index')
    axes[0, 1].set_title('Calinski-Harabasz Index vs K')
    axes[0, 1].grid(alpha=0.3)
    
    # Davies-Bouldin
    axes[1, 0].plot(results_df['K'], results_df['Davies-Bouldin'], marker='o', linewidth=2)
    axes[1, 0].axvline(best_k_db, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('K')
    axes[1, 0].set_ylabel('Davies-Bouldin Index')
    axes[1, 0].set_title('Davies-Bouldin Index vs K (lower is better)')
    axes[1, 0].grid(alpha=0.3)
    
    # Inertia (Elbow)
    axes[1, 1].plot(results_df['K'], results_df['Inertia'], marker='o', linewidth=2)
    axes[1, 1].set_xlabel('K')
    axes[1, 1].set_ylabel('Inertia')
    axes[1, 1].set_title('Elbow Method - Inertia vs K')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    indices_path = figures_dir / "p1_choose_k_indices.png"
    savefig(fig, indices_path)
    print(f"Saved indices plot to {indices_path}")
    
    return {
        'best_k_silhouette': int(best_k_silhouette),
        'best_k_ch': int(best_k_ch),
        'best_k_db': int(best_k_db),
        'labels_dict': labels_dict,
        'results': results_df.to_dict(orient='records'),
    }
