"""Clustering validity metrics."""

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from typing import Dict, Any, Tuple
import pandas as pd


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute silhouette score.
    
    Args:
        X: Feature matrix.
        labels: Cluster labels.
        
    Returns:
        Silhouette score.
    """
    if len(np.unique(labels)) < 2:
        return -1.0
    return silhouette_score(X, labels)


def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Calinski-Harabasz index.
    
    Args:
        X: Feature matrix.
        labels: Cluster labels.
        
    Returns:
        Calinski-Harabasz index.
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    return calinski_harabasz_score(X, labels)


def davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Davies-Bouldin index.
    
    Args:
        X: Feature matrix.
        labels: Cluster labels.
        
    Returns:
        Davies-Bouldin index.
    """
    if len(np.unique(labels)) < 2:
        return float('inf')
    return davies_bouldin_score(X, labels)


def evaluate_k_range(
    X: np.ndarray,
    labels_dict: Dict[int, np.ndarray],
    k_range: range = None,
) -> pd.DataFrame:
    """
    Evaluate clustering validity for multiple K values.
    
    Args:
        X: Feature matrix.
        labels_dict: Dictionary mapping K to labels.
        k_range: Range of K values (if None, derived from labels_dict).
        
    Returns:
        DataFrame with validity metrics.
    """
    if k_range is None:
        k_range = sorted(labels_dict.keys())
    
    results = []
    for k in k_range:
        if k not in labels_dict:
            continue
        labels = labels_dict[k]
        
        # Compute metrics
        sil = silhouette(X, labels)
        ch = calinski_harabasz(X, labels)
        db = davies_bouldin(X, labels)
        
        results.append({
            'K': k,
            'Silhouette': sil,
            'Calinski-Harabasz': ch,
            'Davies-Bouldin': db,
        })
    
    return pd.DataFrame(results)
