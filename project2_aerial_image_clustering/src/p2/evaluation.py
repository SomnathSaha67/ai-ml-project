"""
Project 2: Clustering Evaluation
"""
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Dict

def evaluate_all_clusterings(clusterings: dict, features_dict: dict, seed: int, results_dir) -> dict:
    metrics_table = {}
    for feat_name, algo_dict in clusterings.items():
        metrics_table[feat_name] = {}
        X = features_dict[feat_name]
        for algo, labels in algo_dict.items():
            metrics = {}
            if len(set(labels)) > 1 and (labels != -1).any():
                try:
                    metrics['silhouette'] = float(silhouette_score(X, labels))
                except Exception:
                    metrics['silhouette'] = None
                try:
                    metrics['calinski_harabasz'] = float(calinski_harabasz_score(X, labels))
                except Exception:
                    metrics['calinski_harabasz'] = None
                try:
                    metrics['davies_bouldin'] = float(davies_bouldin_score(X, labels))
                except Exception:
                    metrics['davies_bouldin'] = None
            else:
                metrics['silhouette'] = None
                metrics['calinski_harabasz'] = None
                metrics['davies_bouldin'] = None
            metrics_table[feat_name][algo] = metrics
    return metrics_table
