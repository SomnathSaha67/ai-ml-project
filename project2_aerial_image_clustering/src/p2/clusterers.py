"""
Project 2: Clustering Algorithms
"""
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from typing import Dict

def run_all_clusterers(features_dict: dict, seed: int = 42) -> Dict[str, dict]:
    clusterings = {}
    for feat_name, X in features_dict.items():
        clusterings[feat_name] = {}
        # KMeans
        kmeans = KMeans(n_clusters=4, random_state=seed)
        clusterings[feat_name]['kmeans'] = kmeans.fit_predict(X)
        # GMM
        gmm = GaussianMixture(n_components=4, random_state=seed)
        clusterings[feat_name]['gmm'] = gmm.fit_predict(X)
        # Agglomerative
        agg = AgglomerativeClustering(n_clusters=4)
        clusterings[feat_name]['agg'] = agg.fit_predict(X)
        # DBSCAN
        dbscan = DBSCAN(eps=5, min_samples=5)
        clusterings[feat_name]['dbscan'] = dbscan.fit_predict(X)
    return clusterings
