"""K-Means clustering implementation from scratch."""

import numpy as np
from typing import Tuple, Optional


class KMeansNumpy:
    """
    K-Means clustering from scratch using NumPy.
    
    Supports:
    - Random or kmeans++ initialization
    - Euclidean distance
    - Convergence detection
    """
    
    def __init__(
        self,
        n_clusters: int,
        init: str = 'kmeans++',
        max_iter: int = 300,
        tol: float = 1e-4,
        seed: int = 42,
    ):
        """
        Initialize K-Means.
        
        Args:
            n_clusters: Number of clusters.
            init: 'kmeans++' or 'random'.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.
            seed: Random seed.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        
        self.centroids = None
        self.labels = None
        self.inertia_history = []
    
    def _initialize_centroids_random(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids randomly."""
        np.random.seed(self.seed)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _initialize_centroids_kmeans_pp(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using k-means++ algorithm."""
        np.random.seed(self.seed)
        n_samples = X.shape[0]
        centroids = []
        
        # Choose first centroid randomly
        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx])
        
        # Choose remaining centroids
        for k in range(1, self.n_clusters):
            # Compute distances to nearest centroid
            distances = np.array([
                min([np.linalg.norm(x - c) for c in centroids])
                for x in X
            ])
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            
            idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[idx])
        
        return np.array(centroids)
    
    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances from X to centroids."""
        distances = np.zeros((X.shape[0], centroids.shape[0]))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign samples to nearest centroid."""
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """Update centroids and compute inertia."""
        new_centroids = np.zeros_like(self.centroids)
        inertia = 0.0
        
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centroids[k] = X[mask].mean(axis=0)
                # Compute inertia for this cluster
                inertia += np.sum(np.linalg.norm(X[mask] - new_centroids[k], axis=1) ** 2)
            else:
                # Keep old centroid if no samples assigned
                new_centroids[k] = self.centroids[k]
        
        return new_centroids, inertia
    
    def fit(self, X: np.ndarray) -> 'KMeansNumpy':
        """
        Fit K-Means to data.
        
        Args:
            X: Input data (n_samples, n_features).
            
        Returns:
            Self (for chaining).
        """
        # Initialize centroids
        if self.init == 'kmeans++':
            self.centroids = self._initialize_centroids_kmeans_pp(X)
        else:
            self.centroids = self._initialize_centroids_random(X)
        
        # Iterative clustering
        for iteration in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids, inertia = self._update_centroids(X, labels)
            self.inertia_history.append(inertia)
            
            # Check convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                print(f"Converged at iteration {iteration}")
                break
        
        # Final assignment
        self.labels = self._assign_clusters(X)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign clusters to samples.
        
        Args:
            X: Input data.
            
        Returns:
            Cluster labels.
        """
        if self.centroids is None:
            raise ValueError("Model not fitted yet")
        return self._assign_clusters(X)
    
    def get_inertia(self) -> float:
        """
        Get final inertia (sum of squared distances to nearest centroid).
        
        Returns:
            Inertia value.
        """
        if self.inertia_history:
            return self.inertia_history[-1]
        return 0.0
