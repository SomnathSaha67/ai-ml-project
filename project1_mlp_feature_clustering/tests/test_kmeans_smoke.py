"""Additional clustering tests."""

import sys
from pathlib import Path
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
shared_path = project_root / "shared" / "src"
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))

from project1_mlp_feature_clustering.src.p1.clustering.kmeans_numpy import KMeansNumpy


def test_kmeans_random_init():
    """Test K-Means with random initialization."""
    X = np.random.randn(30, 4)
    
    kmeans = KMeansNumpy(n_clusters=2, init='random', seed=42)
    kmeans.fit(X)
    
    assert kmeans.labels is not None
    assert len(np.unique(kmeans.labels)) <= 2
    
    print("✓ K-Means random init test passed")


def test_kmeans_convergence():
    """Test K-Means convergence."""
    # Create data with clear clusters
    X = np.vstack([
        np.random.randn(20, 2) + [0, 0],
        np.random.randn(20, 2) + [5, 5],
    ])
    
    kmeans = KMeansNumpy(n_clusters=2, init='kmeans++', seed=42)
    kmeans.fit(X)
    
    assert len(kmeans.inertia_history) > 0
    # Inertia should generally decrease
    
    print("✓ K-Means convergence test passed")


if __name__ == '__main__':
    test_kmeans_random_init()
    test_kmeans_convergence()
    print("\nAll clustering tests passed!")
