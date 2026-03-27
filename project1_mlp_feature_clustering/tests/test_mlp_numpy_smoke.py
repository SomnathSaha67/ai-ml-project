"""Project 1 smoke tests."""

import sys
from pathlib import Path
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
shared_path = project_root / "shared" / "src"
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))

from project1_mlp_feature_clustering.src.p1.models.mlp_numpy import MLPNumpy
from project1_mlp_feature_clustering.src.p1.clustering.kmeans_numpy import KMeansNumpy


def test_mlp_numpy_smoke():
    """Smoke test for MLP NumPy."""
    # Create small synthetic dataset
    X_train = np.random.randn(50, 10)
    y_train = np.random.randint(0, 2, 50)
    X_test = np.random.randn(10, 10)
    y_test = np.random.randint(0, 2, 10)
    
    # Create and train MLP
    mlp = MLPNumpy(
        input_size=10,
        hidden_sizes=[8],
        output_size=2,
        hidden_activation='relu',
        output_activation='sigmoid',
        seed=42,
    )
    
    history = mlp.fit(X_train, y_train, epochs=5, batch_size=16)
    
    assert len(history['train_loss']) == 5
    assert len(history['train_accuracy']) == 5
    
    # Test predictions
    y_pred = mlp.predict(X_test)
    assert y_pred.shape == (10,)
    assert np.all((y_pred >= 0) & (y_pred <= 1))
    
    # Test probabilities
    y_proba = mlp.predict_proba(X_test)
    assert y_proba.shape == (10, 2)
    
    print("✓ MLP NumPy smoke test passed")


def test_kmeans_smoke():
    """Smoke test for K-Means NumPy."""
    # Create small synthetic dataset
    X = np.random.randn(50, 5)
    
    # Fit K-Means
    kmeans = KMeansNumpy(n_clusters=3, init='kmeans++', seed=42)
    kmeans.fit(X)
    
    assert kmeans.labels is not None
    assert len(kmeans.labels) == 50
    assert len(np.unique(kmeans.labels)) <= 3
    
    # Test prediction
    X_new = np.random.randn(10, 5)
    labels = kmeans.predict(X_new)
    assert len(labels) == 10
    
    # Test inertia
    inertia = kmeans.get_inertia()
    assert inertia >= 0
    
    print("✓ K-Means NumPy smoke test passed")


if __name__ == '__main__':
    test_mlp_numpy_smoke()
    test_kmeans_smoke()
    print("\nAll smoke tests passed!")
