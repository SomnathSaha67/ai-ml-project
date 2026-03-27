"""Test imports for shared module."""

import sys
from pathlib import Path

# Add shared module to path
shared_path = Path(__file__).parent.parent / "src"
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))


def test_seed_import():
    """Test that seed module can be imported."""
    from shared import seed
    assert hasattr(seed, 'set_global_seed')


def test_io_import():
    """Test that io module can be imported."""
    from shared import io
    assert hasattr(io, 'ensure_dir')
    assert hasattr(io, 'save_json')
    assert hasattr(io, 'load_json')


def test_plotting_import():
    """Test that plotting module can be imported."""
    from shared import plotting
    assert hasattr(plotting, 'set_plot_style')
    assert hasattr(plotting, 'savefig')


def test_metrics_import():
    """Test that metrics module can be imported."""
    from shared import metrics
    assert hasattr(metrics, 'confusion_matrix')
    assert hasattr(metrics, 'precision_recall_f1')
    assert hasattr(metrics, 'roc_auc')


def test_clustering_validity_import():
    """Test that clustering_validity module can be imported."""
    from shared import clustering_validity
    assert hasattr(clustering_validity, 'silhouette')
    assert hasattr(clustering_validity, 'calinski_harabasz')
    assert hasattr(clustering_validity, 'davies_bouldin')


def test_tsne_import():
    """Test that tsne module can be imported."""
    from shared import tsne
    assert hasattr(tsne, 'run_tsne')


if __name__ == '__main__':
    test_seed_import()
    test_io_import()
    test_plotting_import()
    test_metrics_import()
    test_clustering_validity_import()
    test_tsne_import()
    print("All shared module imports successful!")
