"""
Smoke test for Project 2 pipeline
"""
import os
import sys
import pytest

def test_run_all_pipeline():
    # Run the main pipeline script and check for output files
    import subprocess
    root = os.path.dirname(os.path.dirname(__file__))
    run_all = os.path.join(root, 'src', 'p2', 'experiments', 'run_all.py')
    result = subprocess.run([sys.executable, run_all], capture_output=True, text=True)
    assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
    # Check for expected output files
    figures = os.path.join(root, 'reports', 'figures')
    assert os.path.exists(os.path.join(figures, 'p2_tsne_KMeans.png'))
    assert os.path.exists(os.path.join(figures, 'p2_metrics_table.png'))
