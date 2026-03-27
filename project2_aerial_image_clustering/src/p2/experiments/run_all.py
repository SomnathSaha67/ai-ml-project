"""
Main entrypoint for Project 2: Aerial Image Clustering
"""
import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add shared path (robust for subprocess and direct execution)
# Quick fix: Always add absolute shared/src and project2_aerial_image_clustering/src to sys.path
import os
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
shared_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'shared', 'src'))
project2_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
for p in [shared_src_path, project2_src_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

from shared.seed import set_global_seed
from shared.plotting import set_plot_style
from shared.io import ensure_dir, save_json
from shared.screenshot_pack import generate_submission_pack

from p2.datasets import load_or_generate_aerial_dataset
from p2.features import extract_all_features
from p2.clusterers import run_all_clusterers
from p2.evaluation import evaluate_all_clusterings
from p2.visualize import plot_all_tsne, plot_cluster_montage, save_metrics_table_image

def run_all(seed: int = 42):
    set_global_seed(seed)
    set_plot_style()
    project_dir = Path(__file__).parent.parent.parent
    reports_dir = ensure_dir(project_dir / "reports")
    figures_dir = ensure_dir(reports_dir / "figures")
    results_dir = ensure_dir(reports_dir / "results")

    print("\n" + "="*60)
    print("PROJECT 2: AERIAL IMAGE CLUSTERING")
    print("="*60)

    # 1. Data
    X, images = load_or_generate_aerial_dataset(seed=seed)

    # 2. Features
    features_dict = extract_all_features(X, images, seed=seed)

    # 3. Clustering
    clusterings = run_all_clusterers(features_dict, seed=seed)

    # 4. Evaluation
    metrics_table = evaluate_all_clusterings(clusterings, features_dict, seed=seed, results_dir=results_dir)

    # 5. Visualization
    plot_all_tsne(clusterings, features_dict, figures_dir=figures_dir, seed=seed)
    plot_cluster_montage(clusterings, images, figures_dir=figures_dir)
    save_metrics_table_image(metrics_table, figures_dir=figures_dir)

    # 6. Submission pack
    generate_submission_pack(project_name="project2")

    print("\n" + "="*60)
    print("PROJECT 2 COMPLETE")
    print("="*60)
    print(f"Results saved to: {results_dir}")
    print(f"Figures saved to: {figures_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Project 2 experiments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    run_all(seed=args.seed)
