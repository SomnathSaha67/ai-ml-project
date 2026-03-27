"""Screenshot pack generation for submissions."""

import shutil
from pathlib import Path
from typing import List, Dict, Any
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from .io import ensure_dir, save_json


def copy_figures_to_submission(
    src_figures_dir: Path,
    project_name: str,
    file_mappings: Dict[str, str],
) -> None:
    """
    Copy selected figures from project reports to submission.
    
    Args:
        src_figures_dir: Source figures directory.
        project_name: Project name (project1, project2, project3).
        file_mappings: Dict mapping source filename to target filename.
    """
    # Get root and submission dirs
    root = src_figures_dir.parent.parent.parent.parent
    submission_screenshots = root / "submission" / "screenshots" / project_name
    
    ensure_dir(submission_screenshots)
    
    for src_name, tgt_name in file_mappings.items():
        src_path = src_figures_dir / src_name
        tgt_path = submission_screenshots / tgt_name
        
        if src_path.exists():
            shutil.copy(src_path, tgt_path)
            print(f"Copied {src_name} -> {tgt_name}")


def create_terminal_screenshot(
    title: str,
    content: str,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """
    Create a matplotlib figure containing terminal-style text.
    
    Args:
        title: Title for the screenshot.
        content: Text content to display.
        output_path: Path to save PNG.
        dpi: DPI for saving.
    """
    ensure_dir(output_path.parent)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Create title
    fig.text(0.5, 0.95, title, ha='center', fontsize=14, fontweight='bold',
             family='monospace')
    
    # Create content
    fig.text(0.05, 0.85, content, ha='left', va='top', fontsize=9,
             family='monospace', wrap=True)
    
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_metrics_table_image(
    dataframe,
    title: str,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """
    Save a pandas DataFrame as an image.
    
    Args:
        dataframe: Pandas DataFrame to visualize.
        title: Title for the table.
        output_path: Path to save PNG.
        dpi: DPI for saving.
    """
    import pandas as pd
    
    ensure_dir(output_path.parent)
    
    fig, ax = plt.subplots(figsize=(10, len(dataframe) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style header
    for i in range(len(dataframe.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Add title
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def generate_submission_pack(
    project_name: str,
    project_root: Path = None,
) -> None:
    """
    Generate submission evidence pack for a project.
    
    Args:
        project_name: Project name (project1, project2, project3).
        project_root: Root project directory.
    """
    if project_root is None:
        # Infer from this file's location (shared/src/shared)
        project_root = Path(__file__).resolve().parents[4]
    submission_dir = project_root / "submission"
    ensure_dir(submission_dir)
    
    print(f"\nGenerating submission pack for {project_name}...")
    print(f"Submission directory: {submission_dir}")

    # Project 2: copy all main figures
    if project_name == "project2":
        figures_dir = project_root / "project2_aerial_image_clustering" / "src" / "reports" / "figures"
        file_mappings = {
            "p2_tsne_kmeans.png": "p2_tsne_kmeans.png",
            "p2_tsne_gmm.png": "p2_tsne_gmm.png",
            "p2_tsne_agg.png": "p2_tsne_agg.png",
            "p2_tsne_dbscan.png": "p2_tsne_dbscan.png",
            "p2_cluster_montage_kmeans.png": "p2_cluster_montage_kmeans.png",
            "p2_metrics_table.png": "p2_metrics_table.png"
        }
        copy_figures_to_submission(figures_dir, project_name, file_mappings)
