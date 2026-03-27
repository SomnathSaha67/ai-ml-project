"""Plotting utilities and styling."""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union


def set_plot_style() -> None:
    """Set plot style for consistency."""
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10


def savefig(fig: plt.Figure, path: Union[str, Path], dpi: int = 150) -> None:
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib figure.
        path: File path.
        dpi: DPI for saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
