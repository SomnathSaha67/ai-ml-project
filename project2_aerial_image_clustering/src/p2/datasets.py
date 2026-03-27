"""
Project 2: Aerial Image Clustering - Dataset Loader/Generator
"""
import numpy as np
from typing import Tuple

def load_or_generate_aerial_dataset(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads aerial images from data/raw, or generates synthetic data if empty.
    Returns:
        X: (n_samples, n_features) array
        images: (n_samples, H, W, C) array
    """
    # TODO: Implement file loading, fallback to synthetic
    np.random.seed(seed)
    n_samples, H, W, C = 100, 32, 32, 3
    images = np.random.randint(0, 255, size=(n_samples, H, W, C), dtype=np.uint8)
    X = images.reshape(n_samples, -1)
    return X, images
