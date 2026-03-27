"""Datasets for Project 1."""

import sys
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# Add shared module to path
shared_path = Path(__file__).parent.parent.parent.parent / "shared" / "src"
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))

from shared.seed import set_global_seed


def load_breast_cancer_dataset(
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split breast cancer dataset.
    
    Args:
        test_size: Test set fraction.
        seed: Random seed.
        
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    set_global_seed(seed)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def load_iris_dataset(
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split iris dataset.
    
    Args:
        test_size: Test set fraction.
        seed: Random seed.
        
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    set_global_seed(seed)
    
    data = load_iris()
    X, y = data.data, data.target
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
