"""IO utilities for saving and loading data."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path.
        
    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Union[str, Path], obj: Any) -> None:
    """
    Save object to JSON file.
    
    Args:
        path: File path.
        obj: Object to save.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path: Union[str, Path]) -> Any:
    """
    Load object from JSON file.
    
    Args:
        path: File path.
        
    Returns:
        Loaded object.
    """
    path = Path(path)
    with open(path, 'r') as f:
        return json.load(f)


def save_csv(path: Union[str, Path], dataframe: pd.DataFrame) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        path: File path.
        dataframe: DataFrame to save.
    """
    path = Path(path)
    ensure_dir(path.parent)
    dataframe.to_csv(path, index=False)


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        path: File path.
        
    Returns:
        Loaded DataFrame.
    """
    path = Path(path)
    return pd.read_csv(path)
