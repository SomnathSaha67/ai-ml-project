"""Global seed setting for reproducibility."""

import numpy as np
import random


def set_global_seed(seed: int) -> None:
    """
    Set global random seed for reproducibility.
    
    Args:
        seed: Random seed value to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass
