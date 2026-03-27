"""t-SNE visualization utilities."""

import numpy as np
from sklearn.manifold import TSNE


def run_tsne(
    X: np.ndarray,
    seed: int,
    perplexity: float = 30,
    random_state: int = None,
) -> np.ndarray:
    """
    Run t-SNE dimensionality reduction.
    
    Args:
        X: Input feature matrix (n_samples, n_features).
        seed: Seed for reproducibility.
        perplexity: Perplexity parameter.
        random_state: Random state (uses seed if not provided).
        
    Returns:
        2D t-SNE projection (n_samples, 2).
    """
    if random_state is None:
        random_state = seed
    
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, X.shape[0] - 1),
        random_state=random_state,
        verbose=0,
    )
    return tsne.fit_transform(X)
