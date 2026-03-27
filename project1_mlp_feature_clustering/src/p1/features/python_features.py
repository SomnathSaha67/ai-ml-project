"""Python feature extraction techniques."""

import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from skimage.feature import hog
from skimage.transform import resize

# Add paths
shared_path = Path(__file__).parent.parent.parent.parent / "shared" / "src"
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))

from shared.io import ensure_dir


def extract_features_pca(
    X: np.ndarray,
    n_components: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """
    Extract features using PCA (Principal Component Analysis).
    
    Args:
        X: Input data (n_samples, n_features).
        n_components: Number of PCA components.
        seed: Random seed.
        
    Returns:
        PCA-transformed data.
    """
    pca = PCA(n_components=min(n_components, X.shape[1]), random_state=seed)
    X_pca = pca.fit_transform(X)
    
    print(f"PCA: Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"PCA: Shape {X.shape} -> {X_pca.shape}")
    
    return X_pca


def extract_features_tfidf(
    documents: list = None,
    n_features: int = 100,
) -> np.ndarray:
    """
    Extract features using TF-IDF (Text Feature Extraction).
    
    If no documents provided, create synthetic text data.
    
    Args:
        documents: List of text documents (if None, generate synthetic).
        n_features: Maximum number of features.
        
    Returns:
        TF-IDF feature matrix.
    """
    if documents is None:
        # Generate synthetic text data
        words = ['cluster', 'feature', 'data', 'model', 'train', 
                 'test', 'sample', 'matrix', 'vector', 'neural']
        documents = []
        for i in range(50):
            doc = ' '.join(
                np.random.choice(words, size=np.random.randint(3, 8))
            )
            documents.append(doc)
    
    vectorizer = TfidfVectorizer(max_features=n_features, lowercase=True)
    X_tfidf = vectorizer.fit_transform(documents).toarray()
    
    print(f"TF-IDF: Shape {len(documents)} x {X_tfidf.shape[1]}")
    
    return X_tfidf


def extract_features_hog(
    images: np.ndarray = None,
    image_size: tuple = (64, 64),
) -> np.ndarray:
    """
    Extract features using HOG (Histogram of Oriented Gradients).
    
    If no images provided, create synthetic image data.
    
    Args:
        images: Array of images (n_images, height, width) or (n_images, height, width, 3).
        image_size: Size to resize images to.
        
    Returns:
        HOG feature matrix.
    """
    if images is None:
        # Generate synthetic images
        n_images = 50
        images = np.random.randint(0, 256, size=(n_images, 128, 128, 3), dtype=np.uint8)
    
    features = []
    for img in images:
        # Convert to grayscale if RGB
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        
        # Resize
        img_resized = resize(img, image_size)
        
        # Extract HOG features
        hog_features = hog(
            img_resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
        )
        features.append(hog_features)
    
    X_hog = np.array(features)
    
    print(f"HOG: Shape {len(images)} x {X_hog.shape[1]}")
    
    return X_hog


def demonstrate_feature_extraction(save_dir: Path = None):
    """
    Demonstrate feature extraction techniques.
    
    Args:
        save_dir: Directory to save examples.
    """
    from shared.plotting import set_plot_style
    import matplotlib.pyplot as plt
    
    set_plot_style()
    
    if save_dir is None:
        save_dir = Path(__file__).parent.parent.parent / "reports" / "figures"
    
    save_dir = ensure_dir(Path(save_dir))
    
    print("\n=== Feature Extraction Demonstration ===")
    
    # PCA on synthetic data
    print("\n1. PCA Feature Extraction:")
    X_synthetic = np.random.randn(100, 50)
    X_pca = extract_features_pca(X_synthetic, n_components=10, seed=42)
    
    # TF-IDF
    print("\n2. TF-IDF Feature Extraction:")
    X_tfidf = extract_features_tfidf(n_features=100)
    
    # HOG
    print("\n3. HOG Feature Extraction:")
    X_hog = extract_features_hog(image_size=(64, 64))
    
    print("\n=== Feature Extraction Complete ===")
    
    return {
        'pca': X_pca,
        'tfidf': X_tfidf,
        'hog': X_hog,
    }
