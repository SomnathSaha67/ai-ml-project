"""
Project 2: Feature Extraction
"""
import numpy as np
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage.color import rgb2gray
from typing import Dict

def extract_all_features(X: np.ndarray, images: np.ndarray, seed: int = 42) -> Dict[str, np.ndarray]:
    features = {}
    # Baseline: flattened pixels
    features['pixels'] = X
    # Texture: HOG
    hog_feats = []
    for img in images:
        gray = rgb2gray(img)
        hog_feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        hog_feats.append(hog_feat)
    features['hog'] = np.array(hog_feats)
    # PCA
    pca = PCA(n_components=20, random_state=seed)
    features['pca'] = pca.fit_transform(X)
    return features
