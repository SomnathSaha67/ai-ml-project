
# Project 2: Aerial Image Clustering

Author: Somnath Saha

## Overview

In this project, I tackled the challenge of clustering aerial images using a variety of feature extraction and clustering algorithms. The pipeline is fully automated and generates synthetic aerial-like data if no raw images are present, ensuring reproducibility.

## Highlights

- Generated synthetic aerial image data for unsupervised clustering
- Extracted features using pixel baselines, texture descriptors, and PCA
- Compared KMeans, Gaussian Mixture, Agglomerative, and DBSCAN clustering
- Evaluated results with silhouette, Calinski-Harabasz, and Davies-Bouldin indices
- Visualized clusters using t-SNE and montage grids
- All results and plots are saved to `reports/` and key screenshots are copied to `submission/`

## How to Run

Run the full pipeline and generate results:
```bash
python -m project2_aerial_image_clustering.src.p2.experiments.run_all --seed 42
```

## Artifacts

- Metrics tables, t-SNE plots, and cluster montages in `reports/figures/`
- Submission-ready screenshots in `submission/screenshots/project2/`
- Metrics and results in `reports/results/`

## Testing

Run smoke tests:
```bash
pytest project2_aerial_image_clustering/tests/ -v
```

---

This project helped me practice unsupervised learning and computer vision, and to build robust, reproducible ML pipelines.
