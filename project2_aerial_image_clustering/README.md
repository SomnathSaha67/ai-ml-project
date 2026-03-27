# Project 2: Aerial Image Clustering

This project clusters aerial images using multiple algorithms and feature extraction techniques. It is fully automated and generates synthetic data if no raw images are present.

## How to Run

```bash
python -m project2_aerial_image_clustering.src.p2.experiments.run_all --seed 42
```

## Output Artifacts

- All metrics and plots are saved in `reports/figures/` and `reports/results/`.
- Submission screenshots and logs are auto-copied to `submission/` after each run.

## Pipeline

1. **Data Handling**: Loads or generates synthetic aerial-like images.
2. **Feature Extraction**: Baseline (pixels), texture features, PCA.
3. **Clustering Algorithms**: KMeans, GMM, Agglomerative, DBSCAN.
4. **Evaluation**: Silhouette, Calinski-Harabasz, Davies-Bouldin.
5. **Visualization**: t-SNE plots, cluster montages, metrics table.

## Key Files
- `src/p2/experiments/run_all.py`: Main entrypoint.
- `src/p2/datasets.py`: Data loading/generation.
- `src/p2/features.py`: Feature extraction.
- `src/p2/clusterers.py`: Clustering algorithms.
- `src/p2/evaluation.py`: Metrics.
- `src/p2/visualize.py`: Plots and montages.

## Tests

```bash
pytest project2_aerial_image_clustering/tests/ -v
```

## Submission Evidence

See `submission/screenshots/project2/` and `submission/tables/` for required PNGs and CSVs.
