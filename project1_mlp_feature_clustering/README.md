# Project 1: MLP Feature Clustering

Author: Somnath Saha

## Overview

In this project, I implemented a Multi-Layer Perceptron (MLP) from scratch using NumPy and compared it with scikit-learn baselines. I also explored feature extraction techniques and clustering on classic datasets. The project demonstrates my understanding of neural networks, unsupervised learning, and practical Python engineering.

## Highlights

- Built and trained MLPs on the Breast Cancer and Iris datasets
- Implemented feature extraction: PCA, TF-IDF (on synthetic text), and HOG (on images)
- Developed K-Means clustering from scratch, including k-means++ initialization
- Evaluated clustering with silhouette, Calinski-Harabasz, and Davies-Bouldin indices
- Integrated an optional Weka bridge for extra feature engineering (skips gracefully if Weka is not installed)
- All results and plots are saved to `reports/` and key screenshots are copied to `submission/`

## How to Run

Run all experiments and generate results:
```bash
python -m project1_mlp_feature_clustering.src.p1.experiments.run_all --seed 42
```

## Artifacts

- Training curves, confusion matrices, ROC curves, and clustering visualizations in `reports/figures/`
- Submission-ready screenshots in `submission/screenshots/project1/`
- Metrics and results in `reports/results/`

## Testing

Run smoke tests:
```bash
pytest project1_mlp_feature_clustering/tests/ -v
```

---

This project was a great opportunity to deepen my understanding of both supervised and unsupervised learning, and to practice building ML pipelines from scratch.
pytest project1_mlp_feature_clustering/tests/ -v
```

## Output Artifacts

Generated in `project1_mlp_feature_clustering/reports/`:

### Figures (`figures/`)
- `p1_mlp_loss_breast_cancer.png` - Training loss curve
- `p1_mlp_accuracy_breast_cancer.png` - Training accuracy curve
- `p1_confusion_matrix_breast_cancer.png` - Confusion matrix heatmap
- `p1_roc_breast_cancer.png` - ROC curve
- `p1_mlp_loss_iris.png` - Iris training loss
- `p1_mlp_accuracy_iris.png` - Iris training accuracy
- `p1_confusion_matrix_iris.png` - Iris confusion matrix
- `p1_choose_k_indices.png` - Validity indices vs K
- `p1_tsne_clusters.png` - t-SNE visualizaton of clusters

### Results (`results/`)
- `mlp_history_breast_cancer.json` - Training history
- `mlp_metrics_breast_cancer.json` - Test metrics
- `mlp_history_iris.json` - Iris training history
- `mlp_metrics_iris.json` - Iris test metrics
- `choose_k_results.csv` - Validity indices for K=2..10
- `clustering_metrics.json` - Best clustering configuration

## Model Architecture

### MLP Defaults
- **Input**: Dataset features
- **Hidden Layers**: [64, 32]
- **Output**: Number of classes
- **Activation**: ReLU (hidden), Sigmoid/Softmax (output)
- **Epochs**: 100
- **Batch Size**: 32
- **Learning Rate**: 0.01

### K-Means Defaults
- **K Range**: 2 to 10
- **Initialization**: k-means++
- **Max Iterations**: 300

## Configuration

See `.env.example` for environment variables, including:
- `RANDOM_SEED`: Global seed for reproducibility
- `WEKA_JAR_PATH`: Optional path to Weka JAR for feature extraction
