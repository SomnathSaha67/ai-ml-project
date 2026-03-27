# Project 1: MLP Feature Clustering

Train MLPs from scratch using NumPy, extract features with multiple techniques, and cluster data using K-Means.

## Experiments

### 1. MLP Training
- **Breast Cancer Dataset**: Binary classification
- **Iris Dataset**: Multiclass classification (3 classes)
- Network: Configurable hidden layers with ReLU activation
- Optimizer: Adam or SGD
- Loss: Binary Cross-Entropy or Categorical Cross-Entropy

### 2. Feature Extraction
- **PCA**: Principal Component Analysis for dimensionality reduction
- **TF-IDF**: Text feature extraction (synthetic text examples)
- **HOG**: Histogram of Oriented Gradients for image features

### 3. K-Means Clustering
- Implementation from scratch using NumPy
- K-means++ initialization
- Validity indices:
  - Silhouette Score (higher is better)
  - Calinski-Harabasz Index (higher is better)
  - Davies-Bouldin Index (lower is better)
- Elbow method with inertia

### 4. Weka Bridge
- Optional CLI integration for Weka filters
- Graceful fallback if Weka not installed

## Run Instructions

### Run all experiments:
```bash
python -m project1_mlp_feature_clustering.src.p1.experiments.run_all --seed 42
```

### Run tests:
```bash
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
