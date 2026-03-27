docs/

# AI & ML Projects

Author: Somnath Saha

## Overview

This repository contains three of my machine learning projects, each fully automated with its own pipeline, tests, and evidence pack for submission. All code, results, and required screenshots are included and ready for review. I designed and implemented these projects to demonstrate my skills in supervised learning, unsupervised learning, and reinforcement learning.

## Repository Structure

- **project1_mlp_feature_clustering/**: MLP from scratch, feature extraction, clustering, and metrics
- **project2_aerial_image_clustering/**: Aerial image clustering with multiple algorithms and synthetic data
- **project3_rl_gymnasium_bot/**: Deep Q-Network (DQN) agent for CartPole-v1 using Gymnasium
- **shared/**: Reusable utilities for metrics, plotting, seeding, and more
- **submission/**: Contains all required screenshots and logs for submission
- **docs/**: (To be completed) Documentation for running, outputs, and troubleshooting

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run all experiments and generate submission pack:**
   - Project 1:
     ```bash
     python -m project1_mlp_feature_clustering.src.p1.experiments.run_all --seed 42
     ```
   - Project 2:
     ```bash
     python -m project2_aerial_image_clustering.src.p2.experiments.run_all --seed 42
     ```
   - Project 3 (train and evaluate):
     ```bash
     python -m project3_rl_gymnasium_bot.src.p3.train --env CartPole-v1 --seed 42 --episodes 300
     python -m project3_rl_gymnasium_bot.src.p3.eval --env CartPole-v1 --seed 42 --episodes 20 --checkpoint_path project3_rl_gymnasium_bot/data/checkpoints/dqn_cartpole.pt
     ```

## Submission Evidence Pack

All required screenshots for each project are automatically generated and copied to:
- `submission/screenshots/project1/`
- `submission/screenshots/project2/`
- `submission/screenshots/project3/`

Logs and summary tables (if generated) are in `submission/logs/` and `submission/tables/`.

## Project Overviews

### Project 1: MLP Feature Clustering
- Trained MLPs from scratch (NumPy) on Breast Cancer and Iris datasets
- Feature extraction: PCA, TF-IDF, HOG
- K-Means clustering and validity indices
- All results and plots saved to `reports/` and `submission/`

### Project 2: Aerial Image Clustering
- Synthetic aerial-like data generation
- Feature extraction and clustering (KMeans, GMM, Agglomerative, DBSCAN)
- Evaluation metrics and t-SNE visualizations
- All results and plots saved to `reports/` and `submission/`

### Project 3: RL Gymnasium Bot
- DQN agent for CartPole-v1 (PyTorch)
- Training, evaluation, and reward curve plots
- All results and plots saved to `reports/` and `submission/`

## Testing

Run all smoke tests:
```bash
pytest -q
```

## Documentation

See per-project README files and (if present) the `docs/` folder for detailed instructions, pipeline explanations, and troubleshooting tips.

---

**Date:** 2026-03-27

This repository contains three fully automated machine learning projects, each with its own pipeline, tests, and evidence pack for submission. All code, results, and required screenshots are included and ready for review.

## Repository Structure

- **project1_mlp_feature_clustering/**: MLP from scratch, feature extraction, clustering, and metrics
- **project2_aerial_image_clustering/**: Aerial image clustering with multiple algorithms and synthetic data
- **project3_rl_gymnasium_bot/**: Deep Q-Network (DQN) agent for CartPole-v1 using Gymnasium
- **shared/**: Reusable utilities for metrics, plotting, seeding, and more
- **submission/**: Contains all required screenshots and logs for submission
- **docs/**: (To be completed) Documentation for running, outputs, and troubleshooting

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run all experiments and generate submission pack:**
   - Project 1:
     ```bash
     python -m project1_mlp_feature_clustering.src.p1.experiments.run_all --seed 42
     ```
   - Project 2:
     ```bash
     python -m project2_aerial_image_clustering.src.p2.experiments.run_all --seed 42
     ```
   - Project 3 (train and evaluate):
     ```bash
     python -m project3_rl_gymnasium_bot.src.p3.train --env CartPole-v1 --seed 42 --episodes 300
     python -m project3_rl_gymnasium_bot.src.p3.eval --env CartPole-v1 --seed 42 --episodes 20 --checkpoint_path project3_rl_gymnasium_bot/data/checkpoints/dqn_cartpole.pt
     ```

## Submission Evidence Pack

All required screenshots for each project are automatically generated and copied to:
- `submission/screenshots/project1/`
- `submission/screenshots/project2/`
- `submission/screenshots/project3/`

Logs and summary tables (if generated) are in `submission/logs/` and `submission/tables/`.

## Project Overviews

### Project 1: MLP Feature Clustering
- Train MLPs from scratch (NumPy) on Breast Cancer and Iris datasets
- Feature extraction: PCA, TF-IDF, HOG
- K-Means clustering and validity indices
- All results and plots saved to `reports/` and `submission/`

### Project 2: Aerial Image Clustering
- Synthetic aerial-like data generation
- Feature extraction and clustering (KMeans, GMM, Agglomerative, DBSCAN)
- Evaluation metrics and t-SNE visualizations
- All results and plots saved to `reports/` and `submission/`

### Project 3: RL Gymnasium Bot
- DQN agent for CartPole-v1 (PyTorch)
- Training, evaluation, and reward curve plots
- All results and plots saved to `reports/` and `submission/`

## Testing

Run all smoke tests:
```bash
pytest -q
```

## Documentation

See per-project README files and (if present) the `docs/` folder for detailed instructions, pipeline explanations, and troubleshooting tips.

---

**Author:** Somnath Saha  
**Date:** 2026-03-27