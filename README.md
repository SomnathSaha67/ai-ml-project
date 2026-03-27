# ml-ai-lab — AI-Executable Spec (Projects 1–3) + Screenshot Submission Pack

User login: **SomnathSaha67**  
Spec date: **2026-03-27**  
Goal: After reading this README, an AI coding agent should be able to implement **all deliverables** for Projects 1–3 end-to-end in one go **and automatically generate the required “screenshots” evidence** as PNG files.

---

## 0) Non-Negotiable Rules (Agent Must Follow)

1. **Do not change the repository structure** defined in this README.
2. Put reusable utilities in `shared/src/shared/` and import them from projects.
3. Every experiment must:
   - set a deterministic seed
   - save metrics to `reports/results/` as JSON (and optionally CSV)
   - save plots to `reports/figures/` as PNG
4. Implement **both**:
   - a “from scratch with NumPy” version where required
   - and a scikit-learn baseline where useful (for comparison)
5. Provide minimal tests (smoke tests) for the critical modules.
6. Code must run on CPU and finish quickly on a typical laptop:
   - Project 1 MLP: < ~2 minutes on small datasets
   - Project 3 RL (CartPole): a few minutes (configurable)
7. Prefer clarity over micro-optimizations. Add docstrings + type hints.
8. **Screenshots requirement:** since submissions require screenshots, the agent must generate a **submission evidence pack** automatically (Section 11).  
   - Do **not** rely on manual screenshots.
   - Save “screenshot-like” evidence as **PNG exports** of plots + **terminal-style text images** where needed.

---

## 1) Repo Structure (Create Exactly This)

Create the following directories and files:

```text
README.md
requirements.txt
pyproject.toml
.gitignore
.env.example

docs/
  00_overview.md
  01_how_to_run.md
  02_metrics_and_plots.md
  03_data_guidelines.md
  04_submission_screenshots.md

submission/
  README.md
  screenshots/
    project1/
    project2/
    project3/
  logs/
  tables/

shared/
  src/shared/
    __init__.py
    config.py
    io.py
    plotting.py
    metrics.py
    tsne.py
    clustering_validity.py
    seed.py
    screenshot_pack.py
  tests/
    test_imports.py

project1_mlp_feature_clustering/
  README.md
  data/
    raw/
    interim/
    processed/
  notebooks/
  src/p1/
    __init__.py
    datasets.py
    features/
      __init__.py
      python_features.py
      weka_bridge.py
    models/
      __init__.py
      mlp_numpy.py
      optimizers.py
      train_mlp.py
    clustering/
      __init__.py
      kmeans_numpy.py
      choose_k.py
    experiments/
      __init__.py
      run_all.py
  reports/
    figures/
    results/
  tests/
    test_mlp_numpy_smoke.py
    test_kmeans_smoke.py

project2_aerial_image_clustering/
  README.md
  data/
    raw/
    processed/
  notebooks/
  src/p2/
    __init__.py
    datasets.py
    preprocessing.py
    features.py
    clusterers.py
    evaluation.py
    visualize.py
    experiments/
      __init__.py
      run_all.py
  reports/
    figures/
    results/
  tests/
    test_pipeline_smoke.py

project3_rl_gymnasium_bot/
  README.md
  data/
    videos/
    checkpoints/
  notebooks/
  src/p3/
    __init__.py
    train.py
    eval.py
    record.py
    agents/
      __init__.py
      dqn.py
    utils.py
  reports/
    figures/
    results/
  tests/
    test_dqn_smoke.py
```

---

## 2) Dependencies

### 2.1 `requirements.txt` (agent must create)
Must include at least:
- numpy
- matplotlib
- scikit-learn
- scipy
- pandas
- seaborn
- tqdm
- gymnasium
- imageio
- pillow
- pytest

Optional (only if used, keep default working without them):
- torch (recommended for DQN stability)
- hdbscan
- umap-learn

### 2.2 `pyproject.toml`
Provide minimal config so `pytest` works and `src/` imports are clean.

---

## 3) Shared Utilities (Must Implement)

### `shared/src/shared/seed.py`
- `set_global_seed(seed: int) -> None` to seed numpy, python `random`, and any other libs used.

### `shared/src/shared/io.py`
- `ensure_dir(path)`
- `save_json(path, obj)`
- `load_json(path)`
- `save_csv(path, dataframe)` (optional convenience)

### `shared/src/shared/plotting.py`
- `savefig(fig, path, dpi=150)`
- `set_plot_style()` (seaborn/matplotlib style)

### `shared/src/shared/metrics.py`
Implement:
- `confusion_matrix(y_true, y_pred, labels=None)` (can wrap sklearn)
- `precision_recall_f1(y_true, y_pred, average="binary"|"macro"|"micro")`
- `roc_auc(y_true, y_score)`:
  - binary: AUC
  - multiclass: One-vs-rest macro AUC
Plotters:
- confusion matrix heatmap
- ROC curve(s)

### `shared/src/shared/tsne.py`
- `run_tsne(X, seed, perplexity=30) -> X2d` (wrap sklearn TSNE)

### `shared/src/shared/clustering_validity.py`
Implement (wrap sklearn if desired, but unify):
- silhouette score
- calinski-harabasz
- davies-bouldin
- helper `evaluate_k_range(...)` returning a table/dict

---

## 4) Project 1 — MLP (NumPy) + Feature Extraction + Clustering

### 4.1 Datasets (no downloads required)
In `project1.../src/p1/datasets.py`:
- Breast Cancer Wisconsin (binary classification)
- Iris (multiclass)
Return `(X_train, X_test, y_train, y_test)` with scaling.

### 4.2 MLP from scratch (NumPy)
In `models/mlp_numpy.py`:
- configurable hidden layers
- hidden activation: ReLU or tanh
- output: sigmoid (binary) or softmax (multiclass)
- loss: BCE and CCE
- backprop + (mini-)batch gradient descent
Expose:
- `fit(...)` returning training history
- `predict(...)`
- `predict_proba(...)`

In `models/train_mlp.py`:
- training loop records loss + accuracy (train/val)
- save:
  - `project1.../reports/results/mlp_history_<dataset>.json`
  - plots: loss curve + accuracy curve

### 4.3 Evaluation metrics (required)
Compute and save:
- confusion matrix
- precision
- recall
- F1
- ROC curve + AUC (binary + multiclass macro-ovr)
Save:
- `project1.../reports/results/mlp_metrics_<dataset>.json`
- plots in `project1.../reports/figures/`

### 4.4 Feature extraction (Python + Weka)
In `features/python_features.py` implement at least 3 techniques:
1) PCA (tabular)
2) TF-IDF (use a tiny synthetic text dataset if no text data)
3) HOG or LBP (use generated synthetic images or sklearn sample images)

In `features/weka_bridge.py`:
- Weka CLI bridge (subprocess-based)
- Must be optional: if Weka not installed, skip gracefully and log instructions.
- Document `WEKA_JAR_PATH` in `.env.example`.

### 4.5 K-Means clustering (NumPy)
In `clustering/kmeans_numpy.py` implement:
- random or kmeans++ init
- assignment/update loop
- inertia
- outputs: labels, centroids

### 4.6 Choose K using validity indices
In `clustering/choose_k.py`:
- K range (2..10)
- plot silhouette/CH/DB (and optional inertia elbow)
- save results JSON/CSV and plots

### 4.7 t-SNE visualization + silhouette
- t-SNE scatter colored by cluster
- compute silhouette score and save it

### 4.8 Single entrypoint
`project1.../src/p1/experiments/run_all.py` runs:
1) MLP + metrics on Breast Cancer
2) MLP + metrics on Iris
3) Feature extraction demos
4) KMeans + choose K + t-SNE + silhouette

---

## 5) Project 2 — Clustering of Aerial Images (Comparison)

### 5.1 Data handling
Default must run without external data:
- If `data/raw` is empty, generate a **synthetic aerial-like dataset** (textures/patterns) deterministically.

### 5.2 Feature extraction
In `src/p2/features.py` implement:
- flattened pixels baseline
- simple texture features
- optional PCA

### 5.3 Compare clustering algorithms
In `src/p2/clusterers.py` implement:
- KMeans
- Gaussian Mixture Model
- Agglomerative
- DBSCAN (handle noise label -1)

### 5.4 Evaluation + visualization
In `src/p2/evaluation.py`:
- silhouette (when valid)
- calinski-harabasz
- davies-bouldin

In `src/p2/visualize.py`:
- t-SNE plot colored by cluster labels
- montage/grid of example patches per cluster

### 5.5 Single entrypoint
`project2.../src/p2/experiments/run_all.py`:
- build/load dataset
- extract features
- run each algorithm
- save metrics table + plots for each algo

---

## 6) Project 3 — Reinforcement Learning Bot (Gymnasium)

### 6.1 Default environment
- `CartPole-v1`

### 6.2 Agent
Implement DQN in `agents/dqn.py`.
- Prefer PyTorch for stability (add `torch` to requirements if used).
- Must include replay buffer, epsilon-greedy, target network.

### 6.3 Train/Eval/Record scripts
- `src/p3/train.py`: train + save checkpoint + reward curve + summary JSON
- `src/p3/eval.py`: evaluate checkpoint + save JSON
- `src/p3/record.py` (recommended): record short video to `data/videos/`

All scripts accept CLI args:
- `--env`
- `--seed`
- `--episodes`
- `--checkpoint_path`

---

## 7) Tests (Smoke Tests Required)
- import checks
- tiny run on very small data
- keep fast
Command:
```bash
pytest -q
```

---

## 8) Documentation Requirements
Create:
- `/docs/*.md` for running, outputs, troubleshooting
- per-project READMEs with run commands and expected artifacts
- `docs/04_submission_screenshots.md` explaining the evidence pack

---

## 9) “One-Go” Execution Commands (Must Work)

### Project 1
```bash
python -m project1_mlp_feature_clustering.src.p1.experiments.run_all --seed 42
```

### Project 2
```bash
python -m project2_aerial_image_clustering.src.p2.experiments.run_all --seed 42
```

### Project 3
```bash
python -m project3_rl_gymnasium_bot.src.p3.train --env CartPole-v1 --seed 42 --episodes 300
python -m project3_rl_gymnasium_bot.src.p3.eval --env CartPole-v1 --seed 42 --episodes 20 --checkpoint_path project3_rl_gymnasium_bot/data/checkpoints/dqn_cartpole.pt
```

---

## 10) Acceptance Checklist (Agent Must Satisfy)

- [ ] Repo structure exactly matches Section 1
- [ ] Project 1: NumPy MLP + training plots + CM/Prec/Rec/F1/AUC
- [ ] Project 1: feature extraction demos + Weka bridge (optional execution)
- [ ] Project 1: KMeans NumPy + choose K validity indices + t-SNE + silhouette
- [ ] Project 2: multiple clustering algos compared + visualizations + metrics
- [ ] Project 3: Gymnasium RL bot (DQN) train/eval + checkpoint + plots
- [ ] All scripts save outputs to `reports/figures` and `reports/results`
- [ ] `pytest -q` passes
- [ ] **Submission evidence pack generated** (Section 11)

---

## 11) Submission Evidence Pack (Automatic “Screenshots”)

### 11.1 What counts as “screenshots”
Your instructor asked to “upload screenshots”. Instead of manual screenshots, this repo must generate:
- PNG images of all key plots (already saved in `reports/figures/`)
- PLUS a curated copy of the most important figures into:
  - `submission/screenshots/project1/`
  - `submission/screenshots/project2/`
  - `submission/screenshots/project3/`
- PLUS terminal-style evidence logs saved to:
  - `submission/logs/` as `.txt`
- PLUS summary tables saved to:
  - `submission/tables/` as `.csv` (and optional `.json`)

### 11.2 Required screenshot set (minimum)

**Project 1 (must exist in `submission/screenshots/project1/`):**
- `p1_mlp_loss_breast_cancer.png`
- `p1_mlp_accuracy_breast_cancer.png`
- `p1_confusion_matrix_breast_cancer.png`
- `p1_roc_breast_cancer.png`
- `p1_mlp_loss_iris.png`
- `p1_mlp_accuracy_iris.png`
- `p1_confusion_matrix_iris.png`
- `p1_choose_k_indices.png` (silhouette/CH/DB vs K)
- `p1_tsne_clusters.png`

**Project 2 (must exist in `submission/screenshots/project2/`):**
- `p2_tsne_kmeans.png`
- `p2_tsne_gmm.png`
- `p2_tsne_agglomerative.png`
- `p2_tsne_dbscan.png` (if DBSCAN used)
- `p2_cluster_montage_best.png`
- `p2_metrics_table.png` (table rendered as image OR a clean CSV in `submission/tables/` plus a screenshot image)

**Project 3 (must exist in `submission/screenshots/project3/`):**
- `p3_reward_curve.png`
- `p3_eval_summary.png` (terminal-style image OR a matplotlib text panel)
Optional:
- `p3_episode_frame.png` (single frame screenshot) OR `data/videos/*.mp4`

### 11.3 Implementation requirement: screenshot pack generator
Implement `shared/src/shared/screenshot_pack.py` with functions to:
- copy selected figures from each project `reports/figures/` into `submission/screenshots/...`
- generate “terminal screenshot” images:
  - create a matplotlib figure containing monospace text (command + key metrics)
  - save as PNG to submission screenshots
- write logs:
  - save the printed output of each run into `submission/logs/*.txt`
- write a `submission/README.md` that lists all artifacts and where they are.

### 11.4 Single command to build the submission pack
Each project `run_all.py` (and Project 3 train/eval) must call the screenshot pack generator so that after running:
- the project reports are generated
- the `submission/` folder is populated automatically

---

## 12) Final Instruction to the Agent
Implement the entire repository according to this README, satisfying every checklist item, **including the Submission Evidence Pack**, without changing the structure.

END OF SPEC.