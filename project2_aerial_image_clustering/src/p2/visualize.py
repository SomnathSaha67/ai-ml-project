"""
Project 2: Visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from skimage.util import montage as skimage_montage

def plot_all_tsne(clusterings, features_dict, figures_dir, seed=42):
    for feat_name, algo_dict in clusterings.items():
        X = features_dict[feat_name]
        X2d = TSNE(n_components=2, random_state=seed).fit_transform(X)
        for algo, labels in algo_dict.items():
            plt.figure(figsize=(6, 5))
            plt.scatter(X2d[:, 0], X2d[:, 1], c=labels, cmap='tab10', s=20)
            plt.title(f't-SNE: {feat_name} + {algo}')
            plt.tight_layout()
            # Capitalize algo name for filename to match test expectation
            fname = f"p2_tsne_{algo.capitalize()}.png"
            plt.savefig(os.path.join(figures_dir, fname))
            plt.close()

def plot_cluster_montage(clusterings, images, figures_dir):
    # Example: montage for best clustering (first found)
    for feat_name, algo_dict in clusterings.items():
        for algo, labels in algo_dict.items():
            if len(set(labels)) > 1:
                idxs = np.argsort(labels)
                ordered_imgs = images[idxs][:64]
                # Remove multichannel argument for compatibility
                m = skimage_montage(ordered_imgs, channel_axis=-1)
                plt.figure(figsize=(8, 8))
                plt.imshow(m.astype(np.uint8))
                plt.axis('off')
                plt.title(f'Montage: {feat_name} + {algo}')
                fname = f"p2_cluster_montage_{algo}.png"
                plt.savefig(os.path.join(figures_dir, fname))
                plt.close()
                break
        break

def save_metrics_table_image(metrics_table, figures_dir):
    import pandas as pd
    import matplotlib.pyplot as plt
    rows = []
    for feat, algos in metrics_table.items():
        for algo, metrics in algos.items():
            row = {'feature': feat, 'algo': algo}
            row.update(metrics)
            rows.append(row)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 2+len(df)*0.3))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'p2_metrics_table.png'))
    plt.close()
    df.to_csv(os.path.join(figures_dir, 'p2_metrics_table.csv'), index=False)
