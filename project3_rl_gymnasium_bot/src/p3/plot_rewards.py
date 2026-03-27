"""
Plot reward curve for DQN training
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_rewards(rewards_path="../../data/checkpoints/rewards.npy", out_path="../../reports/figures/p3_rewards_curve.png"):
    import pathlib
    project_root = pathlib.Path(__file__).resolve().parents[3]
    rewards_path = project_root / "data" / "checkpoints" / "rewards.npy"
    out_path = project_root / "reports" / "figures" / "p3_rewards_curve.png"
    rewards = np.load(str(rewards_path))
    plt.figure(figsize=(8,4))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN CartPole-v1 Training Reward Curve")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(str(out_path))
    plt.close()
    print(f"Reward curve saved to {out_path}")

if __name__ == "__main__":
    plot_rewards()
