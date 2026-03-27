"""
Smoke test for Project 3 DQN pipeline
"""
import os
import sys
import pytest

def test_imports():
    import torch
    import gymnasium
    import imageio
    import matplotlib
    from agents.dqn_agent import DQNAgent

def test_train_script():
    import subprocess
    root = os.path.dirname(os.path.dirname(__file__))
    train_py = os.path.join(root, 'src', 'p3', 'train_dqn.py')
    result = subprocess.run([sys.executable, train_py, '--episodes', '2'], capture_output=True, text=True)
    assert result.returncode == 0, f"Train failed: {result.stderr}"
    # Check for checkpoint and rewards
    ckpt_dir = os.path.join(root, 'data', 'checkpoints')
    assert os.path.exists(os.path.join(ckpt_dir, 'dqn_cartpole_final.pth'))
    assert os.path.exists(os.path.join(ckpt_dir, 'rewards.npy'))
