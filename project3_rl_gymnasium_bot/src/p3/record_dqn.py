"""
Record DQN agent playing CartPole-v1 and save video
"""
import gymnasium as gym
import numpy as np
import torch
import os
import imageio
from agents.dqn_agent import DQNAgent

def record(checkpoint_path, out_dir="../../data/videos", seed=42):
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    agent.load(checkpoint_path)
    state, _ = env.reset()
    frames = []
    done = False
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, "dqn_cartpole_demo.mp4")
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    checkpoint = "../../data/checkpoints/dqn_cartpole_final.pth"
    record(checkpoint)
