"""
Evaluate DQN agent on CartPole-v1
"""
import gymnasium as gym
import numpy as np
import torch
import os
from agents.dqn_agent import DQNAgent

def evaluate(checkpoint_path, episodes=10, seed=42):
    env = gym.make('CartPole-v1', render_mode=None)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    agent.load(checkpoint_path)
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            ep_reward += reward
        rewards.append(ep_reward)
        print(f"Episode {ep+1}, Reward: {ep_reward}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    return rewards

if __name__ == "__main__":
    checkpoint = "../../data/checkpoints/dqn_cartpole_final.pth"
    evaluate(checkpoint)
