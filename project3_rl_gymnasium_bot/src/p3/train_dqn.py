"""
Train DQN agent on CartPole-v1
"""
import gymnasium as gym
import numpy as np
import os
import torch
from agents.dqn_agent import DQNAgent

def train(seed=42, episodes=500, checkpoint_dir="../../data/checkpoints"):
    import pathlib
    project_root = pathlib.Path(__file__).resolve().parents[3]
    checkpoint_dir = project_root / "data" / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    env = gym.make('CartPole-v1')
    np.random.seed(seed)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            ep_reward += reward
        rewards.append(ep_reward)
        if (ep+1) % 50 == 0:
            agent.save(os.path.join(checkpoint_dir, f"dqn_cartpole_ep{ep+1}.pth"))
            print(f"Episode {ep+1}, Reward: {ep_reward}, Epsilon: {agent.epsilon:.3f}")
    # Save final model
    agent.save(str(checkpoint_dir / "dqn_cartpole_final.pth"))
    np.save(str(checkpoint_dir / "rewards.npy"), np.array(rewards))
    print("Training complete. Rewards saved.")

if __name__ == "__main__":
    train()
