
# Project 3: RL Gymnasium Bot (DQN)

Author: Somnath Saha

## Overview

In this project, I implemented a Deep Q-Network (DQN) agent to solve the CartPole-v1 environment using Gymnasium and PyTorch. The project covers the full RL pipeline from training to evaluation and evidence generation.

## Highlights

- Built a DQN agent with replay buffer, epsilon-greedy policy, and target network
- Trained and evaluated the agent on CartPole-v1, saving checkpoints and reward curves
- All results and plots are saved to `reports/` and key screenshots are copied to `submission/`
- Optionally, the agent can record videos of its performance

## How to Run

Train the agent:
```bash
python -m project3_rl_gymnasium_bot.src.p3.train --env CartPole-v1 --seed 42 --episodes 300
```

Evaluate the agent:
```bash
python -m project3_rl_gymnasium_bot.src.p3.eval --env CartPole-v1 --seed 42 --episodes 20 --checkpoint_path project3_rl_gymnasium_bot/data/checkpoints/dqn_cartpole.pt
```

## Artifacts

- Reward curves, evaluation summaries, and (optionally) videos in `reports/figures/` and `data/videos/`
- Submission-ready screenshots in `submission/screenshots/project3/`
- Metrics and results in `reports/results/`

## Testing

Run smoke tests:
```bash
pytest project3_rl_gymnasium_bot/tests/ -v
```

---

This project was a hands-on way to learn about reinforcement learning, deep learning, and reproducible experimentation in Python.
