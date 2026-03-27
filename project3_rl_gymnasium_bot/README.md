# Project 3: RL Gymnasium Bot (DQN)

This project implements a Deep Q-Network (DQN) agent for the CartPole-v1 environment using Gymnasium. All training, evaluation, and recording scripts are provided and save outputs for submission.

## How to Run

### Train
```bash
python -m project3_rl_gymnasium_bot.src.p3.train --env CartPole-v1 --seed 42 --episodes 300
```

### Evaluate
```bash
python -m project3_rl_gymnasium_bot.src.p3.eval --env CartPole-v1 --seed 42 --episodes 20 --checkpoint_path project3_rl_gymnasium_bot/data/checkpoints/dqn_cartpole.pt
```

## Output Artifacts
- Training/evaluation metrics and plots in `reports/figures/` and `reports/results/`
- Submission screenshots and logs are auto-copied to `submission/` after each run

## Pipeline
1. **DQN Agent**: PyTorch-based, with replay buffer, epsilon-greedy, and target network
2. **Training**: Reward curve, checkpoint, summary JSON
3. **Evaluation**: Loads checkpoint, evaluates, saves summary
4. **Recording**: Optionally records video to `data/videos/`

## Key Files
- `src/p3/train.py`: Training script
- `src/p3/eval.py`: Evaluation script
- `src/p3/record.py`: Video recording
- `src/p3/agents/dqn.py`: DQN agent implementation
- `src/p3/utils.py`: Utilities

## Tests
```bash
pytest project3_rl_gymnasium_bot/tests/ -v
```

## Submission Evidence
See `submission/screenshots/project3/` and `submission/logs/` for required PNGs and logs.
