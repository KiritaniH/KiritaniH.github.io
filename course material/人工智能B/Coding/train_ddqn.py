""""
CartPole Training & Evaluation (PyTorch + Gymnasium)
---------------------------------------------------
- 适配DDQN算法，支持训练/评估DDQN模型
"""

from __future__ import annotations
import os
import time
import numpy as np
import gymnasium as gym
import torch

# -----------------------------
# 替换原DQN导入为DDQN
# -----------------------------
from agents.cartpole_ddqn import DDQNSolver, DDQNConfig
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
# 修改模型路径为DDQN专属（避免覆盖原DQN模型）
MODEL_PATH = os.path.join(MODEL_DIR, "cartpole_ddqn.torch")


def train(num_episodes: int = 500, terminal_penalty: bool = True) -> DDQNSolver:
    """训练DDQN agent（仅替换DQN为DDQN）"""
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # -----------------------------
    # 替换为DDQNSolver和DDQNConfig
    # -----------------------------
    agent = DDQNSolver(obs_dim, act_dim, cfg=DDQNConfig())
    print(f"[Info] Using device: {agent.device}")

    for run in range(1, num_episodes + 1):
        state, info = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0

        while True:
            steps += 1
            action = agent.act(state)
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated


            
            next_state = np.reshape(next_state_raw, (1, obs_dim))
            agent.step(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Run: {run}, Epsilon: {agent.exploration_rate:.3f}, Score: {steps}")
                logger.add_score(steps, run)
                break

    env.close()
    agent.save(MODEL_PATH)
    print(f"[Train] Model saved to {MODEL_PATH}")
    return agent


def evaluate(model_path: str | None = None,
             algorithm: str = "ddqn",
             episodes: int = 5,
             render: bool = True,
             fps: int = 60):
    """评估DDQN agent（添加DDQN的算法分支）"""
    model_dir = MODEL_DIR
    if model_path is None:
        candidates = [f for f in os.listdir(model_dir) if f.endswith(".torch")]
        if not candidates:
            raise FileNotFoundError(f"No saved model found in '{model_dir}/'. Please train first.")
        model_path = os.path.join(model_dir, candidates[0])
        print(f"[Eval] Using detected model: {model_path}")
    else:
        print(f"[Eval] Using provided model: {model_path}")

    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # -----------------------------
    # 添加DDQN的agent实例化分支
    # -----------------------------
    if algorithm.lower() == "dqn":
        from agents.cartpole_dqn import DQNSolver, DQNConfig
        agent = DQNSolver(obs_dim, act_dim, cfg=DQNConfig())
    elif algorithm.lower() == "ddqn":
        agent = DDQNSolver(obs_dim, act_dim, cfg=DDQNConfig())
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    agent.load(model_path)
    print(f"[Eval] Loaded {algorithm.upper()} model from: {model_path}")

    scores = []
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0

        while not done:
            action = agent.act(state, evaluation_mode=True)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1

            if dt > 0:
                time.sleep(dt)

        scores.append(steps)
        print(f"[Eval] Episode {ep}: steps={steps}")

    env.close()
    avg = float(np.mean(scores)) if scores else 0.0
    print(f"[Eval] Average over {episodes} episodes: {avg:.2f}")
    return scores


if __name__ == "__main__":
    # 训练DDQN（500轮），然后评估100轮
   # agent = train(num_episodes=2000, terminal_penalty=False)
    evaluate(model_path="models/cartpole_ddqn.torch", algorithm="ddqn", episodes=100, render=False, fps=60)