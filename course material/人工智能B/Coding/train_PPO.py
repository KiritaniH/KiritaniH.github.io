"""
CartPole Training & Evaluation (PPO)
-----------------------------------
- 严格 on-policy PPO
- 无 ε-greedy
- 无 reward shaping
- 保存 best model，防止策略退化
"""

from __future__ import annotations
import os
import time
import random
from collections import deque

import numpy as np
import gymnasium as gym
import torch

from agents.cartpole_ppo import PPOAgent, PPOConfig
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cartpole_ppo.torch")
SEED = 0


def train(num_episodes: int = 1000) -> PPOAgent:
    # =========================
    # 固定随机种子
    # =========================
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)

    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # =========================
    # PPO 初始化
    # =========================
    cfg = PPOConfig()
    agent = PPOAgent(obs_dim, act_dim, cfg=cfg)

    print(f"[PPO-Train] Device: {cfg.device}")

    # ===== 新增：用于 early best model =====
    recent_scores = deque(maxlen=100)
    best_avg = -float("inf")

    # =========================
    # 训练循环
    # =========================
    for ep in range(1, num_episodes + 1):
        state, _ = env.reset(seed=SEED + ep)
        done = False
        steps = 0

        while not done:
            action = agent.act(state, evaluation_mode=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)

            state = next_state
            steps += 1

        logger.add_score(steps, ep)
        recent_scores.append(steps)

        # ===== 保存 best model（核心）=====
        if len(recent_scores) == 100:
            avg_100 = float(np.mean(recent_scores))
            if avg_100 > best_avg:
                best_avg = avg_100
                agent.save(MODEL_PATH)

        if ep % 10 == 0:
            avg_show = np.mean(recent_scores) if recent_scores else steps
            print(f"[Train] Episode {ep:4d} | Score: {steps:3d} | Avg100: {avg_show:.1f}")

    env.close()
    print(f"[PPO-Train] Best model saved to {MODEL_PATH}")
    return agent


def evaluate(episodes: int = 100, render: bool = False, fps: int = 60):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found, please train first.")

    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = PPOAgent(obs_dim, act_dim, cfg=PPOConfig())
    agent.load(MODEL_PATH)

    scores = []
    dt = 1.0 / fps if render else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=SEED + 1000 + ep)
        done = False
        steps = 0

        while not done:
            action = agent.act(state, evaluation_mode=True)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if dt > 0:
                time.sleep(dt)

        scores.append(steps)
        print(f"[Eval] Episode {ep:3d} | Steps: {steps}")

    env.close()
    avg = np.mean(scores)
    print(f"[Eval] Average score over {episodes} episodes: {avg:.2f}")
    return scores


if __name__ == "__main__":
    train(num_episodes=1000)
    evaluate(episodes=100, render=False)
