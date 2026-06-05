"""
PyTorch Double DQN (DDQN) for CartPole (Gymnasium)
------------------------------------
- 核心改进：拆分“动作选择”和“Q值评估”（用online网络选动作，target网络算Q值）
- 其余逻辑与原DQN一致，仅修改experience_replay中的目标计算部分
"""

from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Default Hyperparameters (与原DQN一致)
# -----------------------------
GAMMA = 0.99
LR = 6e-4
BATCH_SIZE = 64
MEMORY_SIZE = 50_000
INITIAL_EXPLORATION_STEPS = 500
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 0.996
TARGET_UPDATE_STEPS = 150


class QNet(nn.Module):
    """与原DQN的QNet完全一致"""
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, act_dim), 
        )
        
        # 可选：原更优的网络结构（若需要可取消注释）
        # self.net = nn.Sequential(
        #     nn.Linear(obs_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, act_dim),
        # )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """与原DQN的ReplayBuffer完全一致"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        s = np.asarray(s)
        s2 = np.asarray(s2)
        if s.ndim == 2 and s.shape[0] == 1:
            s = s.squeeze(0)
        if s2.ndim == 2 and s2.shape[0] == 1:
            s2 = s2.squeeze(0)
        self.buf.append((s, a, r, s2, 0.0 if done else 1.0))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, m = zip(*batch)
        return (
            np.stack(s, axis=0),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2, axis=0),
            np.array(m, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


@dataclass
class DDQNConfig:
    """将原DQNConfig重命名为DDQNConfig（参数不变）"""
    gamma: float = GAMMA
    lr: float = LR
    batch_size: int = BATCH_SIZE
    memory_size: int = MEMORY_SIZE
    initial_exploration: int = INITIAL_EXPLORATION_STEPS
    eps_start: float = EPS_START
    eps_end: float = EPS_END
    eps_decay: float = EPS_DECAY
    target_update: int = TARGET_UPDATE_STEPS
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DDQNSolver:
    """将原DQNSolver重命名为DDQNSolver，核心修改experience_replay方法"""
    def __init__(self, observation_space: int, action_space: int, cfg: DDQNConfig | None = None):
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or DDQNConfig()

        self.device = torch.device(self.cfg.device)

        # 网络、优化器、缓冲区初始化与原DQN一致
        self.online = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.target = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.update_target(hard=True)

        self.optim = optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.memory = ReplayBuffer(self.cfg.memory_size)

        self.steps = 0
        self.exploration_rate = self.cfg.eps_start

    # -----------------------------
    # Acting & memory（与原DQN完全一致）
    # -----------------------------
    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        if not evaluation_mode and np.random.rand() < self.exploration_rate:
            return random.randrange(self.act_dim)

        with torch.no_grad():
            s_np = np.asarray(state_np, dtype=np.float32)
            if s_np.ndim == 1:
                s_np = s_np[None, :]
            s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
            q = self.online(s)
            a = int(torch.argmax(q, dim=1).item())
        return a

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.push(state, action, reward, next_state, done)

    # -----------------------------
    # Learning from replay（核心修改部分）
    # -----------------------------
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.remember(state, action, reward, next_state, done)
        self.experience_replay()

    def experience_replay(self):
        # 1) 热身与容量检查（与原DQN一致）
        if len(self.memory) < max(self.cfg.batch_size, self.cfg.initial_exploration):
            self._decay_eps()
            return

        # 2) 采样并转换为张量（与原DQN一致）
        s, a, r, s2, m = self.memory.sample(self.cfg.batch_size)

        s_t  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        a_t  = torch.as_tensor(a,  dtype=torch.int64,   device=self.device).unsqueeze(1)
        r_t  = torch.as_tensor(r,  dtype=torch.float32, device=self.device).unsqueeze(1)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        m_t  = torch.as_tensor(m,  dtype=torch.float32, device=self.device).unsqueeze(1)

        # 3) 计算当前Q(s,a)（与原DQN一致）
        q_sa = self.online(s_t).gather(1, a_t)  # [B, 1]

        # -----------------------------
        # DDQN核心修改：目标Q值计算
        # -----------------------------
        with torch.no_grad():
            # 用online网络选择下一状态的最优动作：a* = argmax_a Q_online(s', a)
            next_actions = self.online(s2_t).argmax(dim=1, keepdim=True)  # [B, 1]
            # 用target网络计算该动作的Q值：Q_target(s', a*)
            q_next = self.target(s2_t).gather(1, next_actions)  # [B, 1]
            # 最终目标值 = 即时奖励 + γ * Q_target(s', a*) * 存活掩码
            target = r_t + m_t * self.cfg.gamma * q_next  # [B, 1]

        # 4) 损失计算与反向传播（与原DQN一致）
        loss = nn.functional.mse_loss(q_sa, target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # 5) 探索率衰减与目标网络更新（与原DQN一致）
        self._decay_eps()

        if self.steps % self.cfg.target_update == 0:
            self.update_target(hard=True)

    # -----------------------------
    # 其余方法（与原DQN完全一致）
    # -----------------------------
    def update_target(self, hard: bool = True, tau: float = 0.005):
        if hard:
            self.target.load_state_dict(self.online.state_dict())
        else:
            with torch.no_grad():
                for p_t, p in zip(self.target.parameters(), self.online.parameters()):
                    p_t.data.mul_(1 - tau).add_(tau * p.data)

    def save(self, path: str):
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])

    def _decay_eps(self):
        self.exploration_rate = max(self.cfg.eps_end, self.exploration_rate * self.cfg.eps_decay)
        self.steps += 1

if __name__ == "__main__":
    # 测试类是否能正常定义
        cfg = DDQNConfig()
        agent = DDQNSolver(4, 2, cfg=cfg)  # CartPole的obs_dim=4, act_dim=2
        print("DDQNSolver类定义成功！")