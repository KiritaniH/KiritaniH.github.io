from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95                
    lr: float = 3e-4
    rollout_len: int = 2048          
    update_epochs: int = 10           
    minibatch_size: int = 64
    max_grad_norm: float = 0.5

    clip_eps: float = 0.2

    vf_coef: float = 0.5             
    ent_coef: float = 0.01           

    hidden_sizes: Tuple[int, int] = (128, 128)

    adv_norm: bool = True 
    device: str = "cpu" 


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, int]):
        super().__init__()
        h1, h2 = hidden_sizes
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
        )
        self.actor = nn.Linear(h2, act_dim) 
        self.critic = nn.Linear(h2, 1) 

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.shared(x)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value


class PPOAgent:
    """
    PPO-Clip (on-policy) for discrete action space (e.g., CartPole-v1)
    API required by the project:
      - __init__(obs_dim, act_dim, cfg=None)
      - act(state, evaluation_mode=False)
      - step(state, action, reward, next_state, done)
      - save(path), load(path)
      - PPOConfig
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: Optional[PPOConfig] = None):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cfg = cfg or PPOConfig()

        # device
        self.device = torch.device(self.cfg.device)
        self.net = ActorCritic(self.obs_dim, self.act_dim, self.cfg.hidden_sizes).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        # rollout buffer
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._dones: List[bool] = []
        self._logps: List[float] = []
        self._values: List[float] = []

        self._last_next_state: Optional[np.ndarray] = None
        self._last_done: bool = False

    @torch.no_grad()
    def act(self, state: np.ndarray, evaluation_mode: bool = False) -> int:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, obs_dim]
        logits, _ = self.net(s)
        if evaluation_mode:
            action = int(torch.argmax(logits, dim=-1).item())
            return action
        dist = Categorical(logits=logits)
        action = int(dist.sample().item())
        return action

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.net(s)
            dist = Categorical(logits=logits)
            a = torch.as_tensor([action], dtype=torch.int64, device=self.device)
            logp = dist.log_prob(a).item()
            v = float(value.item())

        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._logps.append(float(logp))
        self._values.append(v)

        self._last_next_state = np.asarray(next_state, dtype=np.float32)
        self._last_done = bool(done)

        if len(self._states) >= self.cfg.rollout_len or done:
            self._update()
            self._clear_buffer()

    def save(self, path: str) -> None:
        payload = {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "cfg": self.cfg.__dict__,
            "state_dict": self.net.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.obs_dim = int(payload["obs_dim"])
        self.act_dim = int(payload["act_dim"])

        cfg_dict = payload.get("cfg", {})
        self.cfg = PPOConfig(**cfg_dict)
        self.device = torch.device(self.cfg.device)

        self.net = ActorCritic(self.obs_dim, self.act_dim, self.cfg.hidden_sizes).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        self.net.load_state_dict(payload["state_dict"])
        if "opt_state_dict" in payload:
            self.opt.load_state_dict(payload["opt_state_dict"])


    def _clear_buffer(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()
        self._logps.clear()
        self._values.clear()
        self._last_next_state = None
        self._last_done = False

    def _compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T = rewards.shape[0]
        adv = np.zeros(T, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - float(dones[t])
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * next_nonterminal * gae
            adv[t] = gae

        returns = adv + values
        return adv, returns

    def _update(self) -> None:
        if len(self._states) == 0:
            return

        if (self._last_next_state is None) or self._last_done:
            last_value = 0.0
        else:
            with torch.no_grad():
                ns = torch.as_tensor(self._last_next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                _, v = self.net(ns)
                last_value = float(v.item())

        states = np.stack(self._states, axis=0)          # [T, obs]
        actions = np.asarray(self._actions, dtype=np.int64)  # [T]
        rewards = np.asarray(self._rewards, dtype=np.float32)  # [T]
        dones = np.asarray(self._dones, dtype=np.bool_)  # [T]
        old_logps = np.asarray(self._logps, dtype=np.float32)  # [T]
        values = np.asarray(self._values, dtype=np.float32)    # [T]

        adv, returns = self._compute_gae(rewards, dones, values, last_value)

        if self.cfg.adv_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_logp_t = torch.as_tensor(old_logps, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        T = s_t.shape[0]
        batch_size = T
        mb = min(self.cfg.minibatch_size, batch_size)

        idx = np.arange(batch_size)
        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_size, mb):
                end = start + mb
                mb_idx = torch.as_tensor(idx[start:end], dtype=torch.int64, device=self.device)

                logits, vpred = self.net(s_t[mb_idx])
                dist = Categorical(logits=logits)

                new_logp = dist.log_prob(a_t[mb_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp_t[mb_idx])

                surr1 = ratio * adv_t[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv_t[mb_idx]
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                value_loss = F.mse_loss(vpred, ret_t[mb_idx])

                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
"""
from agents.ppo_agent import PPOAgent, PPOConfig


    elif algorithm == "ppo":
        cfg = PPOConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            rollout_len=1024,
            update_epochs=10,
            minibatch_size=64,
            lr=3e-4,
            clip_eps=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            lam=0.95,
            gamma=0.99,
        )
"""