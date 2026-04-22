"""Soft Actor-Critic core for continuous insulin-control actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from config import SACConfig
from utils.buffers import ReplayBatch, ReplayBuffer


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor for bounded continuous actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(state)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self(state)
        std = log_std.exp()
        normal = Normal(mu, std)

        z = normal.rsample()
        action = torch.tanh(z)

        # SAC log-prob correction for tanh squashing.
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    """Q-network Q(s, a)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q_net = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.q_net(x)


@dataclass
class SACUpdateStats:
    """Training stats from one SAC update step."""

    actor_loss: float
    critic1_loss: float
    critic2_loss: float
    alpha_loss: float
    alpha: float


class SACCore:
    """Standalone SAC learner with actor, twin critics, and entropy tuning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: SACConfig,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)

        self.actor = GaussianActor(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.critic1 = Critic(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_critic1 = Critic(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim, cfg.hidden_dim).to(self.device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=cfg.critic_lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=cfg.critic_lr)

        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        self.target_entropy = cfg.target_entropy

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action in [-1, 1]."""

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(state_t)
                action = torch.tanh(mu)
            else:
                action, _ = self.actor.sample(state_t)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, replay: ReplayBuffer) -> SACUpdateStats | None:
        """Run one SAC update from replay buffer."""

        if len(replay) < self.cfg.batch_size:
            return None

        batch = replay.sample(self.cfg.batch_size)
        return self._update_from_batch(batch)

    def _update_from_batch(self, batch: ReplayBatch) -> SACUpdateStats:
        states = batch.states
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_states
        dones = batch.dones

        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            q_backup = rewards + (1.0 - dones) * self.cfg.gamma * target_q

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = nn.functional.mse_loss(q1, q_backup)
        critic2_loss = nn.functional.mse_loss(q2, q_backup)

        self.critic1_opt.zero_grad(set_to_none=True)
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self.critic2_opt.step()

        new_actions, log_prob = self.actor.sample(states)
        q1_pi = self.critic1(states, new_actions)
        q2_pi = self.critic2(states, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        return SACUpdateStats(
            actor_loss=float(actor_loss.item()),
            critic1_loss=float(critic1_loss.item()),
            critic2_loss=float(critic2_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            alpha=float(self.alpha.detach().item()),
        )

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        for target_p, online_p in zip(target.parameters(), online.parameters()):
            target_p.data.mul_(1.0 - self.cfg.tau)
            target_p.data.add_(self.cfg.tau * online_p.data)

    def save(self, path: str) -> None:
        """Serialize all trainable SAC state."""

        payload = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic1_opt": self.critic1_opt.state_dict(),
            "critic2_opt": self.critic2_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        """Load trainable SAC state."""

        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic1.load_state_dict(payload["critic1"])
        self.critic2.load_state_dict(payload["critic2"])
        self.target_critic1.load_state_dict(payload["target_critic1"])
        self.target_critic2.load_state_dict(payload["target_critic2"])

        self.log_alpha = payload["log_alpha"].to(self.device).requires_grad_(True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)

        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic1_opt.load_state_dict(payload["critic1_opt"])
        self.critic2_opt.load_state_dict(payload["critic2_opt"])
        self.alpha_opt.load_state_dict(payload["alpha_opt"])
