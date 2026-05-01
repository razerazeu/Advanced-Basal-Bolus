from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.replay_buffer import ReplayBuffer


def mlp(sizes: list[int], activation: type[nn.Module], output_activation: type[nn.Module] | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, action_low: float, action_high: float) -> None:
        super().__init__()
        self.net = mlp([state_dim, hidden_dim, hidden_dim], nn.ReLU, nn.ReLU)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        scale = (action_high - action_low) / 2.0
        bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.tensor([scale], dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor([bias], dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(state)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()

        if deterministic:
            z = mean
        else:
            noise = torch.randn_like(mean)
            z = mean + std * noise

        squashed = torch.tanh(z)
        action = squashed * self.action_scale + self.action_bias

        normal_log_prob = -0.5 * (((z - mean) / (std + 1e-8)) ** 2 + 2.0 * log_std + np.log(2.0 * np.pi))
        normal_log_prob = normal_log_prob.sum(dim=-1, keepdim=True)

        correction = torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        scale_correction = torch.log(self.action_scale + 1e-8).sum().view(1, 1)
        log_prob = normal_log_prob - correction - scale_correction

        if deterministic:
            log_prob = torch.zeros_like(log_prob)
        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = mlp([state_dim + action_dim, hidden_dim, hidden_dim, 1], nn.ReLU, None)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


@dataclass
class SACConfig:
    state_dim: int
    action_dim: int = 1
    action_low: float = 0.0
    action_high: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_dim: int = 256
    buffer_size: int = 100_000
    batch_size: int = 256
    reward_clip: float = 20.0


class SACAgent:
    def __init__(self, config: SACConfig, device: str = "cpu") -> None:
        self.cfg = config
        self.device = torch.device(device)

        self.actor = GaussianPolicy(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            action_low=config.action_low,
            action_high=config.action_high,
        ).to(self.device)

        self.critic1 = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic2 = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_critic1 = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_critic2 = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=config.lr)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=config.lr)

        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.target_entropy = -float(config.action_dim)

        self.replay_buffer = ReplayBuffer(config.state_dim, config.action_dim, capacity=config.buffer_size)
        self.rng = np.random.default_rng()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> float:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(state_t, deterministic=deterministic)
        return float(action.cpu().numpy().reshape(-1)[0])

    def initialize_actor_output_bias(self, action_value: float, log_std_bias: float | None = None) -> None:
        action_value = float(np.clip(action_value, self.cfg.action_low, self.cfg.action_high))
        scale = (self.cfg.action_high - self.cfg.action_low) / 2.0
        bias = (self.cfg.action_high + self.cfg.action_low) / 2.0
        normalized = np.clip((action_value - bias) / scale, -0.999, 0.999)
        latent_bias = float(np.arctanh(normalized))
        with torch.no_grad():
            self.actor.mean.bias.fill_(latent_bias)
            if log_std_bias is not None:
                self.actor.log_std.bias.fill_(float(log_std_bias))

    def store(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray, done: bool) -> None:
        clipped_reward = float(np.clip(reward, -self.cfg.reward_clip, self.cfg.reward_clip))
        self.replay_buffer.store(
            state.astype(np.float32),
            np.asarray([action], dtype=np.float32),
            clipped_reward,
            next_state.astype(np.float32),
            bool(done),
        )

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for p_src, p_tgt in zip(source.parameters(), target.parameters()):
            p_tgt.data.copy_(self.cfg.tau * p_src.data + (1.0 - self.cfg.tau) * p_tgt.data)

    def update(self) -> dict[str, float]:
        batch_size = self.cfg.batch_size
        assert len(self.replay_buffer) >= batch_size, "Buffer not full yet — do not update before this"

        batch = self.replay_buffer.sample(batch_size=batch_size, rng=self.rng)
        state = torch.tensor(batch["state"], dtype=torch.float32, device=self.device)
        action = torch.tensor(batch["action"], dtype=torch.float32, device=self.device)
        reward = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device)
        next_state = torch.tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        done = torch.tensor(batch["done"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_state, deterministic=False)
            q1_next = self.target_critic1(next_state, next_action)
            q2_next = self.target_critic2(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_logp
            q_target = reward + (1.0 - done) * self.cfg.gamma * q_next

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)

        self.critic1_optim.zero_grad(set_to_none=True)
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self.critic2_optim.step()

        new_action, logp = self.actor.sample(state, deterministic=False)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * logp - q_new).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()

        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        critic_loss = float(0.5 * (critic1_loss.item() + critic2_loss.item()))
        return {
            "critic_loss": critic_loss,
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().item()),
        }

    def freeze_actor(self) -> None:
        for p in self.actor.parameters():
            p.requires_grad_(False)

    def unfreeze_actor(self) -> None:
        for p in self.actor.parameters():
            p.requires_grad_(True)

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.cfg.__dict__,
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic1_optim": self.critic1_optim.state_dict(),
            "critic2_optim": self.critic2_optim.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optim": self.alpha_optim.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    def load(self, path: str | Path, strict: bool = True) -> dict[str, Any]:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"], strict=strict)
        self.critic1.load_state_dict(payload["critic1"], strict=strict)
        self.critic2.load_state_dict(payload["critic2"], strict=strict)
        self.target_critic1.load_state_dict(payload["target_critic1"], strict=strict)
        self.target_critic2.load_state_dict(payload["target_critic2"], strict=strict)
        self.actor_optim.load_state_dict(payload["actor_optim"])
        self.critic1_optim.load_state_dict(payload["critic1_optim"])
        self.critic2_optim.load_state_dict(payload["critic2_optim"])

        self.log_alpha = payload["log_alpha"].to(self.device).requires_grad_(True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)
        self.alpha_optim.load_state_dict(payload["alpha_optim"])
        return payload.get("extra", {})


class CentralizedCriticTrainer:
    """Training-only centralized critic over joint state-action."""

    def __init__(self, state_dim: int, action_dim: int = 2, hidden_dim: int = 128, lr: float = 3e-4, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.net = mlp([state_dim + action_dim, hidden_dim, hidden_dim, 1], nn.ReLU, None).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

    def update(self, state: np.ndarray, joint_action: np.ndarray, reward_value: float) -> float:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.tensor(joint_action, dtype=torch.float32, device=self.device).unsqueeze(0)
        r = torch.tensor([[reward_value]], dtype=torch.float32, device=self.device)

        pred = self.net(torch.cat([s, a], dim=-1))
        loss = F.mse_loss(pred, r)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()
        return float(loss.item())

    def value(self, state: np.ndarray, joint_action: np.ndarray) -> float:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = torch.tensor(joint_action, dtype=torch.float32, device=self.device).unsqueeze(0)
            val = self.net(torch.cat([s, a], dim=-1))
        return float(val.item())
