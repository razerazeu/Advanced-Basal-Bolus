"""Replay-buffer utilities for off-policy RL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReplayBatch:
    """A sampled mini-batch from the replay buffer."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Simple NumPy replay buffer backed by fixed-size arrays."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: float | np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition."""

        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)

        self.states[self.ptr] = state.astype(np.float32)
        self.actions[self.ptr] = action_arr
        self.rewards[self.ptr] = np.float32(reward)
        self.next_states[self.ptr] = next_state.astype(np.float32)
        self.dones[self.ptr] = np.float32(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        """Sample transitions uniformly."""

        idx = np.random.randint(0, self.size, size=batch_size)
        return ReplayBatch(
            states=torch.as_tensor(self.states[idx], device=self.device),
            actions=torch.as_tensor(self.actions[idx], device=self.device),
            rewards=torch.as_tensor(self.rewards[idx], device=self.device),
            next_states=torch.as_tensor(self.next_states[idx], device=self.device),
            dones=torch.as_tensor(self.dones[idx], device=self.device),
        )

    def __len__(self) -> int:
        return self.size
