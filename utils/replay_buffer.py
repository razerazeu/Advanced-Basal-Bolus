from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 100_000) -> None:
        self.capacity = int(capacity)
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_state[idx] = next_state
        self.done[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        if self.size < batch_size:
            raise ValueError(f"Not enough samples: {self.size} < {batch_size}")
        rng = rng if rng is not None else np.random.default_rng()
        idx = rng.choice(self.size, size=batch_size, replace=False)
        return {
            "state": self.state[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_state": self.next_state[idx],
            "done": self.done[idx],
        }
