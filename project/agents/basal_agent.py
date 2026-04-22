"""Slow-timescale basal SAC agent."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from agents.sac_core import SACCore, SACUpdateStats
from config import BasalAgentConfig, SACConfig
from utils.buffers import ReplayBuffer


class BasalAgent:
    """Basal controller acting once per day (around 07:00 by default)."""

    def __init__(self, state_dim: int, sac_cfg: SACConfig, cfg: BasalAgentConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.sac_cfg = sac_cfg
        self.sac = SACCore(state_dim=state_dim, action_dim=1, cfg=sac_cfg, device=device)
        self.replay = ReplayBuffer(
            state_dim=state_dim,
            action_dim=1,
            capacity=sac_cfg.replay_size,
            device=self.sac.device,
        )

        self.current_basal = cfg.default_basal
        self.current_action = np.array([0.0], dtype=np.float32)
        self.last_decision_date = None

    def reset(self) -> None:
        """Reset episode-level scheduling state."""

        self.current_basal = self.cfg.default_basal
        self.current_action = np.array([0.0], dtype=np.float32)
        self.last_decision_date = None

    def should_decide(self, sim_time: datetime) -> bool:
        """Daily decision gate."""

        if self.last_decision_date is None:
            return True
        if self.last_decision_date == sim_time.date():
            return False
        return sim_time.hour == self.cfg.decision_hour and sim_time.minute == self.cfg.decision_minute

    def act(
        self,
        state: np.ndarray,
        sim_time: datetime,
        training: bool,
        total_env_steps: int,
    ) -> tuple[float, float, bool]:
        """Return normalized action, mapped basal value, and decision-flag."""

        if not self.should_decide(sim_time):
            return float(self.current_action[0]), float(self.current_basal), False

        if training and total_env_steps < self.sac_cfg.warmup_steps:
            action = np.random.uniform(-1.0, 0.0, size=(1,)).astype(np.float32)
        else:
            action = self.sac.select_action(state, deterministic=not training)

        basal = self._map_action_to_basal(action[0])
        self.current_action = action
        self.current_basal = basal
        self.last_decision_date = sim_time.date()
        return float(action[0]), float(basal), True

    def remember(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay.add(state=state, action=np.array([action], dtype=np.float32), reward=reward, next_state=next_state, done=done)

    def update(self) -> SACUpdateStats | None:
        return self.sac.update(self.replay)

    def save(self, path: str) -> None:
        self.sac.save(path)

    def load(self, path: str) -> None:
        self.sac.load(path)

    def _map_action_to_basal(self, normalized_action: float) -> float:
        """Map [-1, 1] to [min_basal, max_basal] with exponential preference."""

        a = float(np.clip(normalized_action, -1.0, 1.0))
        raw_min = self.cfg.i_max * np.exp(-2.0 * self.cfg.eta)
        raw_max = self.cfg.i_max
        raw = self.cfg.i_max * np.exp(self.cfg.eta * (a - 1.0))

        denom = max(raw_max - raw_min, 1e-6)
        normalized = (raw - raw_min) / denom
        basal = self.cfg.min_basal + normalized * (self.cfg.max_basal - self.cfg.min_basal)
        return float(np.clip(basal, self.cfg.min_basal, self.cfg.max_basal))

    def normalized_from_basal(self, basal_value: float) -> float:
        """Approximate inverse map from executed basal value back to normalized action."""

        basal = float(np.clip(basal_value, self.cfg.min_basal, self.cfg.max_basal))
        if basal <= self.cfg.min_basal + 1e-8:
            return -1.0

        span = max(self.cfg.max_basal - self.cfg.min_basal, 1e-6)
        normalized = (basal - self.cfg.min_basal) / span

        raw_min = self.cfg.i_max * np.exp(-2.0 * self.cfg.eta)
        raw = raw_min + normalized * max(self.cfg.i_max - raw_min, 1e-6)

        eta = max(self.cfg.eta, 1e-6)
        i_max = max(self.cfg.i_max, 1e-6)
        action = 1.0 + (np.log(max(raw, 1e-8) / i_max) / eta)
        return float(np.clip(action, -1.0, 1.0))
