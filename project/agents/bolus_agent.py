"""Fast-timescale bolus SAC agent."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from agents.sac_core import SACCore, SACUpdateStats
from config import BolusAgentConfig, SACConfig
from utils.buffers import ReplayBuffer


class BolusAgent:
    """Bolus controller acting on finer timescale and meal windows."""

    def __init__(self, state_dim: int, sac_cfg: SACConfig, cfg: BolusAgentConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.sac_cfg = sac_cfg
        self.sac = SACCore(state_dim=state_dim, action_dim=1, cfg=sac_cfg, device=device)
        self.replay = ReplayBuffer(
            state_dim=state_dim,
            action_dim=1,
            capacity=sac_cfg.replay_size,
            device=self.sac.device,
        )

    def should_decide(self, sim_time: datetime, meal_grams: float, current_glucose: float) -> bool:
        """Fast decision gate: periodic interval or explicit meal signal."""

        minute_of_day = (sim_time.hour * 60) + sim_time.minute
        interval_gate = minute_of_day % self.cfg.decision_interval_minutes == 0
        meal_gate = meal_grams >= self.cfg.meal_threshold_grams
        correction_gate = current_glucose >= self.cfg.correction_glucose_threshold
        return meal_gate or (interval_gate and correction_gate)

    def act(
        self,
        state: np.ndarray,
        sim_time: datetime,
        meal_grams: float,
        current_glucose: float,
        training: bool,
        total_env_steps: int,
    ) -> tuple[float, float, bool]:
        """Return normalized action, mapped bolus value, and decision-flag."""

        if not self.should_decide(sim_time, meal_grams, current_glucose):
            return -1.0, 0.0, False

        if training and total_env_steps < self.sac_cfg.warmup_steps:
            action = np.random.uniform(-1.0, -0.2, size=(1,)).astype(np.float32)
        else:
            action = self.sac.select_action(state, deterministic=not training)

        bolus = self._map_action_to_bolus(action[0])
        if meal_grams >= self.cfg.meal_threshold_grams:
            meal_total_units = meal_grams / max(self.cfg.meal_bolus_ratio_grams_per_unit, 1e-6)
            meal_cap = meal_total_units / max(self.cfg.delivery_window_minutes, 1e-6)
            bolus = min(bolus, meal_cap)
            if current_glucose <= self.cfg.low_glucose_bolus_threshold:
                bolus = min(bolus, meal_cap * self.cfg.low_glucose_bolus_scale)
        else:
            bolus = min(bolus, self.cfg.max_bolus_no_meal)
        return float(action[0]), float(bolus), True

    def remember(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay.add(state=state, action=np.array([action], dtype=np.float32), reward=reward, next_state=next_state, done=done)

    def update(self) -> SACUpdateStats | None:
        return self.sac.update(self.replay)

    def save(self, path: str) -> None:
        self.sac.save(path)

    def load(self, path: str) -> None:
        self.sac.load(path)

    def _map_action_to_bolus(self, normalized_action: float) -> float:
        """Map [-1, 1] to [min_bolus, max_bolus] with exponential preference."""

        a = float(np.clip(normalized_action, -1.0, 1.0))
        raw_min = self.cfg.i_max * np.exp(-2.0 * self.cfg.eta)
        raw_max = self.cfg.i_max
        raw = self.cfg.i_max * np.exp(self.cfg.eta * (a - 1.0))

        denom = max(raw_max - raw_min, 1e-6)
        normalized = (raw - raw_min) / denom
        bolus = self.cfg.min_bolus + normalized * (self.cfg.max_bolus - self.cfg.min_bolus)
        return float(np.clip(bolus, self.cfg.min_bolus, self.cfg.max_bolus))

    def normalized_from_bolus(self, bolus_value: float) -> float:
        """Approximate inverse map from executed bolus value to normalized action."""

        bolus = float(np.clip(bolus_value, self.cfg.min_bolus, self.cfg.max_bolus))
        if bolus <= self.cfg.min_bolus + 1e-8:
            return -1.0

        span = max(self.cfg.max_bolus - self.cfg.min_bolus, 1e-6)
        normalized = (bolus - self.cfg.min_bolus) / span

        raw_min = self.cfg.i_max * np.exp(-2.0 * self.cfg.eta)
        raw = raw_min + normalized * max(self.cfg.i_max - raw_min, 1e-6)

        eta = max(self.cfg.eta, 1e-6)
        i_max = max(self.cfg.i_max, 1e-6)
        action = 1.0 + (np.log(max(raw, 1e-8) / i_max) / eta)
        return float(np.clip(action, -1.0, 1.0))
