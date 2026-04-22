"""Sliding-window state representation builder."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from config import StateConfig


@dataclass
class StateSnapshot:
    """Snapshot container for debugging and analyses."""

    cgm: np.ndarray
    insulin: np.ndarray
    meal: np.ndarray | None


class SlidingWindowStateBuilder:
    """Maintains rolling history for CGM, insulin, and meal indicators."""

    def __init__(self, config: StateConfig) -> None:
        self.config = config
        self.cgm_hist: deque[float] = deque(maxlen=config.window_size)
        self.insulin_hist: deque[float] = deque(maxlen=config.window_size)
        self.meal_hist: deque[float] = deque(maxlen=config.window_size)
        self.reset()

    def reset(self, initial_cgm: float = 110.0) -> None:
        """Reset all rolling buffers to stable defaults."""

        self.cgm_hist.clear()
        self.insulin_hist.clear()
        self.meal_hist.clear()

        for _ in range(self.config.window_size):
            self.cgm_hist.append(float(initial_cgm))
            self.insulin_hist.append(0.0)
            self.meal_hist.append(0.0)

    def update(self, cgm: float | None, insulin: float | None, meal: float | None = None) -> None:
        """Append newest values with robust handling for missing inputs."""

        cgm_value = self._safe_value(cgm, fallback=self.cgm_hist[-1] if self.cgm_hist else 110.0)
        insulin_value = self._safe_value(insulin, fallback=0.0)
        meal_value = self._safe_value(meal, fallback=0.0)

        self.cgm_hist.append(cgm_value)
        self.insulin_hist.append(insulin_value)
        self.meal_hist.append(meal_value)

    def build_state(self) -> np.ndarray:
        """Build a normalized fixed-size state vector."""

        cgm = np.asarray(self.cgm_hist, dtype=np.float32)
        insulin = np.asarray(self.insulin_hist, dtype=np.float32)

        cgm_norm = (cgm - self.config.glucose_center) / max(self.config.glucose_scale, 1e-6)
        insulin_norm = insulin / max(self.config.insulin_scale, 1e-6)

        channels = [cgm_norm, insulin_norm]

        if self.config.include_meal:
            meal = np.asarray(self.meal_hist, dtype=np.float32)
            meal_norm = meal / max(self.config.meal_scale, 1e-6)
            channels.append(meal_norm)

        return np.concatenate(channels, axis=0).astype(np.float32)

    def preview_state(self, cgm: float, insulin: float, meal: float | None = None) -> np.ndarray:
        """Compute the next state without mutating internal buffers."""

        cgm_arr = np.asarray(list(self.cgm_hist)[1:] + [self._safe_value(cgm, self.cgm_hist[-1])], dtype=np.float32)
        insulin_arr = np.asarray(list(self.insulin_hist)[1:] + [self._safe_value(insulin, 0.0)], dtype=np.float32)

        cgm_norm = (cgm_arr - self.config.glucose_center) / max(self.config.glucose_scale, 1e-6)
        insulin_norm = insulin_arr / max(self.config.insulin_scale, 1e-6)
        channels = [cgm_norm, insulin_norm]

        if self.config.include_meal:
            meal_arr = np.asarray(list(self.meal_hist)[1:] + [self._safe_value(meal, 0.0)], dtype=np.float32)
            meal_norm = meal_arr / max(self.config.meal_scale, 1e-6)
            channels.append(meal_norm)

        return np.concatenate(channels, axis=0).astype(np.float32)

    def snapshot(self) -> StateSnapshot:
        """Return current unnormalized buffers for diagnostics."""

        meal_data: np.ndarray | None = None
        if self.config.include_meal:
            meal_data = np.asarray(self.meal_hist, dtype=np.float32)

        return StateSnapshot(
            cgm=np.asarray(self.cgm_hist, dtype=np.float32),
            insulin=np.asarray(self.insulin_hist, dtype=np.float32),
            meal=meal_data,
        )

    @staticmethod
    def _safe_value(value: float | None, fallback: float) -> float:
        if value is None:
            return float(fallback)
        if isinstance(value, (float, int)):
            val = float(value)
            if np.isnan(val) or np.isinf(val):
                return float(fallback)
            return val
        return float(fallback)
