from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        for x in flat:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2

    @property
    def var(self) -> float:
        if self.count < 2:
            return 1.0
        return max(self.m2 / (self.count - 1), 1e-6)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return (arr - self.mean) / self.std

    def state_dict(self) -> dict[str, float]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.count = int(state.get("count", 0))
        self.mean = float(state.get("mean", 0.0))
        self.m2 = float(state.get("m2", 0.0))


class StateNormalizer:
    """Normalizes CGM and insulin channels; keeps meal indicators as binary values."""

    def __init__(self) -> None:
        self.cgm = RunningStats()
        self.insulin = RunningStats()

    def build_state(
        self,
        cgm_hist: np.ndarray,
        insulin_hist: np.ndarray,
        meal_hist: np.ndarray,
        patient_features: np.ndarray | None = None,
        update: bool = False,
    ) -> np.ndarray:
        cgm_hist = np.asarray(cgm_hist, dtype=np.float32)
        insulin_hist = np.asarray(insulin_hist, dtype=np.float32)
        meal_hist = np.asarray(meal_hist, dtype=np.float32)
        patient_features = (
            np.asarray(patient_features, dtype=np.float32).reshape(-1)
            if patient_features is not None
            else np.asarray([], dtype=np.float32)
        )

        if update:
            self.cgm.update(cgm_hist)
            self.insulin.update(insulin_hist)

        cgm_n = self.cgm.normalize(cgm_hist)
        insulin_n = self.insulin.normalize(insulin_hist)
        meal_raw = meal_hist

        return np.concatenate([cgm_n, insulin_n, meal_raw, patient_features], axis=0).astype(np.float32)

    def state_dict(self) -> dict[str, dict[str, float]]:
        return {"cgm": self.cgm.state_dict(), "insulin": self.insulin.state_dict()}

    def load_state_dict(self, state: dict[str, dict[str, float]]) -> None:
        self.cgm.load_state_dict(state.get("cgm", {}))
        self.insulin.load_state_dict(state.get("insulin", {}))
