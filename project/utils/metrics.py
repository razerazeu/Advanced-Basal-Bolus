"""Evaluation metrics for glucose-control episodes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from utils.reward import basal_episode_reward, bolus_episode_reward


@dataclass
class EpisodeMetrics:
    """Accumulates trajectory data and computes summary statistics."""

    glucose: list[float] = field(default_factory=list)
    basal: list[float] = field(default_factory=list)
    bolus: list[float] = field(default_factory=list)
    meal: list[float] = field(default_factory=list)
    reward_basal_steps: list[float] = field(default_factory=list)
    reward_bolus_steps: list[float] = field(default_factory=list)

    def update(
        self,
        glucose: float,
        basal: float,
        bolus: float,
        meal: float,
        reward_basal: float,
        reward_bolus: float,
    ) -> None:
        self.glucose.append(float(glucose))
        self.basal.append(float(basal))
        self.bolus.append(float(bolus))
        self.meal.append(float(meal))
        self.reward_basal_steps.append(float(reward_basal))
        self.reward_bolus_steps.append(float(reward_bolus))

    def summary(self) -> dict[str, float]:
        if not self.glucose:
            return {
                "time_in_range": 0.0,
                "time_below_range": 0.0,
                "time_above_range": 0.0,
                "mean_glucose": 0.0,
                "min_glucose": 0.0,
                "max_glucose": 0.0,
                "avg_basal": 0.0,
                "avg_bolus": 0.0,
                "hypoglycemia_events": 0.0,
                "hyperglycemia_events": 0.0,
                "reward_basal_episode": 0.0,
                "reward_bolus_episode": 0.0,
            }

        g = np.asarray(self.glucose, dtype=np.float32)
        basal = np.asarray(self.basal, dtype=np.float32)
        bolus = np.asarray(self.bolus, dtype=np.float32)

        in_range = np.logical_and(g >= 70.0, g <= 180.0)
        below = g < 70.0
        above = g > 180.0

        return {
            "time_in_range": float(np.mean(in_range)),
            "time_below_range": float(np.mean(below)),
            "time_above_range": float(np.mean(above)),
            "mean_glucose": float(np.mean(g)),
            "min_glucose": float(np.min(g)),
            "max_glucose": float(np.max(g)),
            "avg_basal": float(np.mean(basal)),
            "avg_bolus": float(np.mean(bolus)),
            "hypoglycemia_events": float(_count_events(g, threshold=70.0, below=True)),
            "hyperglycemia_events": float(_count_events(g, threshold=180.0, below=False)),
            "reward_basal_episode": float(basal_episode_reward(self.glucose)),
            "reward_bolus_episode": float(bolus_episode_reward(self.glucose, self.meal, self.bolus)),
            "reward_basal_sum": float(np.sum(self.reward_basal_steps)),
            "reward_bolus_sum": float(np.sum(self.reward_bolus_steps)),
        }


def _count_events(glucose: np.ndarray, threshold: float, below: bool) -> int:
    """Count threshold-crossing events rather than contiguous points."""

    if len(glucose) == 0:
        return 0

    if below:
        condition = glucose < threshold
    else:
        condition = glucose > threshold

    events = 0
    prev = False
    for flag in condition:
        current = bool(flag)
        if current and not prev:
            events += 1
        prev = current
    return events
