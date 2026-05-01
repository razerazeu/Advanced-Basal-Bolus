from __future__ import annotations

import math
import numpy as np


BG_TARGET = 125.0
ACTION_EPS = 0.01


def glucose_reward(current_cgm: float) -> float:
    """Jaloli-style glucose reward centered around the target BG value."""
    glucose = float(current_cgm)
    if 70.0 <= glucose <= 180.0:
        return float(0.1 * math.exp(-abs(glucose - BG_TARGET) / 100.0))
    return float(-0.01 * abs(glucose - BG_TARGET))


def action_timing_reward(meal_flag: int, insulin_action: float) -> float:
    """Reward insulin delivery around meal events and penalize mismatched timing."""
    has_meal = bool(meal_flag)
    has_action = float(insulin_action) > ACTION_EPS

    if has_action and has_meal:
        return 10.0
    if (not has_action) and (not has_meal):
        return 0.0
    return -2.0


def basal_reward(cgm_history: np.ndarray, current_cgm: float) -> float:
    glucose_window = np.asarray(cgm_history, dtype=np.float32)[-12:]

    def range_score(low: float, high: float) -> float:
        return float(10.0 * np.mean((glucose_window >= low) & (glucose_window <= high)))

    reward = (
        math.exp(range_score(105.0, 115.0) / 2.0)
        + math.exp(range_score(100.0, 120.0) / 2.0)
        + math.exp(range_score(70.0, 180.0) / 2.0)
    )
    if float(current_cgm) < 70.0:
        reward -= 10.0
    return float(reward)


def bolus_reward(current_cgm: float, meal_flag: int, bolus: float) -> float:
    return float(glucose_reward(current_cgm) + action_timing_reward(meal_flag, bolus))
