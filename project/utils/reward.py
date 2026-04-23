"""Reward definitions for basal, bolus, and planner risk."""

from __future__ import annotations

import math
from collections.abc import Sequence


BOLUS_DELIVERY_MINUTES = 15.0
BASAL_REWARD_WINDOW_HOURS = 7
BASAL_REWARD_SCALE = 10.0


def _range_indicator(bg: float, low: float, high: float) -> float:
    return 1.0 if low <= bg <= high else 0.0


def _nested_range_hit_count(bg: float) -> float:
    """Return nested-range hit count from the basal paper structure."""

    return (
        _range_indicator(bg, 105.0, 115.0)
        + _range_indicator(bg, 100.0, 120.0)
        + _range_indicator(bg, 70.0, 180.0)
    )


def basal_step_reward(bg: float) -> float:
    """Backward-compatible one-point basal reward proxy."""

    return basal_step_reward_from_buffer([bg])


def basal_step_reward_from_buffer(bg_buffer: Sequence[float]) -> float:
    """Algorithm-1 style basal reward from rolling BG buffer."""

    n = max(len(bg_buffer), 1)

    c_105_115 = sum(1 for g in bg_buffer if 105.0 < g < 115.0)
    c_100_120 = sum(1 for g in bg_buffer if 100.0 < g < 120.0)
    c_70_180 = sum(1 for g in bg_buffer if 70.0 < g < 180.0)

    r_105_115 = BASAL_REWARD_SCALE * (c_105_115 / n)
    r_100_120 = BASAL_REWARD_SCALE * (c_100_120 / n)
    r_70_180 = BASAL_REWARD_SCALE * (c_70_180 / n)

    return float(
        math.exp(r_105_115 / 2.0)
        + math.exp(r_100_120 / 2.0)
        + math.exp(r_70_180 / 2.0)
    )


def basal_episode_reward(glucose_trace: Sequence[float]) -> float:
    """Algorithm-1 episode reward as sum of per-step rolling-buffer rewards."""

    if len(glucose_trace) == 0:
        return 0.0

    # Simglucose main path runs at 5-minute steps, so 7h corresponds to 84 samples.
    window_steps = int((BASAL_REWARD_WINDOW_HOURS * 60) / 5)
    window_steps = max(window_steps, 1)

    total_reward = 0.0
    for idx in range(len(glucose_trace)):
        start = max(0, idx - window_steps + 1)
        bg_buffer = glucose_trace[start : idx + 1]
        total_reward += basal_step_reward_from_buffer(bg_buffer)

    return float(total_reward)


def bolus_glucose_reward(bg: float) -> float:
    """Glucose-quality term using BGTarget=125 and [70, 180] control band."""

    if 70.0 <= bg <= 180.0:
        return 0.1 * math.exp(-abs(bg - 125.0) / 100.0)
    return -0.01 * abs(bg - 125.0)


def bolus_action_reward(
    meal_grams: float,
    bolus_units: float,
    *,
    meal_window_active: bool,
    recent_meal_grams: float,
) -> float:
    """Action-timing term: +10 together, 0 if neither, -2 otherwise."""

    del meal_window_active, recent_meal_grams

    meal_occurs = meal_grams > 0.0
    bolus_occurs = bolus_units > 1e-4

    if bolus_occurs and meal_occurs:
        return 10.0
    if (not bolus_occurs) and (not meal_occurs):
        return 0.0
    return -2.0


def bolus_step_reward(
    bg: float,
    meal_grams: float,
    bolus_units: float,
    *,
    meal_window_active: bool | None = None,
    recent_meal_grams: float = 0.0,
) -> float:
    """Combined bolus reward: glucose-quality + action-timing terms."""

    timing_window_active = bool(meal_grams > 0.0) if meal_window_active is None else bool(meal_window_active)
    return bolus_glucose_reward(bg) + bolus_action_reward(
        meal_grams,
        bolus_units,
        meal_window_active=timing_window_active,
        recent_meal_grams=recent_meal_grams,
    )


def bolus_episode_reward(glucose_trace: Sequence[float], meal_trace: Sequence[float], bolus_trace: Sequence[float]) -> float:
    """Episode-level bolus reward sum."""

    return float(
        sum(
            bolus_step_reward(bg, meal, bolus, meal_window_active=meal > 0.0, recent_meal_grams=meal)
            for bg, meal, bolus in zip(glucose_trace, meal_trace, bolus_trace)
        )
    )


def risk_index(bg: float) -> float:
    """Kovatchev-style risk index used by planning safety score."""

    bg_clamped = max(bg, 1.0)
    f_bg = 1.509 * ((math.log(bg_clamped) ** 1.084) - 5.381)
    return 10.0 * (f_bg**2)


def planner_risk_reward(next_bg: float) -> float:
    """G2P2C-inspired planner risk: fixed severe-low penalty else negative RI."""

    if next_bg <= 39.0:
        return -15.0
    return -risk_index(next_bg)
