"""Reward definitions for basal, bolus, and planner risk."""

from __future__ import annotations

import math
from collections.abc import Sequence


BOLUS_DELIVERY_MINUTES = 15.0


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
    """Basal nested-range reward with exponentiated range hit count."""

    nested_hits = _nested_range_hit_count(bg)
    base_reward = math.exp(nested_hits)

    # Strongly discourage hypoglycemia so policy does not settle at low glucose.
    if bg < 54.0:
        hypo_penalty = 22.0 + 1.1 * (54.0 - bg)
    elif bg < 70.0:
        hypo_penalty = 8.0 + 0.5 * (70.0 - bg)
    else:
        hypo_penalty = 0.0

    # Mild hyperglycemia penalty keeps control balanced around target range.
    if bg > 250.0:
        hyper_penalty = 2.5 + 0.04 * (bg - 250.0)
    elif bg > 180.0:
        hyper_penalty = 0.6 + 0.015 * (bg - 180.0)
    else:
        hyper_penalty = 0.0

    return base_reward - hypo_penalty - hyper_penalty


def basal_episode_reward(glucose_trace: Sequence[float]) -> float:
    """Episode basal reward exactly as a sum of step-level terms."""

    return float(sum(basal_step_reward(g) for g in glucose_trace))


def bolus_glucose_reward(bg: float) -> float:
    """Glucose-dependent bolus reward from the spec."""

    if bg < 54.0:
        return -10.0 - (0.35 * (54.0 - bg))
    if bg < 70.0:
        return -5.0 - (0.15 * (70.0 - bg))
    if bg <= 180.0:
        return 1.2 * math.exp(-abs(bg - 125.0) / 45.0)
    if bg <= 250.0:
        return -0.03 * (bg - 180.0)
    return -2.1 - 0.05 * (bg - 250.0)


def bolus_action_reward(
    meal_grams: float,
    bolus_units: float,
    *,
    meal_window_active: bool,
    recent_meal_grams: float,
) -> float:
    """Meal-timing reward: reward on-window bolus, penalize mistimed bolus."""

    bolus = bolus_units > 1e-4

    if meal_window_active and bolus:
        meal_reference = max(meal_grams, recent_meal_grams)
        target_bolus = max(0.0, meal_reference / 12.0) / max(BOLUS_DELIVERY_MINUTES, 1e-6)
        dose_error = abs(bolus_units - target_bolus)
        return max(-1.5, 1.6 - (0.35 * dose_error))

    if meal_window_active and (not bolus):
        return -1.8

    if (not meal_window_active) and bolus:
        return -2.6 - (0.5 * bolus_units)

    return 0.1


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
    """Risk-sensitive one-step planner reward."""

    if next_bg < 54.0:
        return -220.0 - (2.0 * (54.0 - next_bg))
    if next_bg < 70.0:
        return -95.0 - (1.0 * (70.0 - next_bg))
    if next_bg > 250.0:
        return -30.0 - (0.2 * (next_bg - 250.0))
    return -risk_index(next_bg)
