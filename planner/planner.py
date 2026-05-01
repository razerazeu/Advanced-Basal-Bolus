from __future__ import annotations

import numpy as np


def predict_glucose(
    G: float,
    basal: float,
    bolus: float,
    meal_carbs: float,
    steps: int = 6,
    insulin_sensitivity_scale: float = 1.0,
) -> list[float]:
    """Simple Hovorka-inspired linear approximation for short-horizon scoring."""
    trajectory: list[float] = []
    G_pred = float(G)
    bolus_profile = [0.15, 0.25, 0.25, 0.20, 0.10, 0.05]
    for i in range(steps):
        basal_units = basal * 5.0
        bolus_units = bolus * (bolus_profile[i] if i < len(bolus_profile) else 0.0)
        insulin_effect = -(basal_units + bolus_units) * 20.0 * float(insulin_sensitivity_scale)
        meal_effect = meal_carbs * 0.45 * max(0.0, 1.0 - abs(i - 2) / 3.0)
        dG = insulin_effect + meal_effect
        G_pred = G_pred + dG
        trajectory.append(float(G_pred))
    return trajectory


def risk(G: float) -> float:
    if G < 70.0:
        return (G - 70.0) ** 2
    if G > 180.0:
        return (G - 180.0) ** 2
    return 0.0


def score_candidate(
    basal: float,
    bolus: float,
    current_G: float,
    meal_carbs: float,
    H: int = 6,
    insulin_sensitivity_scale: float = 1.0,
) -> float:
    trajectory = predict_glucose(
        current_G,
        basal,
        bolus,
        meal_carbs,
        steps=H,
        insulin_sensitivity_scale=insulin_sensitivity_scale,
    )
    return -float(sum(risk(g) for g in trajectory))


def debug_candidate_scores(scores: list[float]) -> None:
    print(
        "Planner candidate scores | "
        f"conservative: {scores[0]:.6f} | nominal: {scores[1]:.6f} | aggressive: {scores[2]:.6f}"
    )


class FrozenSafetyPlanner:
    """Frozen planner: no optimizer, no gradients, action filtering only."""

    def __init__(self, horizon: int = 6, override_bolus_eps: float = 0.01, debug: bool = False) -> None:
        self.horizon = horizon
        self.override_bolus_eps = override_bolus_eps
        self.debug = debug

    def select_action(
        self,
        proposed_basal: float,
        proposed_bolus: float,
        current_cgm: float,
        meal_carbs: float,
        insulin_sensitivity_scale: float = 1.0,
        cgm_trend: float = 0.0,
    ) -> tuple[float, float, int, bool, list[float]]:
        basal = float(np.clip(proposed_basal, 0.0, 0.05))
        bolus = float(np.clip(proposed_bolus, 0.0, 0.5))

        candidates = [
            (basal, float(np.clip(bolus * 0.5, 0.0, 0.5))),
            (basal, float(np.clip(bolus, 0.0, 0.5))),
            (basal, float(np.clip(bolus * 1.5, 0.0, 0.5))),
        ]

        scores = [
            score_candidate(
                b,
                bo,
                current_cgm,
                meal_carbs,
                H=self.horizon,
                insulin_sensitivity_scale=insulin_sensitivity_scale,
            )
            for b, bo in candidates
        ]
        if self.debug:
            debug_candidate_scores(scores)
        if np.allclose(np.asarray(scores), scores[1], rtol=1e-8, atol=1e-8):
            if self.debug:
                print("Planner candidate scores identical; check predict_glucose/action sensitivity")
            best_idx = 1
        else:
            best_idx = int(np.argmax(np.asarray(scores)))
        final_basal, final_bolus = candidates[best_idx]

        override = best_idx != 1 and (
            not np.isclose(final_basal, basal, rtol=1e-8, atol=1e-8)
            or abs(final_bolus - bolus) > self.override_bolus_eps
        )
        if current_cgm < 70.0:
            if final_bolus > 0.0:
                override = True
            final_bolus = 0.0
        elif current_cgm < 90.0:
            if final_bolus > 0.0:
                override = True
            final_bolus = 0.0
        elif current_cgm < 110.0 and cgm_trend < -2.0:
            if final_bolus > 0.0:
                override = True
            final_bolus = 0.0
        elif meal_carbs <= 0.0 and current_cgm < 120.0 and insulin_sensitivity_scale > 1.25:
            if final_bolus > 0.0:
                override = True
            final_bolus = 0.0
        elif meal_carbs > 0.0 and current_cgm < 110.0 and insulin_sensitivity_scale > 1.25:
            capped_bolus = min(final_bolus, bolus * 0.5)
            if capped_bolus < final_bolus:
                override = True
            final_bolus = capped_bolus

        trajectory = predict_glucose(
            current_cgm,
            final_basal,
            final_bolus,
            meal_carbs,
            steps=self.horizon,
            insulin_sensitivity_scale=insulin_sensitivity_scale,
        )
        return final_basal, final_bolus, best_idx, override, trajectory
