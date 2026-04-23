"""G2P2C-inspired predictive planning/safety layer.

This module is inference-time only: it evaluates candidate basal/bolus actions
with a short-horizon surrogate model and returns a safer action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import PlannerConfig
from utils.reward import planner_risk_reward


@dataclass
class PlannerResult:
    """Planner output and debugging metadata."""

    basal: float
    bolus: float
    score: float
    trajectory: list[float]
    mode: str
    predicted_min_glucose: float
    intervened: bool
    hard_safety_applied: bool


class G2P2CPlanner:
    """Lightweight predictive action refiner inspired by G2P2C planning."""

    def __init__(self, cfg: PlannerConfig) -> None:
        self.cfg = cfg

    def refine_action(
        self,
        proposed_basal: float,
        proposed_bolus: float,
        current_glucose: float,
        meal_grams: float,
        recent_insulin: float = 0.0,
    ) -> tuple[float, float, PlannerResult]:
        """Evaluate bolus candidates with fixed basal and pick safest high-score action."""

        basal = float(np.clip(proposed_basal, *self.cfg.basal_bounds))
        bolus = float(np.clip(proposed_bolus, *self.cfg.bolus_bounds))

        if not self.cfg.enabled:
            traj = self._rollout(current_glucose, meal_grams, basal, bolus, initial_recent_insulin=recent_insulin)
            score = self._score_trajectory(traj)
            result = PlannerResult(
                basal=basal,
                bolus=bolus,
                score=score,
                trajectory=traj.tolist(),
                mode="disabled",
                predicted_min_glucose=float(np.min(traj)),
                intervened=False,
                hard_safety_applied=False,
            )
            return basal, bolus, result

        candidates = self._generate_candidates(bolus=bolus)

        best_overall_bolus = bolus
        best_overall_score = float("-inf")
        best_overall_traj = self._rollout(
            current_glucose=current_glucose,
            initial_meal_grams=meal_grams,
            basal=basal,
            bolus=bolus,
            initial_recent_insulin=recent_insulin,
        )

        best_safe_bolus: float | None = None
        best_safe_score = float("-inf")
        best_safe_traj: np.ndarray | None = None

        for cand_bolus in candidates:
            trajectory = self._rollout(
                current_glucose=current_glucose,
                initial_meal_grams=meal_grams,
                basal=basal,
                bolus=cand_bolus,
                initial_recent_insulin=recent_insulin,
            )
            score = self._score_trajectory(trajectory)
            predicted_min = float(np.min(trajectory))

            if score > best_overall_score:
                best_overall_score = score
                best_overall_bolus = cand_bolus
                best_overall_traj = trajectory

            if predicted_min >= self.cfg.severe_hypoglycemia_threshold and score > best_safe_score:
                best_safe_score = score
                best_safe_bolus = cand_bolus
                best_safe_traj = trajectory

        hard_safety_applied = False
        if best_safe_bolus is not None and best_safe_traj is not None:
            selected_bolus = best_safe_bolus
            selected_traj = best_safe_traj
            selected_score = best_safe_score
        else:
            selected_bolus = best_overall_bolus
            selected_traj = best_overall_traj
            selected_score = best_overall_score
            hard_safety_applied = True

        if self.cfg.mode == "soft":
            final_bolus = bolus + self.cfg.soft_adjustment_gain * (selected_bolus - bolus)
            final_bolus = float(np.clip(final_bolus, *self.cfg.bolus_bounds))
        else:
            final_bolus = float(selected_bolus)

        final_basal = basal

        if current_glucose <= self.cfg.severe_hypoglycemia_threshold:
            final_bolus = float(self.cfg.bolus_bounds[0])
            hard_safety_applied = True

        if (
            current_glucose <= self.cfg.low_glucose_threshold
            and recent_insulin >= self.cfg.iob_safety_threshold
        ):
            final_bolus = float(self.cfg.bolus_bounds[0])
            hard_safety_applied = True

        final_trajectory = self._rollout(
            current_glucose=current_glucose,
            initial_meal_grams=meal_grams,
            basal=final_basal,
            bolus=final_bolus,
            initial_recent_insulin=recent_insulin,
        )
        predicted_min = float(np.min(final_trajectory))

        if predicted_min < self.cfg.severe_hypoglycemia_threshold:
            final_bolus = float(self.cfg.bolus_bounds[0])
            hard_safety_applied = True
            final_trajectory = self._rollout(
                current_glucose=current_glucose,
                initial_meal_grams=meal_grams,
                basal=final_basal,
                bolus=final_bolus,
                initial_recent_insulin=recent_insulin,
            )
            predicted_min = float(np.min(final_trajectory))

        final_score = self._score_trajectory(final_trajectory)
        intervened = (abs(final_basal - proposed_basal) > 1e-9) or (abs(final_bolus - proposed_bolus) > 1e-9)

        result = PlannerResult(
            basal=float(final_basal),
            bolus=float(final_bolus),
            score=float(final_score),
            trajectory=final_trajectory.tolist(),
            mode=self.cfg.mode,
            predicted_min_glucose=float(predicted_min),
            intervened=bool(intervened),
            hard_safety_applied=bool(hard_safety_applied),
        )
        return float(final_basal), float(final_bolus), result

    def _generate_candidates(
        self,
        bolus: float,
    ) -> list[float]:
        """Build bolus candidates around SAC proposal while keeping basal fixed."""

        candidates: set[float] = {
            float(np.clip(bolus, *self.cfg.bolus_bounds)),
            float(self.cfg.bolus_bounds[0]),
        }

        for scale in self.cfg.candidate_scales:
            candidates.add(float(np.clip(bolus * scale, *self.cfg.bolus_bounds)))

        for offset in self.cfg.bolus_offsets:
            candidates.add(float(np.clip(bolus + offset, *self.cfg.bolus_bounds)))

        return sorted(candidates)

    def _rollout(
        self,
        current_glucose: float,
        initial_meal_grams: float,
        basal: float,
        bolus: float,
        initial_recent_insulin: float = 0.0,
    ) -> np.ndarray:
        """Short-horizon surrogate glucose simulation."""

        glucose = float(max(current_glucose, 20.0))
        meal_pool = float(max(initial_meal_grams, 0.0))
        iob = float(max(initial_recent_insulin, 0.0))
        trajectory = []

        for step in range(self.cfg.horizon_steps):
            absorbed = meal_pool * self.cfg.carb_absorption_rate
            meal_pool = max(0.0, meal_pool - absorbed)

            # Decay bolus impact after the first planning step.
            bolus_effective = bolus if step == 0 else 0.45 * bolus

            insulin_total = basal + bolus_effective + (self.cfg.iob_weight * iob)
            insulin_drop = self.cfg.insulin_sensitivity * insulin_total
            meal_rise = self.cfg.carb_sensitivity * absorbed
            drift = self.cfg.homeostatic_drift * (self.cfg.target_glucose - glucose)

            iob = (self.cfg.iob_decay * iob) + basal + bolus_effective

            glucose = np.clip(glucose + meal_rise - insulin_drop + drift, 35.0, 500.0)
            trajectory.append(float(glucose))

        return np.asarray(trajectory, dtype=np.float32)

    def _score_trajectory(self, trajectory: np.ndarray) -> float:
        """Planner objective: maximize cumulative one-step risk reward."""

        return float(sum(planner_risk_reward(float(g)) for g in trajectory))
