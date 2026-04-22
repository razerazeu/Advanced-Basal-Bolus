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
        """Evaluate candidate trajectories and select a safer action."""

        basal = float(np.clip(proposed_basal, *self.cfg.basal_bounds))
        bolus = float(np.clip(proposed_bolus, *self.cfg.bolus_bounds))

        if meal_grams <= 0.0 and current_glucose <= 150.0:
            bolus = float(min(bolus, 0.05))
        if meal_grams <= 0.0 and current_glucose <= 130.0:
            bolus = 0.0

        if current_glucose <= self.cfg.low_glucose_threshold:
            basal = float(np.clip(basal * self.cfg.emergency_basal_scale, *self.cfg.basal_bounds))
            bolus = float(np.clip(min(bolus, self.cfg.emergency_bolus_max), *self.cfg.bolus_bounds))

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

        candidates = self._generate_candidates(
            basal=basal,
            bolus=bolus,
            current_glucose=current_glucose,
            meal_grams=meal_grams,
        )

        best_score = float("-inf")
        best_candidate = (basal, bolus)

        for cand_basal, cand_bolus in candidates:
            trajectory = self._rollout(
                current_glucose=current_glucose,
                initial_meal_grams=meal_grams,
                basal=cand_basal,
                bolus=cand_bolus,
                initial_recent_insulin=recent_insulin,
            )
            score = self._score_trajectory(trajectory)
            if score > best_score:
                best_score = score
                best_candidate = (cand_basal, cand_bolus)

        if self.cfg.mode == "hard":
            final_basal, final_bolus = best_candidate
        else:
            # Soft mode nudges the policy output toward the safest candidate.
            final_basal = basal + self.cfg.soft_adjustment_gain * (best_candidate[0] - basal)
            final_bolus = bolus + self.cfg.soft_adjustment_gain * (best_candidate[1] - bolus)

            final_basal = float(np.clip(final_basal, *self.cfg.basal_bounds))
            final_bolus = float(np.clip(final_bolus, *self.cfg.bolus_bounds))

        hard_safety_applied = False

        # Enforce a final hard safety gate even after candidate optimization.
        if current_glucose <= self.cfg.severe_hypoglycemia_threshold:
            final_basal = float(self.cfg.basal_bounds[0])
            final_bolus = float(self.cfg.bolus_bounds[0])
            hard_safety_applied = True
        elif current_glucose <= self.cfg.low_glucose_threshold:
            final_basal = float(np.clip(min(final_basal, basal), *self.cfg.basal_bounds))
            final_bolus = float(np.clip(min(final_bolus, self.cfg.emergency_bolus_max), *self.cfg.bolus_bounds))
            hard_safety_applied = True

        final_trajectory = self._rollout(
            current_glucose=current_glucose,
            initial_meal_grams=meal_grams,
            basal=float(final_basal),
            bolus=float(final_bolus),
            initial_recent_insulin=recent_insulin,
        )
        predicted_min = float(np.min(final_trajectory))
        predicted_low_threshold = self.cfg.low_glucose_threshold + self.cfg.predicted_safety_margin

        if (
            current_glucose <= (self.cfg.low_glucose_threshold + 20.0)
            and recent_insulin >= self.cfg.iob_safety_threshold
        ):
            final_basal = float(self.cfg.basal_bounds[0])
            final_bolus = float(self.cfg.bolus_bounds[0])
            hard_safety_applied = True
            final_trajectory = self._rollout(
                current_glucose=current_glucose,
                initial_meal_grams=meal_grams,
                basal=float(final_basal),
                bolus=float(final_bolus),
                initial_recent_insulin=recent_insulin,
            )
            predicted_min = float(np.min(final_trajectory))

        if predicted_min < predicted_low_threshold:
            safer_basal = float(np.clip(final_basal * self.cfg.emergency_basal_scale, *self.cfg.basal_bounds))
            safer_bolus = float(np.clip(min(final_bolus, self.cfg.emergency_bolus_max), *self.cfg.bolus_bounds))

            if (abs(safer_basal - final_basal) > 1e-9) or (abs(safer_bolus - final_bolus) > 1e-9):
                hard_safety_applied = True

            final_basal, final_bolus = safer_basal, safer_bolus
            final_trajectory = self._rollout(
                current_glucose=current_glucose,
                initial_meal_grams=meal_grams,
                basal=float(final_basal),
                bolus=float(final_bolus),
                initial_recent_insulin=recent_insulin,
            )
            predicted_min = float(np.min(final_trajectory))

        if predicted_min < self.cfg.severe_hypoglycemia_threshold:
            if final_basal > self.cfg.basal_bounds[0] or final_bolus > self.cfg.bolus_bounds[0]:
                hard_safety_applied = True

            final_basal = float(self.cfg.basal_bounds[0])
            final_bolus = float(self.cfg.bolus_bounds[0])
            final_trajectory = self._rollout(
                current_glucose=current_glucose,
                initial_meal_grams=meal_grams,
                basal=float(final_basal),
                bolus=float(final_bolus),
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
        basal: float,
        bolus: float,
        current_glucose: float,
        meal_grams: float,
    ) -> list[tuple[float, float]]:
        """Build candidate actions around SAC proposals."""

        candidates: set[tuple[float, float]] = set()
        allow_aggressive_insulin = (current_glucose >= 220.0) or (meal_grams >= 30.0)

        if allow_aggressive_insulin:
            scales = self.cfg.candidate_scales
            offsets = self.cfg.bolus_offsets
        else:
            scales = tuple(scale for scale in self.cfg.candidate_scales if scale <= 1.0)
            offsets = tuple(offset for offset in self.cfg.bolus_offsets if offset <= 0.0)

        if not scales:
            scales = (1.0,)
        if not offsets:
            offsets = (0.0,)

        conservative_basal = float(np.clip(0.5 * basal, *self.cfg.basal_bounds))
        candidates.add((float(np.clip(basal, *self.cfg.basal_bounds)), 0.0))
        candidates.add((conservative_basal, 0.0))
        candidates.add((self.cfg.basal_bounds[0], 0.0))

        for scale in scales:
            b_cand = float(np.clip(basal * scale, *self.cfg.basal_bounds))
            bo_cand = float(np.clip(bolus * scale, *self.cfg.bolus_bounds))
            candidates.add((b_cand, bo_cand))
            candidates.add((b_cand, 0.0))

            for offset in offsets:
                bo_off = float(np.clip(bo_cand + offset, *self.cfg.bolus_bounds))
                candidates.add((b_cand, bo_off))

        candidates.add(
            (
                float(np.clip(basal, *self.cfg.basal_bounds)),
                float(np.clip(bolus, *self.cfg.bolus_bounds)),
            )
        )

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
        """Planner objective: maximize cumulative risk-sensitive reward."""

        base_score = float(sum(planner_risk_reward(float(g)) for g in trajectory))

        predicted_low_threshold = self.cfg.low_glucose_threshold + self.cfg.predicted_safety_margin
        below_low = int(np.sum(trajectory < predicted_low_threshold))
        below_severe = int(np.sum(trajectory < self.cfg.severe_hypoglycemia_threshold))

        penalty = (below_low * self.cfg.hypoglycemia_penalty) + (
            below_severe * self.cfg.severe_hypoglycemia_penalty
        )
        return base_score - float(penalty)
