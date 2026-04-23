"""Hybrid basal-bolus controller integrating dual SAC policies and planner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from agents.basal_agent import BasalAgent
from agents.bolus_agent import BolusAgent
from env.simglucose_env import Action, SimglucoseEnv
from planner.g2p2c_planner import G2P2CPlanner, PlannerResult
from utils.state_builder import SlidingWindowStateBuilder


@dataclass
class ControllerDecision:
    """Stores all fields needed by the trainer for replay updates."""

    time: datetime
    state: np.ndarray
    glucose: float
    meal: float
    basal_norm_action: float
    bolus_norm_action: float
    proposed_basal: float
    proposed_bolus: float
    final_basal: float
    final_bolus: float
    basal_decision_made: bool
    bolus_decision_made: bool
    planner: PlannerResult


class HybridController:
    """Main integration point for slow basal, fast bolus, and planning safety."""

    def __init__(
        self,
        state_builder: SlidingWindowStateBuilder,
        basal_agent: BasalAgent,
        bolus_agent: BolusAgent,
        planner: G2P2CPlanner,
        step_minutes: int = 5,
    ) -> None:
        self.state_builder = state_builder
        self.basal_agent = basal_agent
        self.bolus_agent = bolus_agent
        self.planner = planner
        self.step_minutes = step_minutes

        self.training = True
        self.basal_training = True
        self.bolus_training = True
        self.bolus_enabled = True
        self.planner_enabled = True
        self.fixed_bolus = 0.0
        self.total_env_steps = 0
        self.basal_phase_steps = 0
        self.bolus_phase_steps = 0
        self.last_total_insulin = 0.0
        self.current_time = datetime(2026, 1, 1, 0, 0)
        self.last_decision: ControllerDecision | None = None

    def set_mode(self, training: bool) -> None:
        """Set controller mode for stochastic training vs deterministic eval."""

        self.training = training
        self.basal_training = training
        self.bolus_training = training

    def configure_phase(
        self,
        *,
        basal_training: bool,
        bolus_training: bool,
        bolus_enabled: bool,
        planner_enabled: bool,
        fixed_bolus: float = 0.0,
    ) -> None:
        """Configure policy behavior for staged training and evaluation."""

        prev_basal_training = self.basal_training
        prev_bolus_training = self.bolus_training

        self.basal_training = basal_training
        self.bolus_training = bolus_training
        self.bolus_enabled = bolus_enabled
        self.planner_enabled = planner_enabled
        self.fixed_bolus = max(0.0, float(fixed_bolus))
        self.training = bool(basal_training or bolus_training)

        if basal_training and not prev_basal_training:
            self.basal_phase_steps = 0
        if bolus_training and not prev_bolus_training:
            self.bolus_phase_steps = 0

    def reset(self, start_time: datetime, initial_cgm: float = 110.0) -> None:
        """Reset internal controller state at episode start."""

        self.current_time = start_time
        self.last_total_insulin = 0.0
        self.last_decision = None

        self.state_builder.reset(initial_cgm=initial_cgm)
        self.basal_agent.reset()

    def policy(self, observation: object, reward: float, done: bool, info: dict | None) -> Action:
        """Simglucose-compatible controller API returning Action(basal, bolus)."""

        del reward, done  # Included for compatibility with controller signatures.
        info = info or {}

        sim_time = self._extract_time(info)
        glucose = SimglucoseEnv.extract_glucose(observation)
        meal = self._extract_meal(info)

        # Update rolling history before action selection.
        self.state_builder.update(cgm=glucose, insulin=self.last_total_insulin, meal=meal)
        state = self.state_builder.build_state()

        basal_norm, proposed_basal, basal_made = self.basal_agent.act(
            state=state,
            sim_time=sim_time,
            training=self.basal_training,
            total_env_steps=self.basal_phase_steps,
        )

        if self.bolus_enabled:
            bolus_norm, proposed_bolus, bolus_made = self.bolus_agent.act(
                state=state,
                sim_time=sim_time,
                meal_grams=meal,
                current_glucose=glucose,
                training=self.bolus_training,
                total_env_steps=self.bolus_phase_steps,
            )
        else:
            bolus_norm, proposed_bolus, bolus_made = -1.0, self.fixed_bolus, False

        if self.planner_enabled:
            final_basal, final_bolus, planner_result = self.planner.refine_action(
                proposed_basal=proposed_basal,
                proposed_bolus=proposed_bolus,
                current_glucose=glucose,
                meal_grams=meal,
                recent_insulin=self.last_total_insulin,
            )
        else:
            final_basal = float(np.clip(proposed_basal, *self.planner.cfg.basal_bounds))
            final_bolus = float(np.clip(proposed_bolus, *self.planner.cfg.bolus_bounds))
            planner_result = PlannerResult(
                basal=final_basal,
                bolus=final_bolus,
                score=0.0,
                trajectory=[float(glucose)],
                mode="bypass",
                predicted_min_glucose=float(glucose),
                intervened=False,
                hard_safety_applied=False,
            )

        action = Action(basal=float(final_basal), bolus=float(final_bolus))
        self.last_total_insulin = float(final_basal + final_bolus)

        self.last_decision = ControllerDecision(
            time=sim_time,
            state=state.copy(),
            glucose=float(glucose),
            meal=float(meal),
            basal_norm_action=float(basal_norm),
            bolus_norm_action=float(bolus_norm),
            proposed_basal=float(proposed_basal),
            proposed_bolus=float(proposed_bolus),
            final_basal=float(final_basal),
            final_bolus=float(final_bolus),
            basal_decision_made=bool(basal_made),
            bolus_decision_made=bool(bolus_made),
            planner=planner_result,
        )

        if self.basal_training:
            self.basal_phase_steps += 1
        if self.bolus_training and self.bolus_enabled:
            self.bolus_phase_steps += 1

        self.total_env_steps += 1
        return action

    def _extract_time(self, info: dict) -> datetime:
        maybe_time = info.get("time")
        if isinstance(maybe_time, datetime):
            self.current_time = maybe_time
            return maybe_time

        self.current_time = self.current_time + timedelta(minutes=self.step_minutes)
        return self.current_time

    @staticmethod
    def _extract_meal(info: dict) -> float:
        for key in ("meal", "meal_grams", "CHO", "carbs"):
            if key in info:
                try:
                    return float(info[key])
                except (TypeError, ValueError):
                    continue
        return 0.0
