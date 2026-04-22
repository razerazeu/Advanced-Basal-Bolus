"""Evaluation utility for trained basal/bolus SAC checkpoints."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from agents.basal_agent import BasalAgent
from agents.bolus_agent import BolusAgent
from config import ProjectConfig
from controller.hybrid_controller import HybridController
from env.simglucose_env import SimglucoseEnv
from planner.g2p2c_planner import G2P2CPlanner
from utils.metrics import EpisodeMetrics
from utils.reward import basal_step_reward, bolus_step_reward
from utils.seed import set_global_seed
from utils.state_builder import SlidingWindowStateBuilder


class Evaluator:
    """Deterministic evaluation runner with no learning updates."""

    def __init__(self, config: ProjectConfig, checkpoint_dir: Path) -> None:
        self.cfg = config
        if (checkpoint_dir / "basal_latest.pt").exists() or (checkpoint_dir / "bolus_latest.pt").exists():
            self.checkpoint_dir = checkpoint_dir
        elif (checkpoint_dir / "checkpoints").exists():
            self.checkpoint_dir = checkpoint_dir / "checkpoints"
        else:
            self.checkpoint_dir = checkpoint_dir

        set_global_seed(config.seed)

        self.env = SimglucoseEnv(config.env, training=False)
        self.state_builder = SlidingWindowStateBuilder(config.state)

        self.basal_agent = BasalAgent(
            state_dim=config.state.state_dim,
            sac_cfg=config.sac_basal,
            cfg=config.basal_agent,
            device=config.training.device,
        )
        self.bolus_agent = BolusAgent(
            state_dim=config.state.state_dim,
            sac_cfg=config.sac_bolus,
            cfg=config.bolus_agent,
            device=config.training.device,
        )

        self._load_latest_or_named()

        self.planner = G2P2CPlanner(config.planner)
        self.controller = HybridController(
            state_builder=self.state_builder,
            basal_agent=self.basal_agent,
            bolus_agent=self.bolus_agent,
            planner=self.planner,
            step_minutes=config.env.step_minutes,
        )
        self.controller.configure_phase(
            basal_training=False,
            bolus_training=False,
            bolus_enabled=True,
            planner_enabled=True,
            fixed_bolus=0.0,
        )
        self._meal_window_steps = max(
            1,
            int(
                round(
                    self.cfg.training.bolus_meal_timing_window_minutes / max(float(self.cfg.env.step_minutes), 1.0)
                )
            ),
        )

    def evaluate(
        self,
        num_episodes: int | None = None,
        scenarios: tuple[str, ...] | None = None,
    ) -> list[dict[str, float | str]]:
        """Run deterministic evaluation across one or more scenarios."""

        scenario_list = scenarios or self.cfg.training.eval_scenarios
        episodes_per_scenario = (
            int(num_episodes)
            if num_episodes is not None
            else max(int(self.cfg.training.eval_episodes_per_scenario), int(self.env.patient_pool_size))
        )

        summaries: list[dict[str, float | str]] = []
        global_episode = 1

        for scenario in scenario_list:
            self.env.set_scenario(scenario)
            scenario_summaries, global_episode = self._evaluate_scenario(
                scenario=scenario,
                num_episodes=episodes_per_scenario,
                global_episode_start=global_episode,
            )
            summaries.extend(scenario_summaries)

        self.env.close()
        return summaries

    def _evaluate_scenario(
        self,
        *,
        scenario: str,
        num_episodes: int,
        global_episode_start: int,
    ) -> tuple[list[dict[str, float | str]], int]:
        """Evaluate a single scenario for a fixed number of episodes."""

        summaries: list[dict[str, float | str]] = []
        global_episode = global_episode_start

        for _ in range(num_episodes):
            observation, info = self.env.reset()
            initial_glucose = SimglucoseEnv.extract_glucose(observation)
            start_time = info.get("time") if isinstance(info.get("time"), datetime) else self.cfg.env.start_time
            self.controller.reset(start_time=start_time, initial_cgm=initial_glucose)

            patient_name = str(info.get("patient_name", "unknown"))
            episode_seed = float(info.get("episode_seed", -1.0))

            metrics = EpisodeMetrics()
            done = False
            steps = 0
            planner_interventions = 0
            planner_hard_safety_count = 0
            planner_predicted_low_steps = 0
            meal_window_steps_remaining = 0
            recent_meal_grams = 0.0

            while not done and steps < self.cfg.training.max_steps_per_episode:
                action = self.controller.policy(observation=observation, reward=0.0, done=done, info=info)
                decision = self.controller.last_decision
                step_output = self.env.step(action)

                observation = step_output.observation
                info = step_output.info
                done = step_output.done

                glucose = SimglucoseEnv.extract_glucose(observation)
                meal = self._extract_meal(info)

                if meal > 0.0:
                    meal_window_steps_remaining = max(meal_window_steps_remaining, self._meal_window_steps)
                    recent_meal_grams = meal
                meal_window_active = meal_window_steps_remaining > 0
                if meal_window_steps_remaining > 0:
                    meal_window_steps_remaining -= 1
                else:
                    recent_meal_grams = 0.0

                if decision is not None:
                    if decision.planner.intervened:
                        planner_interventions += 1
                    if decision.planner.hard_safety_applied:
                        planner_hard_safety_count += 1
                    if decision.planner.predicted_min_glucose < self.cfg.planner.low_glucose_threshold:
                        planner_predicted_low_steps += 1

                metrics.update(
                    glucose=glucose,
                    basal=float(action.basal),
                    bolus=float(action.bolus),
                    meal=meal,
                    reward_basal=basal_step_reward(glucose),
                    reward_bolus=bolus_step_reward(
                        glucose,
                        meal,
                        float(action.bolus),
                        meal_window_active=meal_window_active,
                        recent_meal_grams=recent_meal_grams,
                    ),
                )

                steps += 1

            summary = metrics.summary()
            summary["episode"] = float(global_episode)
            summary["steps"] = float(steps)
            summary["using_mock_env"] = float(self.env.using_mock)
            summary["planner_intervention_steps"] = float(planner_interventions)
            summary["planner_intervention_rate"] = float(planner_interventions / max(steps, 1))
            summary["planner_hard_safety_steps"] = float(planner_hard_safety_count)
            summary["planner_predicted_low_steps"] = float(planner_predicted_low_steps)
            summary["patient_name"] = patient_name
            summary["episode_seed"] = episode_seed
            summary["scenario"] = scenario
            summary["phase"] = "eval"
            summaries.append(summary)
            global_episode += 1

        return summaries, global_episode

    def save_summary(self, summaries: list[dict[str, float | str]], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)

    def _load_latest_or_named(self) -> None:
        basal = self._find_checkpoint("basal")
        bolus = self._find_checkpoint("bolus")
        self.basal_agent.load(str(basal))
        self.bolus_agent.load(str(bolus))

    def _find_checkpoint(self, prefix: str) -> Path:
        latest = self.checkpoint_dir / f"{prefix}_latest.pt"
        if latest.exists():
            return latest

        matches = sorted(self.checkpoint_dir.glob(f"{prefix}_*.pt"))
        if not matches:
            raise FileNotFoundError(f"No checkpoint found for prefix '{prefix}' in {self.checkpoint_dir}")
        return matches[-1]

    @staticmethod
    def _extract_meal(info: dict) -> float:
        for key in ("meal", "meal_grams", "CHO", "carbs"):
            if key in info:
                try:
                    return float(info[key])
                except (TypeError, ValueError):
                    continue
        return 0.0
