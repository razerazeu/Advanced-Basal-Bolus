"""Training orchestration for dual-SAC + planner glucose control."""

from __future__ import annotations

import csv
import json
from collections import deque
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
from utils.reward import BASAL_REWARD_WINDOW_HOURS, basal_step_reward_from_buffer, bolus_step_reward
from utils.seed import set_global_seed
from utils.state_builder import SlidingWindowStateBuilder


class Trainer:
    """End-to-end training loop coordinating env, controller, and SAC updates."""

    def __init__(self, config: ProjectConfig) -> None:
        self.cfg = config
        set_global_seed(config.seed)

        self.output_dir = self._build_output_dir(config.training.log_dir, config.training.run_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.env = SimglucoseEnv(config.env, training=True)
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

        self.planner = G2P2CPlanner(config.planner)
        self.controller = HybridController(
            state_builder=self.state_builder,
            basal_agent=self.basal_agent,
            bolus_agent=self.bolus_agent,
            planner=self.planner,
            step_minutes=config.env.step_minutes,
        )

        self.train_metrics_path = self.output_dir / "train_metrics.csv"
        self._csv_fields: list[str] | None = None
        self._meal_window_steps = max(
            1,
            int(
                round(
                    self.cfg.training.bolus_meal_timing_window_minutes / max(float(self.cfg.env.step_minutes), 1.0)
                )
            ),
        )
        self._basal_reward_window_steps = max(
            1,
            int(round((BASAL_REWARD_WINDOW_HOURS * 60) / max(float(self.cfg.env.step_minutes), 1.0))),
        )

    def train(self) -> list[dict[str, float | str]]:
        """Run staged training and return combined phase summaries."""

        return self.train_staged()

    def train_staged(self) -> list[dict[str, float | str]]:
        """Run basal pretraining then bolus training with frozen basal policy."""

        all_summaries: list[dict[str, float | str]] = []

        basal_summaries = self._run_phase(
            phase_name="basal_pretrain",
            scenario=self.cfg.training.basal_pretrain_scenario,
            episodes=self.cfg.training.basal_pretrain_episodes,
            train_basal=True,
            train_bolus=False,
            bolus_enabled=False,
            planner_enabled=self.cfg.training.use_planner_during_training,
            fasting_only=True,
            global_episode_offset=0,
        )
        all_summaries.extend(basal_summaries)

        bolus_summaries = self._run_phase(
            phase_name="bolus_train",
            scenario=self.cfg.training.bolus_train_scenario,
            episodes=self.cfg.training.bolus_train_episodes,
            train_basal=not self.cfg.training.freeze_basal_during_bolus,
            train_bolus=True,
            bolus_enabled=True,
            planner_enabled=self.cfg.training.use_planner_during_training,
            fasting_only=False,
            global_episode_offset=len(all_summaries),
        )
        all_summaries.extend(bolus_summaries)

        self.save_checkpoints(tag="latest")
        self._write_summary(self.output_dir / "train_summary.json", all_summaries)

        self.env.close()
        return all_summaries

    def pretrain_basal(self) -> list[dict[str, float | str]]:
        """Run only basal pretraining (Phase 1)."""

        summaries = self._run_phase(
            phase_name="basal_pretrain",
            scenario=self.cfg.training.basal_pretrain_scenario,
            episodes=self.cfg.training.basal_pretrain_episodes,
            train_basal=True,
            train_bolus=False,
            bolus_enabled=False,
            planner_enabled=self.cfg.training.use_planner_during_training,
            fasting_only=True,
            global_episode_offset=0,
        )
        self.save_checkpoints(tag="latest")
        self._write_summary(self.output_dir / "train_summary.json", summaries)
        self.env.close()
        return summaries

    def train_bolus(self, basal_checkpoint: Path | None = None) -> list[dict[str, float | str]]:
        """Run only bolus training (Phase 2), optionally loading pretrained basal."""

        if basal_checkpoint is not None:
            self.basal_agent.load(str(basal_checkpoint))
        elif self.cfg.training.freeze_basal_during_bolus:
            raise ValueError(
                "Running bolus-only training with frozen basal requires --basal-checkpoint. "
                "Use --mode train for staged training or provide a pretrained basal checkpoint."
            )

        summaries = self._run_phase(
            phase_name="bolus_train",
            scenario=self.cfg.training.bolus_train_scenario,
            episodes=self.cfg.training.bolus_train_episodes,
            train_basal=not self.cfg.training.freeze_basal_during_bolus,
            train_bolus=True,
            bolus_enabled=True,
            planner_enabled=self.cfg.training.use_planner_during_training,
            fasting_only=False,
            global_episode_offset=0,
        )
        self.save_checkpoints(tag="latest")
        self._write_summary(self.output_dir / "train_summary.json", summaries)
        self.env.close()
        return summaries

    def _run_phase(
        self,
        *,
        phase_name: str,
        scenario: str,
        episodes: int,
        train_basal: bool,
        train_bolus: bool,
        bolus_enabled: bool,
        planner_enabled: bool,
        fasting_only: bool,
        global_episode_offset: int,
    ) -> list[dict[str, float | str]]:
        self.env.set_scenario(scenario)
        self.controller.configure_phase(
            basal_training=train_basal,
            bolus_training=train_bolus,
            bolus_enabled=bolus_enabled,
            planner_enabled=planner_enabled,
            fixed_bolus=0.0,
        )

        summaries: list[dict[str, float | str]] = []
        for local_episode in range(1, episodes + 1):
            episode = global_episode_offset + local_episode
            summary = self._run_episode(
                episode=episode,
                phase_name=phase_name,
                scenario_name=scenario,
                train_basal=train_basal,
                train_bolus=train_bolus,
                fasting_only=fasting_only,
            )
            summaries.append(summary)
            self._append_metrics_csv(summary)

            if local_episode % self.cfg.training.checkpoint_every == 0:
                self.save_checkpoints(tag=f"{phase_name}_ep_{local_episode}")

        self.save_checkpoints(tag=f"{phase_name}_latest")
        self._write_summary(self.output_dir / f"{phase_name}_summary.json", summaries)
        return summaries

    def _run_episode(
        self,
        *,
        episode: int,
        phase_name: str,
        scenario_name: str,
        train_basal: bool,
        train_bolus: bool,
        fasting_only: bool,
    ) -> dict[str, float | str]:
        observation, info = self.env.reset()
        initial_glucose = SimglucoseEnv.extract_glucose(observation)
        start_time = info.get("time") if isinstance(info.get("time"), datetime) else self.cfg.env.start_time
        self.controller.reset(start_time=start_time, initial_cgm=initial_glucose)

        patient_name = str(info.get("patient_name", "unknown"))
        episode_seed = float(info.get("episode_seed", -1.0))

        metrics = EpisodeMetrics()
        basal_actor_losses: list[float] = []
        bolus_actor_losses: list[float] = []
        basal_critic_losses: list[float] = []
        bolus_critic_losses: list[float] = []
        planner_interventions = 0
        planner_hard_safety_count = 0
        planner_predicted_low_steps = 0
        basal_updates = 0
        bolus_updates = 0
        meal_window_steps_remaining = 0
        recent_meal_grams = 0.0
        basal_bg_buffer: deque[float] = deque(maxlen=self._basal_reward_window_steps)

        done = False
        step_count = 0

        while not done and step_count < self.cfg.training.max_steps_per_episode:
            if fasting_only and self._past_fasting_window(info):
                break

            action = self.controller.policy(observation=observation, reward=0.0, done=done, info=info)
            decision = self.controller.last_decision

            step_output = self.env.step(action)
            next_obs = step_output.observation
            done = step_output.done
            next_info = step_output.info

            next_glucose = SimglucoseEnv.extract_glucose(next_obs)
            next_meal = self._extract_meal(next_info)

            basal_bg_buffer.append(next_glucose)

            if next_meal > 0.0:
                meal_window_steps_remaining = max(meal_window_steps_remaining, self._meal_window_steps)
                recent_meal_grams = next_meal
            meal_window_active = meal_window_steps_remaining > 0

            next_state = self.state_builder.preview_state(
                cgm=next_glucose,
                insulin=float(action.basal + action.bolus),
                meal=next_meal,
            )

            r_basal = basal_step_reward_from_buffer(tuple(basal_bg_buffer))
            r_bolus = bolus_step_reward(
                next_glucose,
                next_meal,
                float(action.bolus),
                meal_window_active=meal_window_active,
                recent_meal_grams=recent_meal_grams,
            )

            if meal_window_steps_remaining > 0:
                meal_window_steps_remaining -= 1
            else:
                recent_meal_grams = 0.0

            if decision is not None and train_basal:
                basal_action_for_replay = self.basal_agent.normalized_from_basal(decision.final_basal)
                self.basal_agent.remember(
                    state=decision.state,
                    action=basal_action_for_replay,
                    reward=r_basal,
                    next_state=next_state,
                    done=done,
                )

            if decision is not None and train_bolus:
                bolus_action_for_replay = self.bolus_agent.normalized_from_bolus(decision.final_bolus)
                self.bolus_agent.remember(
                    state=decision.state,
                    action=bolus_action_for_replay,
                    reward=r_bolus,
                    next_state=next_state,
                    done=done,
                )

            if decision is not None:
                planner_active = decision.planner.mode not in ("bypass", "disabled")
                if planner_active:
                    if decision.planner.intervened:
                        planner_interventions += 1
                    if decision.planner.hard_safety_applied:
                        planner_hard_safety_count += 1
                    if decision.planner.predicted_min_glucose < self.cfg.planner.low_glucose_threshold:
                        planner_predicted_low_steps += 1

            if train_basal:
                for _ in range(self.cfg.sac_basal.updates_per_step):
                    basal_stats = self.basal_agent.update()
                    if basal_stats is not None:
                        basal_updates += 1
                        basal_actor_losses.append(basal_stats.actor_loss)
                        basal_critic_losses.append(0.5 * (basal_stats.critic1_loss + basal_stats.critic2_loss))

            if train_bolus:
                for _ in range(self.cfg.sac_bolus.updates_per_step):
                    bolus_stats = self.bolus_agent.update()
                    if bolus_stats is not None:
                        bolus_updates += 1
                        bolus_actor_losses.append(bolus_stats.actor_loss)
                        bolus_critic_losses.append(0.5 * (bolus_stats.critic1_loss + bolus_stats.critic2_loss))

            metrics.update(
                glucose=next_glucose,
                basal=float(action.basal),
                bolus=float(action.bolus),
                meal=next_meal,
                reward_basal=r_basal,
                reward_bolus=r_bolus,
            )

            observation = next_obs
            info = next_info
            step_count += 1

        summary = metrics.summary()
        summary["episode"] = float(episode)
        summary["steps"] = float(step_count)
        summary["using_mock_env"] = float(self.env.using_mock)
        summary["basal_actor_loss_mean"] = float(np.mean(basal_actor_losses)) if basal_actor_losses else 0.0
        summary["bolus_actor_loss_mean"] = float(np.mean(bolus_actor_losses)) if bolus_actor_losses else 0.0
        summary["basal_critic_loss_mean"] = float(np.mean(basal_critic_losses)) if basal_critic_losses else 0.0
        summary["bolus_critic_loss_mean"] = float(np.mean(bolus_critic_losses)) if bolus_critic_losses else 0.0
        summary["planner_intervention_steps"] = float(planner_interventions)
        summary["planner_intervention_rate"] = float(planner_interventions / max(step_count, 1))
        summary["planner_hard_safety_steps"] = float(planner_hard_safety_count)
        summary["planner_predicted_low_steps"] = float(planner_predicted_low_steps)
        summary["basal_updates"] = float(basal_updates)
        summary["bolus_updates"] = float(bolus_updates)
        summary["basal_replay_size"] = float(len(self.basal_agent.replay))
        summary["bolus_replay_size"] = float(len(self.bolus_agent.replay))
        summary["patient_name"] = patient_name
        summary["episode_seed"] = episode_seed
        summary["phase"] = phase_name
        summary["scenario"] = scenario_name
        summary["planner_enabled"] = float(self.controller.planner_enabled)
        summary["train_basal"] = float(train_basal)
        summary["train_bolus"] = float(train_bolus)
        return summary

    def save_checkpoints(self, tag: str = "latest") -> None:
        """Save both agents' checkpoints."""

        basal_path = self.ckpt_dir / f"basal_{tag}.pt"
        bolus_path = self.ckpt_dir / f"bolus_{tag}.pt"
        self.basal_agent.save(str(basal_path))
        self.bolus_agent.save(str(bolus_path))

    def _append_metrics_csv(self, metrics: dict[str, float | str]) -> None:
        if self._csv_fields is None:
            self._csv_fields = list(metrics.keys())

        write_header = not self.train_metrics_path.exists()
        with self.train_metrics_path.open("a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self._csv_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

    @staticmethod
    def _build_output_dir(root: Path, run_name: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return root / f"{run_name}_{timestamp}"

    @staticmethod
    def _write_summary(path: Path, summaries: list[dict[str, float | str]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)

    @staticmethod
    def _extract_meal(info: dict) -> float:
        for key in ("meal", "meal_grams", "CHO", "carbs"):
            if key in info:
                try:
                    return float(info[key])
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _past_fasting_window(self, info: dict) -> bool:
        end_minutes = (self.cfg.training.basal_fasting_end_hour * 60) + self.cfg.training.basal_fasting_end_minute
        current_time = info.get("time")
        if not isinstance(current_time, datetime):
            return False
        current_minutes = (current_time.hour * 60) + current_time.minute
        return current_minutes >= end_minutes
