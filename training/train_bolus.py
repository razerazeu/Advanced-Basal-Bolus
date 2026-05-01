from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.sac_basal import BasalSACAgent
from agents.sac_bolus import BolusSACAgent
from env.scenarios import meal_step_info
from env.simglucose_wrapper import DAY_STEPS, STATE_DIM, SimglucoseEnvWrapper
from evaluation.evaluate import append_jsonl, compute_metrics
from planner.planner import FrozenSafetyPlanner
from training.rewards import basal_reward, bolus_reward
from utils.normalizer import StateNormalizer


def train_bolus_phase(
    patient_names: list[str],
    normalizer: StateNormalizer,
    basal_checkpoint_path: str | Path = "checkpoints/basal_pretrained.pt",
    num_episodes: int = 300,
    seed: int = 314,
    device: str = "cpu",
    strict_checks: bool = False,
    bolus_checkpoint_path: str | Path = "checkpoints/bolus_trained.pt",
    log_path: str | Path = "logs/train_bolus.jsonl",
) -> tuple[BasalSACAgent, BolusSACAgent]:
    log_path = Path(log_path)
    if log_path.exists():
        log_path.unlink()

    basal_agent = BasalSACAgent(state_dim=STATE_DIM, device=device)
    extra = basal_agent.load(basal_checkpoint_path)
    norm_state = extra.get("normalizer") if isinstance(extra, dict) else None
    if isinstance(norm_state, dict):
        normalizer.load_state_dict(norm_state)

    basal_agent.freeze_actor()

    bolus_agent = BolusSACAgent(state_dim=STATE_DIM, device=device)
    planner = FrozenSafetyPlanner(horizon=6)
    window_entries: list[dict] = []
    scenario_a_episodes = max(1, num_episodes // 2)

    for episode in range(1, num_episodes + 1):
        scenario_name = "A" if episode <= scenario_a_episodes else "B"
        patient = patient_names[(episode - 1) % len(patient_names)]
        env = SimglucoseEnvWrapper(
            patient_name=patient,
            scenario_name=scenario_name,
            normalizer=normalizer,
            seed=int(seed + episode),
            update_normalizer=True,
        )

        state = env.reset()
        cached_basal = env.clip_basal(env.u2ss)

        glucose_trace: list[float] = []
        basal_trace: list[float] = []
        bolus_trace: list[float] = []
        r_basal_trace: list[float] = []
        r_bolus_trace: list[float] = []
        planner_overrides = 0

        for step in range(DAY_STEPS):
            minute_of_day = (step * 5) % (24 * 60)
            if minute_of_day == 7 * 60:
                cached_basal = env.clip_basal(basal_agent.select_action(state, deterministic=True))

            if step % 3 == 0:
                proposed_bolus = env.clip_bolus(bolus_agent.select_action(state, deterministic=False))
            else:
                proposed_bolus = 0.0

            current_cgm = float(env.cgm_hist[-1])
            _meal_flag_now, meal_carbs_now = meal_step_info(env.meals, env.minute_of_day, step_minutes=5)
            final_basal, bolus, _best_idx, override, _traj = planner.select_action(
                proposed_basal=cached_basal,
                proposed_bolus=proposed_bolus,
                current_cgm=current_cgm,
                meal_carbs=meal_carbs_now,
                insulin_sensitivity_scale=env.insulin_sensitivity_scale,
                cgm_trend=env.cgm_trend(),
            )
            if override:
                planner_overrides += 1
            final_basal = env.clip_basal(final_basal)
            bolus = env.clip_bolus(bolus)

            next_state, done, info = env.step(final_basal, bolus)
            cgm_history, _, meal_history = env.history_arrays()

            r_basal = basal_reward(cgm_history, info["current_cgm"])
            meal_recent = int(np.any(meal_history[-12:] > 0.0))
            r_bolus = bolus_reward(info["current_cgm"], meal_recent, bolus)

            bolus_agent.store(state, bolus, r_bolus, next_state, done)

            if len(bolus_agent.replay_buffer) >= bolus_agent.cfg.batch_size:
                losses = bolus_agent.update()
                bolus_loss = losses["critic_loss"]
                if strict_checks:
                    assert bolus_loss != 0, "Bolus critic loss is zero — check reward signal"

            glucose_trace.append(float(info["current_cgm"]))
            basal_trace.append(float(final_basal))
            bolus_trace.append(float(bolus))
            r_basal_trace.append(float(r_basal))
            r_bolus_trace.append(float(r_bolus))

            state = next_state
            if done:
                break

        if strict_checks:
            assert min(glucose_trace) > 39, "Patient hitting simulator floor — check action clipping"

        metrics = compute_metrics(glucose_trace, basal_trace, bolus_trace)
        log_entry = {
            "episode": episode,
            "patient": patient,
            "scenario": scenario_name,
            **metrics,
            "r_basal_mean": float(np.mean(r_basal_trace)),
            "r_bolus_mean": float(np.mean(r_bolus_trace)),
            "planner_overrides": int(planner_overrides),
        }
        append_jsonl(log_path, log_entry)
        window_entries.append(log_entry)

        if episode % 10 == 0:
            start_episode = episode - 9
            mean_tir = float(np.mean([entry["TIR"] for entry in window_entries]))
            mean_tbr = float(np.mean([entry["TBR"] for entry in window_entries]))
            mean_g = float(np.mean([entry["mean_G"] for entry in window_entries]))
            total_overrides = int(sum(entry["planner_overrides"] for entry in window_entries))
            print(
                f"Episodes {start_episode}-{episode} | mean TIR: {mean_tir:.1f}% | "
                f"mean TBR: {mean_tbr:.1f}% | mean_G: {mean_g:.1f} | "
                f"planner overrides: {total_overrides} (total)"
            )
            window_entries.clear()

    bolus_checkpoint_path = Path(bolus_checkpoint_path)
    bolus_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    bolus_agent.save(bolus_checkpoint_path, extra={"normalizer": normalizer.state_dict(), "phase": "bolus_train"})

    return basal_agent, bolus_agent
