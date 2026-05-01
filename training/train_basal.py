from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.sac_basal import BasalSACAgent
from env.simglucose_wrapper import DAY_STEPS, STATE_DIM, SimglucoseEnvWrapper
from evaluation.evaluate import append_jsonl, compute_metrics
from training.rewards import basal_reward, bolus_reward
from utils.normalizer import StateNormalizer


def train_basal_phase(
    patient_names: list[str],
    normalizer: StateNormalizer,
    num_episodes: int = 200,
    seed: int = 42,
    device: str = "cpu",
    strict_checks: bool = False,
    checkpoint_path: str | Path = "checkpoints/basal_pretrained.pt",
    log_path: str | Path = "logs/train_basal.jsonl",
) -> BasalSACAgent:
    log_path = Path(log_path)
    if log_path.exists():
        log_path.unlink()

    agent = BasalSACAgent(state_dim=STATE_DIM, device=device)
    window_entries: list[dict] = []

    warm_start_rates: list[float] = []
    for idx, patient_name in enumerate(patient_names):
        warm_env = SimglucoseEnvWrapper(
            patient_name=patient_name,
            scenario_name="A",
            normalizer=normalizer,
            seed=int(seed + 10_000 + idx),
            update_normalizer=False,
        )
        warm_start_rates.append(warm_env.clip_basal(warm_env.u2ss))
    if warm_start_rates:
        agent.initialize_actor_output_bias(float(np.mean(warm_start_rates)), log_std_bias=-3.0)

    print("PHASE 1: basal only training")

    for episode in range(1, num_episodes + 1):
        patient = patient_names[(episode - 1) % len(patient_names)]
        env = SimglucoseEnvWrapper(
            patient_name=patient,
            scenario_name="A",
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

        for step in range(DAY_STEPS):
            minute_of_day = (step * 5) % (24 * 60)
            if minute_of_day == 7 * 60:
                cached_basal = env.clip_basal(agent.select_action(state, deterministic=False))

            if step % 3 == 0:
                bolus = env.clip_bolus(env.u2ss)
            else:
                bolus = 0.0

            next_state, done, info = env.step(cached_basal, bolus)
            cgm_history, _, meal_history = env.history_arrays()

            r_basal = basal_reward(cgm_history, info["current_cgm"])
            meal_recent = int(np.any(meal_history[-12:] > 0.0))
            r_bolus = bolus_reward(info["current_cgm"], meal_recent, bolus)

            agent.store(state, cached_basal, r_basal, next_state, done)

            if len(agent.replay_buffer) >= agent.cfg.batch_size:
                losses = agent.update()
                basal_loss = losses["critic_loss"]
                if strict_checks:
                    assert basal_loss != 0, "Basal critic loss is zero — check reward signal"

            glucose_trace.append(float(info["current_cgm"]))
            basal_trace.append(float(cached_basal))
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
            "scenario": "A",
            **metrics,
            "r_basal_mean": float(np.mean(r_basal_trace)),
            "r_bolus_mean": float(np.mean(r_bolus_trace)),
            "planner_overrides": 0,
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

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(checkpoint_path, extra={"normalizer": normalizer.state_dict(), "phase": "basal_pretrain"})
    return agent
