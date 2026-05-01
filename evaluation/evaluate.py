from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from env.scenarios import meal_step_info
from env.simglucose_wrapper import DAY_STEPS, SimglucoseEnvWrapper
from planner.planner import FrozenSafetyPlanner
from training.rewards import basal_reward, bolus_reward
from utils.normalizer import StateNormalizer


def compute_metrics(glucose_trace: list[float], basal_trace: list[float], bolus_trace: list[float]) -> dict[str, float]:
    avg_daily_basal = float(sum(b * 5.0 for b in basal_trace))
    avg_daily_bolus = float(sum(bolus_trace))

    g = np.asarray(glucose_trace, dtype=np.float32)
    return {
        "min_G": float(np.min(g)),
        "max_G": float(np.max(g)),
        "mean_G": float(np.mean(g)),
        "TBR2": float(100.0 * np.mean(g < 54.0)),
        "TBR": float(100.0 * np.mean(g < 70.0)),
        "TIR": float(100.0 * np.mean((g >= 70.0) & (g <= 180.0))),
        "TAR": float(100.0 * np.mean(g > 180.0)),
        "TAR2": float(100.0 * np.mean(g > 250.0)),
        "avg_daily_bolus": avg_daily_bolus,
        "avg_daily_basal": avg_daily_basal,
    }


def append_jsonl(path: str | Path, entry: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        last_line = lines[-1] if lines else ""
        if last_line == line:
            return
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def summarize_eval_entries(entries: list[dict], scenario_name: str) -> dict:
    metric_keys = [
        "min_G",
        "max_G",
        "mean_G",
        "TBR2",
        "TBR",
        "TIR",
        "TAR",
        "TAR2",
        "avg_daily_bolus",
        "avg_daily_basal",
        "r_basal_mean",
        "r_bolus_mean",
        "planner_overrides",
    ]
    summary = {
        "entry_type": "scenario_summary",
        "episode": None,
        "patient": "ALL",
        "scenario": scenario_name,
        "num_patients": len(entries),
    }
    for key in metric_keys:
        values = np.asarray([entry[key] for entry in entries if key in entry], dtype=np.float32)
        if values.size == 0:
            continue
        summary[key] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return summary


def evaluate_episode(
    episode_idx: int,
    patient_name: str,
    scenario_name: str,
    basal_agent,
    bolus_agent,
    planner: FrozenSafetyPlanner,
    normalizer: StateNormalizer,
    seed: int,
    log_path: str | Path | None = None,
) -> dict:
    env = SimglucoseEnvWrapper(
        patient_name=patient_name,
        scenario_name=scenario_name,
        normalizer=normalizer,
        seed=seed,
        update_normalizer=False,
    )

    state = env.reset()
    cached_basal = env.clip_basal(env.u2ss)

    glucose_trace: list[float] = []
    basal_trace: list[float] = []
    bolus_trace: list[float] = []
    rb_trace: list[float] = []
    rbo_trace: list[float] = []
    planner_overrides = 0

    for step in range(DAY_STEPS):
        minute_of_day = (step * 5) % (24 * 60)
        if minute_of_day == 7 * 60:
            cached_basal = env.clip_basal(basal_agent.select_action(state, deterministic=True))

        if step % 3 == 0:
            proposed_bolus = env.clip_bolus(bolus_agent.select_action(state, deterministic=True))
        else:
            proposed_bolus = 0.0

        cgm_now = float(env.cgm_hist[-1])
        _flag, meal_carbs = meal_step_info(env.meals, env.minute_of_day, step_minutes=5)

        final_basal, final_bolus, _best_idx, override, _traj = planner.select_action(
            proposed_basal=cached_basal,
            proposed_bolus=proposed_bolus,
            current_cgm=cgm_now,
            meal_carbs=meal_carbs,
            insulin_sensitivity_scale=env.insulin_sensitivity_scale,
            cgm_trend=env.cgm_trend(),
        )
        if override:
            planner_overrides += 1
        final_basal = env.clip_basal(final_basal)
        final_bolus = env.clip_bolus(final_bolus)

        next_state, done, info = env.step(final_basal, final_bolus)

        cgm_history, _, meal_history = env.history_arrays()
        r_basal = basal_reward(cgm_history, info["current_cgm"])
        meal_recent = int(np.any(meal_history[-12:] > 0.0))
        r_bolus = bolus_reward(info["current_cgm"], meal_recent, final_bolus)

        glucose_trace.append(float(info["current_cgm"]))
        basal_trace.append(float(final_basal))
        bolus_trace.append(float(final_bolus))
        rb_trace.append(r_basal)
        rbo_trace.append(r_bolus)

        state = next_state
        if done:
            break

    metrics = compute_metrics(glucose_trace, basal_trace, bolus_trace)
    entry = {
        "episode": episode_idx,
        "patient": patient_name,
        "scenario": scenario_name,
        **metrics,
        "r_basal_mean": float(np.mean(rb_trace) if rb_trace else 0.0),
        "r_bolus_mean": float(np.mean(rbo_trace) if rbo_trace else 0.0),
        "planner_overrides": int(planner_overrides),
    }

    if log_path is not None:
        append_jsonl(log_path, entry)

    return entry
