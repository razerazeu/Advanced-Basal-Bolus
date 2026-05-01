from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from agents.sac_base import CentralizedCriticTrainer
from agents.sac_basal import BasalSACAgent
from agents.sac_bolus import BolusSACAgent
from env.simglucose_wrapper import DAY_STEPS, SimglucoseEnvWrapper
from evaluation.evaluate import append_jsonl, compute_metrics, evaluate_episode, summarize_eval_entries
from planner.planner import FrozenSafetyPlanner
from training.rewards import basal_reward, bolus_reward
from training.train_basal import train_basal_phase
from training.train_bolus import train_bolus_phase
from utils.normalizer import StateNormalizer

NUM_PATIENTS = 3
ALL_ADULTS = [f"adult#{i:03d}" for i in range(1, 11)]

PHASE1_EPISODES = 200
PHASE2_EPISODES = 300
PHASE3_EPISODES = 100


def run_combined_phase(
    patient_names: list[str],
    normalizer: StateNormalizer,
    basal_checkpoint_path: str = "checkpoints/basal_pretrained.pt",
    bolus_checkpoint_path: str = "checkpoints/bolus_trained.pt",
    num_episodes: int = PHASE3_EPISODES,
    seed: int = 999,
    device: str = "cpu",
    strict_checks: bool = False,
    log_path: str = "logs/train_combined.jsonl",
) -> tuple[BasalSACAgent, BolusSACAgent]:
    log_path = Path(log_path)
    if log_path.exists():
        log_path.unlink()

    basal_agent = BasalSACAgent(state_dim=36, device=device)
    basal_extra = basal_agent.load(basal_checkpoint_path)
    if isinstance(basal_extra, dict) and isinstance(basal_extra.get("normalizer"), dict):
        normalizer.load_state_dict(basal_extra["normalizer"])

    bolus_agent = BolusSACAgent(state_dim=36, device=device)
    bolus_extra = bolus_agent.load(bolus_checkpoint_path)
    if isinstance(bolus_extra, dict) and isinstance(bolus_extra.get("normalizer"), dict):
        normalizer.load_state_dict(bolus_extra["normalizer"])

    basal_agent.unfreeze_actor()

    planner = FrozenSafetyPlanner(horizon=6)
    centralized_critic = CentralizedCriticTrainer(state_dim=36, action_dim=2, device=device)

    scenarios = ["A", "B", "C"]
    window_entries: list[dict] = []

    for episode in range(1, num_episodes + 1):
        scenario_name = scenarios[(episode - 1) % len(scenarios)]
        patient = patient_names[(episode - 1) % len(patient_names)]

        env = SimglucoseEnvWrapper(
            patient_name=patient,
            scenario_name=scenario_name,
            normalizer=normalizer,
            seed=int(seed + episode),
            update_normalizer=True,
        )

        state = env.reset()
        cached_basal = float(np.clip(env.u2ss, 0.0, 0.05))

        glucose_trace: list[float] = []
        basal_trace: list[float] = []
        bolus_trace: list[float] = []
        r_basal_trace: list[float] = []
        r_bolus_trace: list[float] = []
        planner_overrides = 0

        for step in range(DAY_STEPS):
            minute_of_day = (step * 5) % (24 * 60)
            if minute_of_day == 7 * 60:
                cached_basal = float(np.clip(basal_agent.select_action(state, deterministic=False), 0.0, 0.05))

            if step % 3 == 0:
                proposed_bolus = float(np.clip(bolus_agent.select_action(state, deterministic=False), 0.0, 0.5))
            else:
                proposed_bolus = 0.0

            current_cgm = float(env.cgm_hist[-1])
            from env.scenarios import meal_step_info

            meal_flag_now, meal_carbs_now = meal_step_info(env.meals, env.minute_of_day, step_minutes=5)

            final_basal, final_bolus, _best_idx, override, _traj = planner.select_action(
                proposed_basal=cached_basal,
                proposed_bolus=proposed_bolus,
                current_cgm=current_cgm,
                meal_carbs=meal_carbs_now,
            )
            if override:
                planner_overrides += 1

            next_state, done, info = env.step(final_basal, final_bolus)

            cgm_history, _, meal_history = env.history_arrays()
            r_basal = basal_reward(cgm_history, info["current_cgm"])
            meal_recent = int(np.any(meal_history[-12:] > 0.0))
            r_bolus = bolus_reward(info["current_cgm"], meal_recent, final_bolus)

            # CRITICAL: separate rewards and independent storage.
            basal_agent.store(state, final_basal, r_basal, next_state, done)
            bolus_agent.store(state, final_bolus, r_bolus, next_state, done)

            if len(basal_agent.replay_buffer) >= basal_agent.cfg.batch_size:
                basal_losses = basal_agent.update()
                if strict_checks:
                    assert basal_losses["critic_loss"] != 0, "Basal critic loss is zero — check reward signal"

            if len(bolus_agent.replay_buffer) >= bolus_agent.cfg.batch_size:
                bolus_losses = bolus_agent.update()
                if strict_checks:
                    assert bolus_losses["critic_loss"] != 0, "Bolus critic loss is zero — check reward signal"

            centralized_critic.update(state, np.asarray([final_basal, final_bolus], dtype=np.float32), 0.5 * (r_basal + r_bolus))

            glucose_trace.append(float(info["current_cgm"]))
            basal_trace.append(float(final_basal))
            bolus_trace.append(float(final_bolus))
            r_basal_trace.append(float(r_basal))
            r_bolus_trace.append(float(r_bolus))

            state = next_state
            if done:
                break

        if strict_checks:
            assert min(glucose_trace) > 39, "Patient hitting simulator floor — check action clipping"
            assert planner_overrides > 0, "Planner never changed an action — check scoring logic"

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

    basal_agent.save(basal_checkpoint_path, extra={"normalizer": normalizer.state_dict(), "phase": "combined_finetune"})
    bolus_agent.save(bolus_checkpoint_path, extra={"normalizer": normalizer.state_dict(), "phase": "combined_finetune"})

    return basal_agent, bolus_agent


def run_evaluation(
    patient_names: list[str],
    normalizer: StateNormalizer,
    basal_agent: BasalSACAgent,
    bolus_agent: BolusSACAgent,
    scenario_name: str,
    seed: int = 2026,
) -> None:
    eval_log_path = Path("logs/eval.jsonl")
    if eval_log_path.exists():
        eval_log_path.unlink()

    planner = FrozenSafetyPlanner(horizon=6)

    scenarios = [scenario_name] if scenario_name in {"A", "B", "C"} else ["A", "B", "C"]

    eval_episode = 1
    for sc in scenarios:
        scenario_entries: list[dict] = []
        for patient in patient_names:
            entry = evaluate_episode(
                episode_idx=eval_episode,
                patient_name=patient,
                scenario_name=sc,
                basal_agent=basal_agent,
                bolus_agent=bolus_agent,
                planner=planner,
                normalizer=normalizer,
                seed=seed + eval_episode,
                log_path="logs/eval.jsonl",
            )
            scenario_entries.append(entry)
            print(
                f"[Eval] Ep {eval_episode} | {patient} | Sc {sc} | "
                f"TIR: {entry['TIR']:.1f}% | TBR: {entry['TBR']:.1f}% | "
                f"mean_G: {entry['mean_G']:.1f} | planner overrides: {entry['planner_overrides']}"
            )
            eval_episode += 1

        summary = summarize_eval_entries(scenario_entries, sc)
        append_jsonl("logs/eval.jsonl", summary)
        print(
            f"[Eval Summary] Sc {sc} | mean TIR: {summary['TIR']:.1f}% | "
            f"mean TBR: {summary['TBR']:.1f}% | mean_G: {summary['mean_G']:.1f} | "
            f"planner overrides: {summary['planner_overrides']:.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent SAC + Frozen Planner for Simglucose")
    parser.add_argument("--scenario", type=str, default="A", choices=["A", "B", "C", "all"], help="Evaluation scenario")
    parser.add_argument("--patients", type=int, default=NUM_PATIENTS, help="Number of adult patients to use")
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["basal", "bolus", "combined", "all"],
        help="Which phase to run",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase1-episodes", type=int, default=PHASE1_EPISODES)
    parser.add_argument("--phase2-episodes", type=int, default=PHASE2_EPISODES)
    parser.add_argument("--phase3-episodes", type=int, default=PHASE3_EPISODES)
    parser.add_argument("--strict-checks", action="store_true", help="Enable hard assertion checks from validation section")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    num_patients = max(1, min(int(args.patients), 10))
    patient_names = ALL_ADULTS[:num_patients]

    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    normalizer = StateNormalizer()

    basal_agent: BasalSACAgent | None = None
    bolus_agent: BolusSACAgent | None = None

    if args.phase in {"basal", "all"}:
        basal_agent = train_basal_phase(
            patient_names=patient_names,
            normalizer=normalizer,
            num_episodes=args.phase1_episodes,
            seed=args.seed,
            device=args.device,
            strict_checks=args.strict_checks,
            checkpoint_path="checkpoints/basal_pretrained.pt",
            log_path="logs/train_basal.jsonl",
        )

    if args.phase in {"bolus", "all"}:
        print("Phase 2: Bolus training with frozen basal")
        basal_agent, bolus_agent = train_bolus_phase(
            patient_names=patient_names,
            normalizer=normalizer,
            basal_checkpoint_path="checkpoints/basal_pretrained.pt",
            num_episodes=args.phase2_episodes,
            seed=args.seed + 1000,
            device=args.device,
            strict_checks=args.strict_checks,
            bolus_checkpoint_path="checkpoints/bolus_trained.pt",
            log_path="logs/train_bolus.jsonl",
        )

    if args.phase in {"combined", "all"}:
        print("Phase 3: Combined execution + fine-tuning with frozen planner")
        basal_agent, bolus_agent = run_combined_phase(
            patient_names=patient_names,
            normalizer=normalizer,
            basal_checkpoint_path="checkpoints/basal_pretrained.pt",
            bolus_checkpoint_path="checkpoints/bolus_trained.pt",
            num_episodes=args.phase3_episodes,
            seed=args.seed + 2000,
            device=args.device,
            strict_checks=args.strict_checks,
            log_path="logs/train_combined.jsonl",
        )

    basal_ckpt = Path("checkpoints/basal_pretrained.pt")
    bolus_ckpt = Path("checkpoints/bolus_trained.pt")

    if basal_agent is None:
        if basal_ckpt.exists():
            basal_agent = BasalSACAgent(state_dim=36, device=args.device)
            basal_extra = basal_agent.load(basal_ckpt)
            if isinstance(basal_extra, dict) and isinstance(basal_extra.get("normalizer"), dict):
                normalizer.load_state_dict(basal_extra["normalizer"])

    if bolus_agent is None:
        if bolus_ckpt.exists():
            bolus_agent = BolusSACAgent(state_dim=36, device=args.device)
            bolus_extra = bolus_agent.load(bolus_ckpt)
            if isinstance(bolus_extra, dict) and isinstance(bolus_extra.get("normalizer"), dict):
                normalizer.load_state_dict(bolus_extra["normalizer"])

    if basal_agent is None or bolus_agent is None:
        print("Skipping evaluation: both checkpoints are required (basal_pretrained.pt and bolus_trained.pt).")
        return

    eval_scenario = args.scenario
    run_evaluation(
        patient_names=patient_names,
        normalizer=normalizer,
        basal_agent=basal_agent,
        bolus_agent=bolus_agent,
        scenario_name=eval_scenario,
        seed=args.seed + 3000,
    )

if __name__ == "__main__":
    main()