"""CLI entrypoint for dual-timescale SAC + planner glucose-control simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import DEFAULT_PATIENT_POOL_20, default_config
from training.evaluator import Evaluator
from training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Type-1 diabetes control simulation with dual SAC + planner")
    parser.add_argument("--mode", choices=["train", "pretrain-basal", "train-bolus", "eval"], default="train")
    parser.add_argument("--episodes", type=int, default=None, help="Override bolus-training episodes")
    parser.add_argument("--basal-episodes", type=int, default=None, help="Override basal pretraining episodes")
    parser.add_argument("--bolus-episodes", type=int, default=None, help="Override bolus training episodes")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Episodes per scenario during evaluation")
    parser.add_argument("--use-mock-env", action="store_true", help="Force fallback mock env if Simglucose is unavailable")
    parser.add_argument("--use-20-patients", action="store_true", help="Use optional 20-patient pool extension")
    parser.add_argument(
        "--scenario",
        choices=["A", "B", "C", "all"],
        default="all",
        help="Scenario to evaluate; defaults to all A/B/C",
    )
    parser.add_argument("--basal-checkpoint", type=str, default=None, help="Basal checkpoint path for bolus-only mode")
    parser.add_argument("--disable-planner-eval", action="store_true", help="Disable planner during evaluation")
    parser.add_argument(
        "--fixed-eval-seed",
        action="store_true",
        help="Use identical environment seed on every eval episode (fully deterministic replay)",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint folder for eval mode")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config()

    if args.use_mock_env:
        cfg.env.use_mock_env = True
    if args.use_20_patients:
        cfg.env.patient_names = DEFAULT_PATIENT_POOL_20
    if args.episodes is not None:
        cfg.training.num_episodes = args.episodes
        cfg.training.bolus_train_episodes = args.episodes
    if args.basal_episodes is not None:
        cfg.training.basal_pretrain_episodes = args.basal_episodes
    if args.bolus_episodes is not None:
        cfg.training.bolus_train_episodes = args.bolus_episodes
    if args.eval_episodes is not None:
        cfg.training.eval_episodes_per_scenario = args.eval_episodes
    if args.device is not None:
        cfg.training.device = args.device
    if args.fixed_eval_seed:
        cfg.env.vary_seed_per_episode = False

    if args.mode == "train":
        trainer = Trainer(cfg)
        summaries = trainer.train_staged()
        print(json.dumps(summaries[-1], indent=2))
        print(f"Artifacts saved under: {trainer.output_dir}")
        return

    if args.mode == "pretrain-basal":
        trainer = Trainer(cfg)
        summaries = trainer.pretrain_basal()
        print(json.dumps(summaries[-1], indent=2))
        print(f"Artifacts saved under: {trainer.output_dir}")
        return

    if args.mode == "train-bolus":
        trainer = Trainer(cfg)
        basal_checkpoint = Path(args.basal_checkpoint) if args.basal_checkpoint else None
        summaries = trainer.train_bolus(basal_checkpoint=basal_checkpoint)
        print(json.dumps(summaries[-1], indent=2))
        print(f"Artifacts saved under: {trainer.output_dir}")
        return

    if args.disable_planner_eval:
        cfg.planner.enabled = False

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("outputs")
    evaluator = Evaluator(cfg, checkpoint_dir=checkpoint_dir)
    if args.scenario == "all":
        scenarios = cfg.training.eval_scenarios
    else:
        scenarios = (args.scenario,)
    summaries = evaluator.evaluate(num_episodes=args.eval_episodes, scenarios=scenarios)

    out_path = evaluator.checkpoint_dir / "eval_summary.json"
    evaluator.save_summary(summaries, out_path)

    mean_tir = sum(item["time_in_range"] for item in summaries) / max(len(summaries), 1)
    payload = {
        "episodes": len(summaries),
        "mean_time_in_range": mean_tir,
        "scenarios": list(scenarios),
    }
    print(json.dumps(payload, indent=2))
    print(f"Evaluation summary saved to: {out_path}")


if __name__ == "__main__":
    main()
