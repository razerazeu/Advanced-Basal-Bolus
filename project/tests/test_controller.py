from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.basal_agent import BasalAgent
from agents.bolus_agent import BolusAgent
from config import default_config
from controller.hybrid_controller import HybridController
from planner.g2p2c_planner import G2P2CPlanner
from training.trainer import Trainer
from utils.state_builder import SlidingWindowStateBuilder


def test_controller_returns_action() -> None:
    cfg = default_config()

    state_builder = SlidingWindowStateBuilder(cfg.state)
    basal_agent = BasalAgent(cfg.state.state_dim, cfg.sac_basal, cfg.basal_agent)
    bolus_agent = BolusAgent(cfg.state.state_dim, cfg.sac_bolus, cfg.bolus_agent)
    planner = G2P2CPlanner(cfg.planner)

    controller = HybridController(
        state_builder=state_builder,
        basal_agent=basal_agent,
        bolus_agent=bolus_agent,
        planner=planner,
        step_minutes=cfg.env.step_minutes,
    )

    controller.reset(start_time=cfg.env.start_time, initial_cgm=120.0)

    action = controller.policy(
        observation={"CGM": 140.0},
        reward=0.0,
        done=False,
        info={"time": cfg.env.start_time, "meal": 25.0},
    )

    assert hasattr(action, "basal")
    assert hasattr(action, "bolus")
    assert float(action.basal) >= 0.0
    assert float(action.bolus) >= 0.0


def test_one_episode_runs_end_to_end(tmp_path: Path) -> None:
    cfg = default_config()
    cfg.env.use_mock_env = True
    cfg.env.episode_days = 1
    cfg.training.basal_pretrain_episodes = 1
    cfg.training.bolus_train_episodes = 1
    cfg.training.max_steps_per_episode = 200
    cfg.training.checkpoint_every = 1
    cfg.training.bolus_train_scenario = "A"
    cfg.training.log_dir = tmp_path
    cfg.sac_basal.warmup_steps = 8
    cfg.sac_bolus.warmup_steps = 8
    cfg.sac_basal.batch_size = 8
    cfg.sac_bolus.batch_size = 8

    trainer = Trainer(cfg)
    summaries = trainer.train_staged()

    assert len(summaries) == 2
    assert summaries[0]["phase"] == "basal_pretrain"
    assert summaries[1]["phase"] == "bolus_train"
    assert summaries[0]["scenario"] == "A"
    assert summaries[1]["scenario"] == "A"
    assert summaries[0]["steps"] > 0
    assert summaries[1]["steps"] > 0
    assert summaries[0]["basal_replay_size"] > 0
    assert summaries[1]["bolus_replay_size"] > 0
