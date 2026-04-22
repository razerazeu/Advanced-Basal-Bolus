from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import default_config
from utils.reward import basal_episode_reward, bolus_episode_reward, planner_risk_reward
from utils.state_builder import SlidingWindowStateBuilder


def test_state_builder_shape_and_preview() -> None:
    cfg = default_config()
    cfg.state.window_size = 6
    cfg.state.include_meal = True

    builder = SlidingWindowStateBuilder(cfg.state)
    builder.reset(initial_cgm=120.0)

    for i in range(10):
        builder.update(cgm=110.0 + i, insulin=0.2 * i, meal=5.0 if i % 3 == 0 else 0.0)

    state = builder.build_state()
    preview = builder.preview_state(cgm=145.0, insulin=0.5, meal=20.0)

    expected_dim = cfg.state.window_size * 3
    assert state.shape == (expected_dim,)
    assert preview.shape == (expected_dim,)
    assert np.isfinite(state).all()


def test_reward_functions_do_not_crash() -> None:
    glucose = [60.0, 95.0, 130.0, 210.0]
    meals = [0.0, 30.0, 0.0, 0.0]
    bolus = [0.0, 2.5, 0.0, 1.0]

    r_basal = basal_episode_reward(glucose)
    r_bolus = bolus_episode_reward(glucose, meals, bolus)
    r_plan = planner_risk_reward(80.0)

    assert isinstance(r_basal, float)
    assert isinstance(r_bolus, float)
    assert isinstance(r_plan, float)
