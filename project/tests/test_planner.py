from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import default_config
from planner.g2p2c_planner import G2P2CPlanner


def test_planner_returns_valid_action() -> None:
    cfg = default_config()
    planner = G2P2CPlanner(cfg.planner)

    basal, bolus, result = planner.refine_action(
        proposed_basal=1.0,
        proposed_bolus=3.0,
        current_glucose=210.0,
        meal_grams=40.0,
    )

    assert cfg.planner.basal_bounds[0] <= basal <= cfg.planner.basal_bounds[1]
    assert cfg.planner.bolus_bounds[0] <= bolus <= cfg.planner.bolus_bounds[1]
    assert isinstance(result.score, float)
    assert len(result.trajectory) == cfg.planner.horizon_steps


def test_planner_hard_safety_zeroes_bolus_when_low() -> None:
    cfg = default_config()
    planner = G2P2CPlanner(cfg.planner)

    basal, bolus, result = planner.refine_action(
        proposed_basal=0.4,
        proposed_bolus=0.8,
        current_glucose=80.0,
        meal_grams=0.0,
        recent_insulin=0.25,
    )

    assert basal == 0.4
    assert bolus == 0.0
    assert result.hard_safety_applied
