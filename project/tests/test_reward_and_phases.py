from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import default_config
from training.trainer import Trainer
from utils.reward import (
    BASAL_REWARD_SCALE,
    basal_episode_reward,
    bolus_action_reward,
    bolus_glucose_reward,
    bolus_step_reward,
)


def test_basal_episode_reward_uses_nested_exponentiated_band_counts() -> None:
    glucose = [110.0, 90.0, 200.0, 130.0]

    expected = 0.0
    for idx in range(len(glucose)):
        window = glucose[: idx + 1]
        n = len(window)
        c_105_115 = sum(1 for g in window if 105.0 < g < 115.0)
        c_100_120 = sum(1 for g in window if 100.0 < g < 120.0)
        c_70_180 = sum(1 for g in window if 70.0 < g < 180.0)

        expected += (
            math.exp((BASAL_REWARD_SCALE * (c_105_115 / n)) / 2.0)
            + math.exp((BASAL_REWARD_SCALE * (c_100_120 / n)) / 2.0)
            + math.exp((BASAL_REWARD_SCALE * (c_70_180 / n)) / 2.0)
        )

    assert basal_episode_reward(glucose) == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_bolus_glucose_reward_matches_target_formula() -> None:
    assert bolus_glucose_reward(125.0) == pytest.approx(0.1, rel=1e-6)
    assert bolus_glucose_reward(39.0) == pytest.approx(-0.86, rel=1e-6)
    assert bolus_glucose_reward(200.0) == pytest.approx(-0.75, rel=1e-6)


def test_bolus_action_reward_matches_timing_logic() -> None:
    assert bolus_action_reward(50.0, 0.2, meal_window_active=True, recent_meal_grams=50.0) == 10.0
    assert bolus_action_reward(0.0, 0.0, meal_window_active=False, recent_meal_grams=0.0) == 0.0
    assert bolus_action_reward(60.0, 0.0, meal_window_active=True, recent_meal_grams=60.0) == -2.0
    assert bolus_action_reward(0.0, 0.3, meal_window_active=False, recent_meal_grams=0.0) == -2.0


def test_bolus_step_reward_is_glucose_plus_action_terms() -> None:
    # 0.1 + 10.0 when in-range glucose and meal+action coincide.
    assert bolus_step_reward(125.0, 75.0, 0.2, meal_window_active=True, recent_meal_grams=75.0) == pytest.approx(
        10.1,
        rel=1e-6,
    )


def test_standalone_bolus_training_requires_pretrained_basal_checkpoint(tmp_path: Path) -> None:
    cfg = default_config()
    cfg.env.use_mock_env = True
    cfg.training.log_dir = tmp_path
    cfg.training.bolus_train_episodes = 1

    trainer = Trainer(cfg)
    with pytest.raises(ValueError):
        trainer.train_bolus(basal_checkpoint=None)
