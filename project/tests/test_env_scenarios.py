from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import default_config
from env.simglucose_env import Action, SimglucoseEnv


def _collect_mock_meals(env: SimglucoseEnv, steps: int) -> list[tuple[int, float]]:
    env.reset()
    meals: list[tuple[int, float]] = []
    for step in range(1, steps + 1):
        out = env.step(Action(basal=0.0, bolus=0.0))
        meal = float(out.info.get("meal", 0.0))
        if meal > 0.0:
            meals.append((step, meal))
    return meals


def test_scenario_a_fixed_meals_in_mock_env() -> None:
    cfg = default_config()
    cfg.env.use_mock_env = True
    cfg.env.episode_days = 1
    cfg.env.step_minutes = 5
    cfg.env.scenario = "A"
    cfg.env.vary_seed_per_episode = False

    env = SimglucoseEnv(cfg.env, training=False)
    meals = _collect_mock_meals(env, steps=288)
    env.close()

    assert [m[0] for m in meals] == [84, 156, 228]
    assert [round(m[1], 2) for m in meals] == [50.0, 75.0, 75.0]


def test_scenario_b_randomized_per_episode_in_mock_env() -> None:
    cfg = default_config()
    cfg.env.use_mock_env = True
    cfg.env.episode_days = 1
    cfg.env.step_minutes = 5
    cfg.env.scenario = "B"
    cfg.env.vary_seed_per_episode = True

    env = SimglucoseEnv(cfg.env, training=True)
    meals_ep1 = _collect_mock_meals(env, steps=288)
    meals_ep2 = _collect_mock_meals(env, steps=288)
    env.close()

    assert meals_ep1 != meals_ep2


def test_scenario_c_reduces_effective_insulin_in_mock_env() -> None:
    cfg_b = default_config()
    cfg_b.env.use_mock_env = True
    cfg_b.env.episode_days = 1
    cfg_b.env.scenario = "B"
    cfg_b.env.vary_seed_per_episode = False

    cfg_c = default_config()
    cfg_c.env.use_mock_env = True
    cfg_c.env.episode_days = 1
    cfg_c.env.scenario = "C"
    cfg_c.env.vary_seed_per_episode = False

    env_b = SimglucoseEnv(cfg_b.env, training=False)
    env_c = SimglucoseEnv(cfg_c.env, training=False)

    env_b.reset()
    env_c.reset()

    action = Action(basal=0.6, bolus=0.8)
    out_b = env_b.step(action)
    out_c = env_c.step(action)

    env_b.close()
    env_c.close()

    glucose_b = SimglucoseEnv.extract_glucose(out_b.observation)
    glucose_c = SimglucoseEnv.extract_glucose(out_c.observation)

    assert glucose_c > glucose_b
    assert out_c.info.get("insulin_effectiveness") == 0.6
