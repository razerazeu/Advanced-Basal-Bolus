"""Central configuration objects for the hybrid basal-bolus research prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


ScenarioName = Literal["A", "B", "C"]


DEFAULT_ADULT_POOL_10: tuple[str, ...] = tuple([f"adult#{idx:03d}" for idx in range(1, 11)])
DEFAULT_PATIENT_POOL_20: tuple[str, ...] = tuple(
    [f"adolescent#{idx:03d}" for idx in range(1, 11)] + [f"adult#{idx:03d}" for idx in range(1, 11)]
)


@dataclass
class EnvConfig:
    """Settings for the Simglucose (or mock fallback) environment."""

    patient_name: str = "adult#001"
    patient_names: tuple[str, ...] = DEFAULT_ADULT_POOL_10
    scenario: ScenarioName = "A"
    scenario_a_meal_times_min: tuple[int, int, int] = (7 * 60, 13 * 60, 19 * 60)
    scenario_a_meal_carbs: tuple[float, float, float] = (50.0, 75.0, 75.0)
    scenario_b_time_jitter_minutes: int = 90
    scenario_b_carb_noise_ratio: float = 0.20
    scenario_b_min_meal_carbs: float = 20.0
    scenario_b_max_meal_carbs: float = 120.0
    scenario_c_insulin_resistance: float = 0.40
    sensor_name: str = "Dexcom"
    pump_name: str = "Insulet"
    use_mock_env: bool = False
    vary_seed_per_episode: bool = True
    step_minutes: int = 5
    episode_days: int = 2
    seed: int = 7
    start_time: datetime = datetime(2026, 1, 1, 0, 0)

    def insulin_effectiveness(self, scenario: ScenarioName | None = None) -> float:
        """Return insulin effectiveness scale for the active scenario."""

        active = scenario or self.scenario
        if active == "C":
            return float(max(0.0, 1.0 - self.scenario_c_insulin_resistance))
        return 1.0


@dataclass
class SACConfig:
    """Hyperparameters for SAC continuous-control training."""

    hidden_dim: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy: float = -1.0
    batch_size: int = 128
    replay_size: int = 100_000
    warmup_steps: int = 128
    updates_per_step: int = 1


@dataclass
class BasalAgentConfig:
    """Basal-agent timing and insulin-mapping parameters."""

    decision_hour: int = 0
    decision_minute: int = 0
    min_basal: float = 0.0
    max_basal: float = 0.6
    i_max: float = 2.0
    eta: float = 1.5
    default_basal: float = 0.08


@dataclass
class BolusAgentConfig:
    """Bolus-agent timing and insulin-mapping parameters."""

    decision_interval_minutes: int = 15
    meal_threshold_grams: float = 10.0
    correction_glucose_threshold: float = 185.0
    max_bolus_no_meal: float = 0.10
    min_bolus: float = 0.0
    max_bolus: float = 1.0
    meal_bolus_ratio_grams_per_unit: float = 12.0
    low_glucose_bolus_threshold: float = 140.0
    low_glucose_bolus_scale: float = 0.2
    delivery_window_minutes: float = 15.0
    i_max: float = 10.0
    eta: float = 1.5


@dataclass
class PlannerConfig:
    """G2P2C-inspired predictive safety-layer parameters."""

    enabled: bool = True
    mode: Literal["hard", "soft"] = "hard"
    horizon_steps: int = 6
    candidate_scales: tuple[float, ...] = (0.8, 1.0, 1.2)
    bolus_offsets: tuple[float, ...] = (-0.15, -0.05, 0.0)
    basal_bounds: tuple[float, float] = (0.0, 0.6)
    bolus_bounds: tuple[float, float] = (0.0, 1.0)
    soft_adjustment_gain: float = 0.35
    low_glucose_threshold: float = 115.0
    severe_hypoglycemia_threshold: float = 85.0
    predicted_safety_margin: float = 15.0
    emergency_basal_scale: float = 0.05
    emergency_bolus_max: float = 0.0
    hypoglycemia_penalty: float = 60.0
    severe_hypoglycemia_penalty: float = 180.0

    # Lightweight surrogate-dynamics coefficients.
    target_glucose: float = 115.0
    carb_absorption_rate: float = 0.35
    carb_sensitivity: float = 0.8
    insulin_sensitivity: float = 24.0
    homeostatic_drift: float = 0.03
    iob_decay: float = 0.75
    iob_weight: float = 0.65
    iob_safety_threshold: float = 0.03


@dataclass
class StateConfig:
    """Sliding-window state representation settings."""

    window_size: int = 12
    include_meal: bool = True
    glucose_center: float = 110.0
    glucose_scale: float = 50.0
    insulin_scale: float = 10.0
    meal_scale: float = 100.0

    @property
    def state_dim(self) -> int:
        channels = 3 if self.include_meal else 2
        return self.window_size * channels


@dataclass
class TrainingConfig:
    """Top-level loop and logging settings."""

    # Legacy compatibility. If CLI overrides --episodes, this maps to bolus_train_episodes.
    num_episodes: int = 12
    basal_pretrain_episodes: int = 8
    bolus_train_episodes: int = 12
    max_steps_per_episode: int = 576
    eval_episodes_per_scenario: int = 10
    checkpoint_every: int = 5
    log_dir: Path = Path("outputs")
    run_name: str = "dual_sac_g2p2c"
    device: str = "cpu"
    basal_pretrain_scenario: ScenarioName = "A"
    bolus_train_scenario: ScenarioName = "A"
    eval_scenarios: tuple[ScenarioName, ...] = ("A", "B", "C")
    basal_fasting_end_hour: int = 7
    basal_fasting_end_minute: int = 0
    freeze_basal_during_bolus: bool = True
    use_planner_during_training: bool = False
    bolus_meal_timing_window_minutes: int = 30


@dataclass
class ProjectConfig:
    """All configuration for the complete prototype."""

    env: EnvConfig = field(default_factory=EnvConfig)
    sac_basal: SACConfig = field(default_factory=SACConfig)
    sac_bolus: SACConfig = field(default_factory=SACConfig)
    basal_agent: BasalAgentConfig = field(default_factory=BasalAgentConfig)
    bolus_agent: BolusAgentConfig = field(default_factory=BolusAgentConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    state: StateConfig = field(default_factory=StateConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 7


def default_config() -> ProjectConfig:
    """Build a default project configuration object."""

    return ProjectConfig()
