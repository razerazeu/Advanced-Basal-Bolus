from __future__ import annotations

from agents.sac_base import SACAgent, SACConfig

BOLUS_GAMMA = 0.95


class BolusSACAgent(SACAgent):
    def __init__(self, state_dim: int, device: str = "cpu") -> None:
        cfg = SACConfig(
            state_dim=state_dim,
            action_dim=1,
            action_low=0.0,
            action_high=0.5,
            gamma=BOLUS_GAMMA,
        )
        super().__init__(config=cfg, device=device)
        self.initialize_actor_output_bias(0.0, log_std_bias=-3.0)
