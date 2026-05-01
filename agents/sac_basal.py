from __future__ import annotations

from agents.sac_base import SACAgent, SACConfig

BASAL_GAMMA = 0.99


class BasalSACAgent(SACAgent):
    def __init__(self, state_dim: int, device: str = "cpu") -> None:
        cfg = SACConfig(
            state_dim=state_dim,
            action_dim=1,
            action_low=0.0,
            action_high=0.05,
            gamma=BASAL_GAMMA,
        )
        super().__init__(config=cfg, device=device)
