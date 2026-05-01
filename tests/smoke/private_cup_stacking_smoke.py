"""Private smoke fixture for AUT-85 — external --task .py rollout test.

Lives outside the aicapstone repo working tree (under /tmp/private_smoke/) to
prove that scripts/rollout.py can resolve and load a Python task module that
is not committed to the repo. Mirrors the existing CupStackingEnvCfg with a
private TASK_ID so it does not collide with the in-tree gym registration.
"""

from __future__ import annotations

import gymnasium as gym

from isaaclab.utils import configclass
from simulator.tasks.cup_stacking.cup_stacking_env_cfg import CupStackingEnvCfg


@configclass
class PrivateCupStackingSmokeEnvCfg(CupStackingEnvCfg):
    """Thin subclass of the in-tree CupStackingEnvCfg with private tweaks."""

    task_description: str = (
        "AUT-85 private smoke: pick the blue cup, place it on the pink cup."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.eye = (0.85, 0.90, 0.70)


TASK_ID = "Private-Smoke-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PrivateCupStackingSmokeEnvCfg",
    },
)
