"""Fixture: single ``gym.register`` call with no ``TASK_ID`` — loader infers id."""
from dataclasses import dataclass

import gymnasium as gym


@dataclass
class DummyCfg:
    task_description: str = "single register fixture"


gym.register(
    id="Private-Single-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:DummyCfg"},
)
