"""Fixture: registers two gym ids with no ``TASK_ID`` — loader must raise."""
from dataclasses import dataclass

import gymnasium as gym


@dataclass
class DummyCfg:
    task_description: str = "multi register fixture"


gym.register(
    id="Private-Multi-A-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:DummyCfg"},
)


gym.register(
    id="Private-Multi-B-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:DummyCfg"},
)
