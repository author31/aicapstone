"""Fixture: external task that declares ``TASK_ID`` and calls ``gym.register``."""
from dataclasses import dataclass

import gymnasium as gym


@dataclass
class DummyCfg:
    """Stub env cfg — no IsaacLab dependency."""

    task_description: str = "private fixture task"


TASK_ID = "Private-Test-v0"


gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:DummyCfg"},
)
