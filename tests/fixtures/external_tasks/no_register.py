"""Fixture: defines a class but never calls ``gym.register``."""
from dataclasses import dataclass


@dataclass
class DummyCfg:
    task_description: str = "no register fixture"
