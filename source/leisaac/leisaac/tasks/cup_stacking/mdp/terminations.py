from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def task_done(
    env: ManagerBasedRLEnv | DirectRLEnv,
    blue_cup_cfg: SceneEntityCfg,
    pink_cup_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_threshold: float = 0.10,
) -> torch.Tensor:
    """Determine if the cup stacking task is complete.

    This function checks whether all success conditions for the task have been met:
    1. the blue cup is within the target x/y range of the pink cup
    2. the blue cup is above the pink cup by at least ``height_threshold``

    Args:
        env: The RL environment instance.
        blue_cup_cfg: Configuration for the blue cup entity.
        pink_cup_cfg: Configuration for the pink cup entity.
        x_range: Range of x positions relative to the pink cup for task completion.
        y_range: Range of y positions relative to the pink cup for task completion.
        height_threshold: Minimum z offset from the pink cup to the blue cup.
    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    blue_cup: RigidObject = env.scene[blue_cup_cfg.name]
    pink_cup: RigidObject = env.scene[pink_cup_cfg.name]

    blue_cup_pos = blue_cup.data.root_pos_w - env.scene.env_origins
    pink_cup_pos = pink_cup.data.root_pos_w - env.scene.env_origins

    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    done = torch.logical_and(done, blue_cup_pos[:, 0] < pink_cup_pos[:, 0] + x_range[1])
    done = torch.logical_and(done, blue_cup_pos[:, 0] > pink_cup_pos[:, 0] + x_range[0])
    done = torch.logical_and(done, blue_cup_pos[:, 1] < pink_cup_pos[:, 1] + y_range[1])
    done = torch.logical_and(done, blue_cup_pos[:, 1] > pink_cup_pos[:, 1] + y_range[0])
    done = torch.logical_and(done, blue_cup_pos[:, 2] > pink_cup_pos[:, 2] + height_threshold)
    return done
