import gymnasium as gym


gym.register(
    id="HCIS-CutleryArrangement-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cutlery_arrangement_env_cfg:CutleryArrangementEnvCfg",
    },
)
