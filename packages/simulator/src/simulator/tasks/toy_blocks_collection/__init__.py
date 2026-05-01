import gymnasium as gym


gym.register(
    id="HCIS-ToyBlocksCollection-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.toy_blocks_collection_env_cfg:ToyBlocksCollectionEnvCfg",
    },
)
