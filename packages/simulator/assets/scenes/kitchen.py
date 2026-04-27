from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from leisaac.utils.constant import ASSETS_ROOT

"""Configuration for the Custom Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

KITCHEN_SCENE_USD_PATH = str(SCENES_ROOT / "kitchen" / "scene.usd")

KITCHEN_SCENE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/Environments/Simple_Room/simple_room.usd",
    )
)
