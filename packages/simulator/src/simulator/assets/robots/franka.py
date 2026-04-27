from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


from leisaac.utils.constant import ASSETS_ROOT

FRANKA_ASSET_PATH = Path(ASSETS_ROOT) / "robots" / "franka.usd"

# FRANKA_PANDA_CFG = FRANKA_PANDA_CFG.replace(
#     spawn=FRANKA_PANDA_CFG.spawn.replace(
#         usd_path=str(FRANKA_ASSET_PATH),
#     )
# )
