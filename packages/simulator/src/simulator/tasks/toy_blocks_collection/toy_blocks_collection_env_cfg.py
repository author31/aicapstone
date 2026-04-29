import math

import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.utils import configclass

from leisaac.utils.general_assets import parse_usd_and_create_subassets
from simulator import ASSETS_ROOT
from simulator.utils.object_poses_loader import ObjectPoseConfig
from simulator.assets.scenes.ED305_kitchen import KITCHEN_CFG, KITCHEN_USD_PATH

from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)

LIVING_OBJECTS_ROOT = ASSETS_ROOT / "scenes" / "living_room" / "objects"

TAG_TO_OBJECT: dict[int, str] = {1: "bridge", 2: "cylinder", 3: "triangle", 4: "storage_box"}
ANCHOR_TAG_ID: int = 0
ANCHOR_WORLD_POSE: tuple[float, float, float] = (0.5, -0.2, 0.0)
OBJECT_Z: float = 0.05
OBJECT_ROLL: float = 0.0
OBJECT_PITCH: float = 0.0


@configclass
class ToyBlocksCollectionSceneCfg(SingleArmFrankaTaskSceneCfg):
    """Scene configuration for the toy blocks collection task."""

    scene: AssetBaseCfg = KITCHEN_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    bridge: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/bridge",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Bridge" / "Bridge.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
    )

    cylinder: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/cylinder",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Cylinder" / "Cylinder.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
    )

    triangle: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/triangle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Triangle" / "Triangle.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
    )

    storage_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/storage_box",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LIVING_OBJECTS_ROOT / "Storage_Box" / "storage_box.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
    )


def toys_in_box(
    env,
    bridge_cfg: SceneEntityCfg,
    cylinder_cfg: SceneEntityCfg,
    triangle_cfg: SceneEntityCfg,
    storage_box_cfg: SceneEntityCfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> torch.Tensor:
    """Termination: bridge, cylinder, triangle all within (x,y,z)_range of storage_box."""
    bridge: RigidObject = env.scene[bridge_cfg.name]
    cylinder: RigidObject = env.scene[cylinder_cfg.name]
    triangle: RigidObject = env.scene[triangle_cfg.name]
    storage_box: RigidObject = env.scene[storage_box_cfg.name]

    bridge_pos = bridge.data.root_pos_w - env.scene.env_origins
    cylinder_pos = cylinder.data.root_pos_w - env.scene.env_origins
    triangle_pos = triangle.data.root_pos_w - env.scene.env_origins
    storage_box_pos = storage_box.data.root_pos_w - env.scene.env_origins

    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for toy_pos in (bridge_pos, cylinder_pos, triangle_pos):
        done = torch.logical_and(done, toy_pos[:, 0] < storage_box_pos[:, 0] + x_range[1])
        done = torch.logical_and(done, toy_pos[:, 0] > storage_box_pos[:, 0] + x_range[0])
        done = torch.logical_and(done, toy_pos[:, 1] < storage_box_pos[:, 1] + y_range[1])
        done = torch.logical_and(done, toy_pos[:, 1] > storage_box_pos[:, 1] + y_range[0])
        done = torch.logical_and(done, toy_pos[:, 2] < storage_box_pos[:, 2] + z_range[1])
        done = torch.logical_and(done, toy_pos[:, 2] > storage_box_pos[:, 2] + z_range[0])

    return done


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    """Termination configuration for the toy blocks collection task."""

    success = DoneTerm(
        func=toys_in_box,
        params={
            "bridge_cfg": SceneEntityCfg("bridge"),
            "cylinder_cfg": SceneEntityCfg("cylinder"),
            "triangle_cfg": SceneEntityCfg("triangle"),
            "storage_box_cfg": SceneEntityCfg("storage_box"),
            "x_range": (-0.05, 0.05),
            "y_range": (-0.05, 0.05),
            "z_range": (-0.05, 0.05),
        },
    )


@configclass
class ToyBlocksCollectionEnvCfg(SingleArmFrankaTaskEnvCfg):
    """Configuration for the toy blocks collection task environment."""

    scene: ToyBlocksCollectionSceneCfg = ToyBlocksCollectionSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "pick up the toys and place them into the storage box."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (0.8, 0.87, 0.67)
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (0.35, -0.74, 0.01)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)
        self.scene.robot.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -math.pi / 4.0,
            "panda_joint3": 0.0,
            "panda_joint4": -3.0 * math.pi / 4.0,
            "panda_joint5": 0.0,
            "panda_joint6": math.pi / 2.0,
            "panda_joint7": math.pi / 4.0,
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        }

        parse_usd_and_create_subassets(KITCHEN_USD_PATH, self)

        self.object_pose_cfg = ObjectPoseConfig(
            tag_to_object=TAG_TO_OBJECT,
            anchor_tag_id=ANCHOR_TAG_ID,
            anchor_world_pose=ANCHOR_WORLD_POSE,
            object_z=OBJECT_Z,
            object_roll=OBJECT_ROLL,
            object_pitch=OBJECT_PITCH,
        )
