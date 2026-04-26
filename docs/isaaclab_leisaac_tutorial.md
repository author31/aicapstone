# Isaac Lab + LeIsaac Configuration Tutorial

A walkthrough of how a task is wired together in this repo, aimed at newcomers who have read the Isaac Lab docs but never opened our `packages/simulator/` tree.

The running example is `LeIsaac-HCIS-CupStacking-SingleArm-v0`. Files referenced:

- `packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py` — reusable base config for any single-arm Franka task.
- `packages/simulator/src/simulator/tasks/cup_stacking/cup_stacking_env_cfg.py` — concrete cup-stacking task built on top of the template.
- `packages/simulator/src/simulator/tasks/cup_stacking/__init__.py` — gym registration.
- `packages/simulator/src/simulator/utils/object_poses_loader.py` — UMI anchor → world pose loader.

---

## 1. Mental model

Isaac Lab uses a **manager-based RL env** (`ManagerBasedRLEnvCfg`). A task is a tree of `@configclass` dataclasses describing:

| Manager | What it owns |
|---------|--------------|
| `scene` | USD assets, robot, sensors, lights (everything you see) |
| `observations` | Per-step obs terms (joints, images, last action, …) |
| `actions` | How `step(action)` maps onto robot joints / gripper |
| `events` | Reset/randomization callbacks |
| `rewards` | Per-step reward terms |
| `terminations` | Done conditions |
| `recorders` | What gets logged into demo episodes |

LeIsaac ([LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac)) sits on top of Isaac Lab and adds:

- Robot & scene asset configs (`leisaac.assets.*`).
- Teleop devices and an `action_process` helper.
- A LeRobot dataset handler that streams demos straight into a `lerobot` HF dataset.
- Convenience helpers like `parse_usd_and_create_subassets` to lift sub-prims out of a complex USD scene into Isaac Lab `RigidObjectCfg` entries.

Our `simulator/` package is a thin layer on top of LeIsaac that hosts project-specific task configs.

---

## 2. The single-arm Franka template

`single_arm_franka_cfg.py` is the base every Franka task should subclass. It pre-wires sensors, observations, the teleop pipeline, and the LeRobot frame builder, so a new task only has to specify scene assets and termination logic.

### 2.1 Scene (`SingleArmFrankaTaskSceneCfg`)

```python
class SingleArmFrankaTaskSceneCfg(InteractiveSceneCfg):
    scene: AssetBaseCfg = MISSING                         # subclass fills in
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = ...                   # end-effector + finger frames
    wrist:    TiledCameraCfg     = ...                    # 640×480 RGB on panda_hand
    front:    TiledCameraCfg     = ...                    # 640×480 RGB front view
    light = AssetBaseCfg(spawn=sim_utils.DomeLightCfg(...))
```

Key idioms:

- `{ENV_REGEX_NS}` expands per parallel env. Anything bound under `/World/...` (e.g. `front` camera) is **shared** across envs — useful for fixed external cameras, dangerous if you wanted per-env randomization.
- `ee_frame` declares three named frames so MDP terms can reference `panda_hand`, `tool_leftfinger`, `tool_rightfinger` without hardcoding prim paths.
- Cameras are `TiledCameraCfg`, which renders all envs in a single batched draw.

### 2.2 Observations (`SingleArmFrankaObservationsCfg`)

A single `policy` group exposes:

- `joint_pos`, `joint_vel`, `joint_pos_rel`, `joint_vel_rel`
- `actions` (last action returned by the env)
- `wrist`, `front` RGB images (raw uint8, `normalize=False`)
- `joint_pos_target` (most recent target sent to the controller — useful when training on action-following datasets)

`enable_corruption = True` lets `mdp` corruption terms attach noise. `concatenate_terms = False` keeps each term as its own dict entry, which is what LeRobot expects.

### 2.3 Actions

`SingleArmFrankaActionsCfg` declares `arm_action` and `gripper_action` as `MISSING`. They are filled in at runtime by `use_teleop_device`:

```python
self.actions.arm_action = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    scale=1.0,
    use_default_offset=False,
)
self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_finger_joint.*"],
    open_command_expr={"panda_finger_joint.*": 0.04},
    close_command_expr={"panda_finger_joint.*": 0.0},
)
self.scene.robot.spawn.rigid_props.disable_gravity = True
```

Gravity is disabled on the robot during keyboard/gamepad teleop so the arm holds its pose between commands. Only `keyboard` and `gamepad` teleop devices are supported on Franka — VR/leader-arm flows raise `ValueError`.

### 2.4 Events, rewards, terminations

- `events.reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")` — on every `reset()`, snap everything back to its `init_state`.
- `rewards = SingleArmFrankaRewardsCfg()` — empty by default; collect demos, no shaped reward.
- `terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)` — episode caps at `episode_length_s` (15 s by default).

### 2.5 LeRobot frame builder

`build_lerobot_frame` is the contract between Isaac Lab episode data and a LeRobot dataset row. Each call returns:

```python
{
    "action":             <last applied action>,
    "observation.state":  <last joint_pos>,
    "task":               <self.task_description>,
    "observation.images.<camera>": <rgb frame>,    # one per camera in dataset_cfg
}
```

The image keys are driven by `dataset_cfg.features` — keep `features` in sync with the cameras you declared in the scene, otherwise the recorder silently drops frames.

### 2.6 `__post_init__` defaults

```python
self.decimation = 1
self.episode_length_s = 15
self.viewer.eye = (1.4, -0.9, 1.2)
self.viewer.lookat = (2.0, -0.5, 1.0)
self.sim.physx.bounce_threshold_velocity = 0.01
self.sim.physx.friction_correlation_distance = 0.00625
self.sim.render.enable_translucency = True
self.default_feature_joint_names = [f"{j}.pos" for j in FRANKA_JOINT_NAMES]
```

`default_feature_joint_names` becomes the column ordering in LeRobot state vectors. Override `decimation` and `episode_length_s` on a per-task basis as needed.

---

## 3. The cup-stacking task

`cup_stacking_env_cfg.py` shows the minimum a new task has to provide.

### 3.1 Scene subclass

```python
class CupStackingSceneCfg(SingleArmFrankaTaskSceneCfg):
    scene: AssetBaseCfg = KITCHEN_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    blue_cup: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/blue_cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(KITCHEN_OBJECTS_ROOT / "BlueCup" / "BlueCup.usd"),
            mass_props=MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.36, -0.4, 0.12), rot=(1, 0, 0, 0)),
    )
    pink_cup: RigidObjectCfg = ...
```

- `KITCHEN_CFG` comes from `leisaac.assets.scenes.ED305_kitchen` and brings in the kitchen USD plus its physics defaults.
- Cup USDs live under `packages/simulator/assets/scenes/kitchen/objects/`. The path is built from `simulator.ASSET_ROOT`, which resolves to `packages/simulator/assets/` regardless of CWD.
- Quaternion convention is **(w, x, y, z)** everywhere in Isaac Lab — `(1, 0, 0, 0)` is identity.

### 3.2 Termination

```python
def blue_cup_on_top_pink_cup(env, blue_cup_cfg, pink_cup_cfg, x_range, y_range, height_threshold):
    blue = env.scene[blue_cup_cfg.name].data.root_pos_w - env.scene.env_origins
    pink = env.scene[pink_cup_cfg.name].data.root_pos_w - env.scene.env_origins
    done  = (blue[:, 0] > pink[:, 0] + x_range[0]) & (blue[:, 0] < pink[:, 0] + x_range[1])
    done &= (blue[:, 1] > pink[:, 1] + y_range[0]) & (blue[:, 1] < pink[:, 1] + y_range[1])
    done &= (blue[:, 2] > pink[:, 2] + height_threshold)
    return done
```

Notes:

- Always subtract `env.scene.env_origins` before comparing positions. Multi-env worlds are tiled in `x`/`y`, so raw `root_pos_w` is offset per env.
- Termination functions must return a boolean tensor of shape `(num_envs,)`. Combine via `&`/`|`, not Python `and`/`or`.
- `DoneTerm` takes the function plus a `params` dict. The dict is forwarded as keyword args, so `SceneEntityCfg("blue_cup")` resolves at runtime to the blue cup `RigidObject`.

### 3.3 Env subclass

```python
class CupStackingEnvCfg(SingleArmFrankaTaskEnvCfg):
    scene:        CupStackingSceneCfg               = CupStackingSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg    = SingleArmFrankaObservationsCfg()
    terminations: TerminationsCfg                   = TerminationsCfg()
    task_description: str = "pick up the blue cup and place it on the pink cup."

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.eye = (0.8, 0.87, 0.67)
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (0.35, -0.74, 0.01)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)
        self.scene.robot.init_state.joint_pos = { ... }     # home pose

        parse_usd_and_create_subassets(KITCHEN_USD_PATH, self)

        if self.object_poses_path is not None:
            ...
```

Things worth pointing out:

- `env_spacing=8.0` is the tile size between parallel envs. Pick something larger than the bounding radius of your scene.
- `task_description` is what gets baked into every LeRobot frame's `"task"` field — it is also the natural-language conditioning string for VLA policies. Keep it short and consistent across episodes.
- `parse_usd_and_create_subassets(KITCHEN_USD_PATH, self)` walks the kitchen USD and registers every `Xform` it finds as a Isaac Lab sub-asset. This is what lets you reference `{ENV_REGEX_NS}/Scene/blue_cup` even though the prim is buried inside the kitchen USD.
- `dynamic_reset_gripper_effort_limit = False` keeps the gripper effort constant across resets — needed so collected demos don't drift if the controller silently re-tunes itself between episodes.

### 3.4 Anchor-relative object poses (UMI integration)

The block at the bottom of `__post_init__` connects this env to UMI's per-session `object_poses.json`:

```python
TAG_TO_OBJECT     = {1: "blue_cup", 2: "pink_cup"}
ANCHOR_TAG_ID     = 0
ANCHOR_WORLD_POSE = (0.0, 0.0, 0.0)     # (x, y, yaw_rad) in world frame
OBJECT_Z          = 0.12

if self.object_poses_path is not None:
    pose_cfg = ObjectPoseConfig(
        tag_to_object=TAG_TO_OBJECT,
        anchor_tag_id=ANCHOR_TAG_ID,
        anchor_world_pose=ANCHOR_WORLD_POSE,
        object_z=OBJECT_Z,
        object_roll=OBJECT_ROLL,
        object_pitch=OBJECT_PITCH,
    )
    loaded_poses = load_object_poses(self.object_poses_path, pose_cfg)
    for obj_name, (pos, rot) in loaded_poses.items():
        getattr(self.scene, obj_name).init_state.pos = pos
        getattr(self.scene, obj_name).init_state.rot = rot
```

How it works:

1. UMI captures a scene with an ArUco anchor tag (`anchor_tag_id`) plus per-object tags. It writes `object_poses.json` with `(x, y, yaw)` for each tag in the **anchor frame**.
2. `ObjectPoseConfig` records, per task: where the anchor sits in Isaac Lab's world (`anchor_world_pose`), the fixed `z` to drop everything onto, and the roll/pitch (most tabletop scenes leave both at `0`).
3. `load_object_poses` applies the SE(2) transform `world = anchor_world ⊕ anchor_frame_pose` and returns `(pos_xyz, quat_wxyz)` ready to drop into a `RigidObjectCfg.InitialStateCfg`.
4. Validation is strict: missing tags, duplicate tags, an anchor mismatch, or an unmapped tag all raise `ObjectPosesError` instead of silently using stale defaults.

To wire a new task into this loader, define `TAG_TO_OBJECT`, `ANCHOR_TAG_ID`, `ANCHOR_WORLD_POSE`, and `OBJECT_Z` at module scope and copy the `if self.object_poses_path is not None:` block. `object_poses_path` is set on the env from the CLI (datagen/teleop scripts forward `--object-poses-path`).

---

## 4. Gym registration

```python
# packages/simulator/src/simulator/tasks/cup_stacking/__init__.py
gym.register(
    id="LeIsaac-HCIS-CupStacking-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.cup_stacking_env_cfg:CupStackingEnvCfg"},
)
```

- `entry_point` is **always** `isaaclab.envs:ManagerBasedRLEnv` for this style of task — only the `env_cfg_entry_point` changes.
- The id pattern is `LeIsaac-HCIS-<TaskName>-<RobotKind>-v<N>`. Bump `vN` whenever you make a backwards-incompatible config change so old demo datasets stay associated with the right env.

---

## 5. Recipe: add a new single-arm Franka task

1. Create `packages/simulator/src/simulator/tasks/<my_task>/`.
2. `__init__.py`: copy the `gym.register(...)` block; point `env_cfg_entry_point` at your new module.
3. `<my_task>_env_cfg.py`:
   - Subclass `SingleArmFrankaTaskSceneCfg`; override `scene` and add per-object `RigidObjectCfg` fields.
   - Subclass `SingleArmFrankaTerminationsCfg`; add a `success` `DoneTerm` with your own predicate (return shape `(num_envs,)`, subtract `env_origins`).
   - Subclass `SingleArmFrankaTaskEnvCfg`; in `__post_init__` set viewer pose, robot home pose, `task_description`, and call `parse_usd_and_create_subassets` if you reuse a packaged USD scene.
   - Optional: declare `TAG_TO_OBJECT` / `ANCHOR_*` constants and the `load_object_poses` block to honor UMI anchor data.
4. Register the task module under `simulator.tasks` (auto-discovered when the package is imported).
5. Smoke test:

   ```bash
   uv run python scripts/environments/teleoperation/teleop_se3_agent.py \
       --task=LeIsaac-HCIS-<MyTask>-SingleArm-v0 \
       --teleop_device=keyboard \
       --num_envs=1 \
       --device=cuda \
       --enable_cameras
   ```

   If the env spawns and the cameras show frames, you're ready to record demos.

---

## 6. Common pitfalls

- **Prim path under the wrong namespace.** Per-env assets need `{ENV_REGEX_NS}/...`, not `/World/...`. World-fixed assets need `/World/...`. Mixing them up is the #1 cause of "I added an object but only env 0 sees it."
- **Forgot to subtract `env.scene.env_origins`.** Termination/reward predicates that pass on env 0 and silently fail on env ≥ 1 are almost always this bug.
- **Quaternion order.** Isaac Lab uses `(w, x, y, z)`. UMI/SciPy/ROS often use `(x, y, z, w)`. The pose loader emits the right order — don't second-guess it.
- **LeRobot dataset feature drift.** If you add a camera to the scene but forget to add `observation.images.<name>` to `LeRobotDatasetCfg.features`, recorded demos will be missing that camera with no error.
- **Editing `dependencies/IsaacLab/` directly.** That's a submodule; commits there don't end up in this repo. Put project-specific overrides in `packages/simulator/`.

---

## 7. Where to read next

- `packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py` — the file you'll subclass most often.
- `packages/simulator/src/simulator/utils/object_poses_loader.py` — the docstrings spell out the JSON schema and validation rules.
- [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac) — upstream LeIsaac for the asset configs, teleop devices, and the LeRobot dataset handler.
- `dependencies/IsaacLab/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/manager_based_env_cfg.py` — the manager contract you're conforming to.
