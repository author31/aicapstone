# Defining a Cup-Stacking State Machine

This tutorial explains how to define a scripted state machine for the cup-stacking task in:

```text
source/leisaac/leisaac/datagen/state_machine/cup_stacking.py
```

The goal is to generate demonstrations for the task:

```text
Pick up the blue cup and place it on the pink cup.
```

The cup-stacking environment is defined in:

```text
source/leisaac/leisaac/tasks/cup_stacking/cup_stacking_env_cfg.py
```

The task primarily uses the Franka Panda robot through:

```text
source/leisaac/leisaac/tasks/template/single_arm_franka_cfg.py
```

## 1. What a State Machine Does

A state machine is a scripted policy. It does not learn. It generates actions step by step so the robot can complete a task and record demonstrations.

For cup stacking, the scripted policy usually follows this sequence:

1. Move the Franka gripper above the blue cup.
2. Lower to the grasp pose.
3. Close the gripper.
4. Lift the blue cup.
5. Move above the pink cup.
6. Lower to the stacking pose.
7. Open the gripper.
8. Lift away or return home.
9. Check whether the blue cup is on top of the pink cup.

Every LeIsaac state machine should implement the interface from:

```text
source/leisaac/leisaac/datagen/state_machine/base.py
```

The required methods are:

```python
def setup(self, env) -> None:
    ...

def check_success(self, env) -> bool:
    ...

def get_action(self, env) -> torch.Tensor:
    ...

def advance(self) -> None:
    ...

def reset(self) -> None:
    ...

@property
def is_episode_done(self) -> bool:
    ...
```

The optional hook is:

```python
def pre_step(self, env) -> None:
    ...
```

Use `pre_step()` only when something must happen immediately before the environment step, such as directly blending joints back to a rest pose.

## 2. Understand the Franka Action Format

The cup-stacking task uses `SingleArmFrankaTaskEnvCfg`.

For Franka keyboard/gamepad style control, the action tensor is 8D:

```text
[dx, dy, dz, droll, dpitch, dyaw, d_panda_joint1, gripper]
```

That means `get_action()` must return:

```python
torch.Tensor  # shape: (env.num_envs, 8)
```

The current cup-stacking state machine builds this format with:

```python
return torch.cat([delta_pos_root, delta_rot_root, base_joint_delta, gripper_cmd], dim=-1)
```

For the Franka gripper:

```python
_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
```

Positive means open. Non-positive means close.

## 3. Define Task Constants

Start by defining object names, end-effector names, timing, and pose offsets.

Example:

```python
_BLUE_CUP_NAME = "blue_cup"
_PINK_CUP_NAME = "pink_cup"
_EE_BODY_NAME = "panda_hand"

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0

_APPROACH_STEPS = 120
_HOME_START_STEP = 760
_MAX_CARTESIAN_DELTA = 0.018

_HOVER_Z_OFFSET = 0.28
_GRASP_Z_OFFSET = 0.16
_LIFT_Z_OFFSET = 0.34
_STACK_HEIGHT = 0.12
_PLACE_Z_OFFSET = _GRASP_Z_OFFSET + _STACK_HEIGHT
_RELEASE_LIFT_Z_OFFSET = 0.30

_GRASP_XY_OFFSET = (-0.015, 0.0)
_PLACE_XY_OFFSET = (0.0, 0.0)
```

These constants are the main tuning knobs.

Tune these first if the behavior is wrong:

```python
_GRASP_XY_OFFSET
_PLACE_XY_OFFSET
_GRASP_Z_OFFSET
_PLACE_Z_OFFSET
_MAX_CARTESIAN_DELTA
```

## 4. Define the Franka Rest Pose

Use the Franka joint names from `single_arm_franka_cfg.py`:

```python
_FRANKA_REST_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}
```

This rest pose is useful for setup and for returning the robot to a clean final state.

## 5. Implement `setup()`

`setup()` runs once after the environment is created.

Use it to:

1. Find the `panda_hand` body index.
2. Build the Franka rest joint tensor.
3. Move the robot to the rest pose.
4. Record the rest end-effector position.

Example:

```python
def setup(self, env) -> None:
    robot = env.scene["robot"]
    self._ee_body_idx = _find_body_index(robot, _EE_BODY_NAME)

    joint_names = list(robot.data.joint_names)
    self._rest_joint_pos = robot.data.joint_pos.clone()
    for idx, name in enumerate(joint_names):
        if name in _FRANKA_REST_JOINT_POS:
            self._rest_joint_pos[:, idx] = _FRANKA_REST_JOINT_POS[name]

    robot.write_joint_state_to_sim(
        position=self._rest_joint_pos,
        velocity=torch.zeros_like(self._rest_joint_pos),
    )
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    self._rest_ee_pos_w = self._ee_pos_w(robot).clone()
```

The helper to find a body index can look like this:

```python
def _find_body_index(robot, body_name: str) -> int:
    if hasattr(robot, "find_bodies"):
        body_ids, _ = robot.find_bodies(body_name)
        if len(body_ids) > 0:
            return int(body_ids[0])

    body_names = getattr(robot.data, "body_names", None)
    if body_names is not None and body_name in body_names:
        return body_names.index(body_name)

    return -1
```

## 6. Implement Success Checking

Do not treat reaching the last state-machine step as success.

Success should be based on object positions:

```text
blue cup x/y is close to pink cup x/y
blue cup z is above pink cup z
```

Example:

```python
def check_success(self, env) -> bool:
    blue_cup_pos = env.scene[_BLUE_CUP_NAME].data.root_pos_w - env.scene.env_origins
    pink_cup_pos = env.scene[_PINK_CUP_NAME].data.root_pos_w - env.scene.env_origins

    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    done = torch.logical_and(done, blue_cup_pos[:, 0] < pink_cup_pos[:, 0] + 0.05)
    done = torch.logical_and(done, blue_cup_pos[:, 0] > pink_cup_pos[:, 0] - 0.05)
    done = torch.logical_and(done, blue_cup_pos[:, 1] < pink_cup_pos[:, 1] + 0.05)
    done = torch.logical_and(done, blue_cup_pos[:, 1] > pink_cup_pos[:, 1] - 0.05)
    done = torch.logical_and(done, blue_cup_pos[:, 2] > pink_cup_pos[:, 2] + 0.10)
    return bool(done.all().item())
```

If your cups have different dimensions or root prim locations, tune the height threshold.

## 7. Implement the Phase Timeline

The main logic lives in `get_action()`.

Use the step count to choose the current phase:

```python
def get_action(self, env) -> torch.Tensor:
    robot = env.scene["robot"]
    robot.write_joint_damping_to_sim(damping=10.0)

    device = env.device
    num_envs = env.num_envs
    step = self._step_count

    blue_cup_pos_w = env.scene[_BLUE_CUP_NAME].data.root_pos_w.clone()
    pink_cup_pos_w = env.scene[_PINK_CUP_NAME].data.root_pos_w.clone()

    if step == 0:
        self._initial_ee_pos_w = self._ee_pos_w(robot).clone()

    if step < _APPROACH_STEPS:
        target_pos_w, gripper_cmd = self._phase_approach_blue(blue_cup_pos_w, num_envs, device)
    elif step < 200:
        target_pos_w, gripper_cmd = self._phase_move_above_blue(blue_cup_pos_w, num_envs, device)
    elif step < 300:
        target_pos_w, gripper_cmd = self._phase_lower_to_blue(blue_cup_pos_w, num_envs, device)
    elif step < 380:
        target_pos_w, gripper_cmd = self._phase_grasp_blue(blue_cup_pos_w, num_envs, device)
    elif step < 500:
        target_pos_w, gripper_cmd = self._phase_lift_blue(blue_cup_pos_w, num_envs, device)
    elif step < 620:
        target_pos_w, gripper_cmd = self._phase_move_above_pink(pink_cup_pos_w, num_envs, device)
    elif step < 700:
        target_pos_w, gripper_cmd = self._phase_lower_to_stack(pink_cup_pos_w, num_envs, device)
    elif step < 740:
        target_pos_w, gripper_cmd = self._phase_release_blue(pink_cup_pos_w, num_envs, device)
    elif step < _HOME_START_STEP:
        target_pos_w, gripper_cmd = self._phase_lift_from_stack(pink_cup_pos_w, num_envs, device)
    else:
        target_pos_w, gripper_cmd = self._phase_return_home(num_envs, device)

    return self._relative_franka_action(env, target_pos_w, gripper_cmd)
```

This phase table is the easiest part to customize.

If the robot moves too fast, increase the duration of a phase. For example:

```python
elif step < 700:
    target_pos_w, gripper_cmd = self._phase_move_above_pink(...)
```

If the robot closes the gripper before it reaches the cup, extend the lower phase:

```python
elif step < 340:
    target_pos_w, gripper_cmd = self._phase_lower_to_blue(...)
```

## 8. Implement Phase Methods

Each phase should return:

```python
target_pos_w, gripper_cmd
```

Example approach phase:

```python
def _phase_approach_blue(self, blue_cup_pos_w, num_envs, device):
    hover_target = self._blue_hover_target(blue_cup_pos_w)
    alpha = self._step_count / _APPROACH_STEPS
    if self._initial_ee_pos_w is not None:
        target_pos_w = (1.0 - alpha) * self._initial_ee_pos_w + alpha * hover_target
    else:
        target_pos_w = hover_target
    return target_pos_w, _constant_gripper(num_envs, device, _GRIPPER_OPEN)
```

Example grasp phase:

```python
def _phase_grasp_blue(self, blue_cup_pos_w, num_envs, device):
    return self._blue_grasp_target(blue_cup_pos_w), _constant_gripper(num_envs, device, _GRIPPER_CLOSE)
```

Example release phase:

```python
def _phase_release_blue(self, pink_cup_pos_w, num_envs, device):
    return self._stack_place_target(pink_cup_pos_w), _constant_gripper(num_envs, device, _GRIPPER_OPEN)
```

Keep these phase methods small. This makes tuning much easier.

## 9. Define Target Helpers

Target helpers convert object positions into desired end-effector positions.

Example:

```python
def _blue_hover_target(self, blue_cup_pos_w: torch.Tensor) -> torch.Tensor:
    target_pos_w = _offset_xy(blue_cup_pos_w.clone(), _GRASP_XY_OFFSET)
    target_pos_w[:, 2] += _HOVER_Z_OFFSET
    return target_pos_w

def _blue_grasp_target(self, blue_cup_pos_w: torch.Tensor) -> torch.Tensor:
    target_pos_w = _offset_xy(blue_cup_pos_w.clone(), _GRASP_XY_OFFSET)
    target_pos_w[:, 2] += _GRASP_Z_OFFSET
    return target_pos_w

def _stack_place_target(self, pink_cup_pos_w: torch.Tensor) -> torch.Tensor:
    target_pos_w = _offset_xy(pink_cup_pos_w.clone(), _PLACE_XY_OFFSET)
    target_pos_w[:, 2] += _PLACE_Z_OFFSET
    return target_pos_w
```

The helper for xy offsets is:

```python
def _offset_xy(pos_w: torch.Tensor, xy_offset: tuple[float, float]) -> torch.Tensor:
    pos_w[:, 0] += xy_offset[0]
    pos_w[:, 1] += xy_offset[1]
    return pos_w
```

## 10. Convert World Targets to Relative Franka Actions

The phase methods return world-space targets. The Franka action expects relative commands in the robot-root frame.

Use this conversion:

```python
def _relative_franka_action(
    self,
    env,
    target_pos_w: torch.Tensor,
    gripper_cmd: torch.Tensor,
) -> torch.Tensor:
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    root_quat_inv = quat_inv(root_quat_w)

    ee_pos_w = self._ee_pos_w(robot)
    target_pos_root = quat_apply(root_quat_inv, target_pos_w - root_pos_w)
    ee_pos_root = quat_apply(root_quat_inv, ee_pos_w - root_pos_w)
    delta_pos_root = _clamp_delta(target_pos_root - ee_pos_root)

    delta_rot_root = torch.zeros(env.num_envs, 3, device=env.device)
    base_joint_delta = torch.zeros(env.num_envs, 1, device=env.device)
    return torch.cat([delta_pos_root, delta_rot_root, base_joint_delta, gripper_cmd], dim=-1)
```

The clamp prevents unstable jumps:

```python
def _clamp_delta(delta: torch.Tensor, max_norm: float = _MAX_CARTESIAN_DELTA) -> torch.Tensor:
    norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return delta * scale
```

If the robot is too slow, increase `_MAX_CARTESIAN_DELTA` slightly.

If the robot overshoots or becomes unstable, decrease `_MAX_CARTESIAN_DELTA`.

## 11. Implement `advance()` and `reset()`

`advance()` should only update the internal timeline:

```python
def advance(self) -> None:
    self._step_count += 1
    if self._step_count >= self.MAX_STEPS:
        self._episode_done = True
```

`reset()` should clear per-episode state:

```python
def reset(self) -> None:
    self._step_count = 0
    self._episode_done = False
    self._initial_ee_pos_w = None
    self._home_start_pos = None
```

Do not clear setup-time values such as `_rest_joint_pos` unless the robot or scene has changed.

## 12. Minimal Full Skeleton

This is a minimal version you can use as a starting point:

```python
class MyCupStackingStateMachine(StateMachineBase):
    MAX_STEPS = 900

    def __init__(self):
        self._step_count = 0
        self._episode_done = False
        self._ee_body_idx = -1

    def setup(self, env):
        robot = env.scene["robot"]
        self._ee_body_idx = _find_body_index(robot, "panda_hand")

    def check_success(self, env) -> bool:
        blue = env.scene["blue_cup"].data.root_pos_w - env.scene.env_origins
        pink = env.scene["pink_cup"].data.root_pos_w - env.scene.env_origins

        close_xy = torch.logical_and(
            torch.abs(blue[:, 0] - pink[:, 0]) < 0.05,
            torch.abs(blue[:, 1] - pink[:, 1]) < 0.05,
        )
        above = blue[:, 2] > pink[:, 2] + 0.10
        return bool(torch.logical_and(close_xy, above).all().item())

    def get_action(self, env):
        step = self._step_count
        blue = env.scene["blue_cup"].data.root_pos_w.clone()
        pink = env.scene["pink_cup"].data.root_pos_w.clone()

        if step < 120:
            target = blue.clone()
            target[:, 2] += 0.28
            gripper = _constant_gripper(env.num_envs, env.device, _GRIPPER_OPEN)
        elif step < 260:
            target = blue.clone()
            target[:, 2] += 0.16
            gripper = _constant_gripper(env.num_envs, env.device, _GRIPPER_OPEN)
        elif step < 360:
            target = blue.clone()
            target[:, 2] += 0.16
            gripper = _constant_gripper(env.num_envs, env.device, _GRIPPER_CLOSE)
        elif step < 500:
            target = blue.clone()
            target[:, 2] += 0.34
            gripper = _constant_gripper(env.num_envs, env.device, _GRIPPER_CLOSE)
        elif step < 650:
            target = pink.clone()
            target[:, 2] += 0.34
            gripper = _constant_gripper(env.num_envs, env.device, _GRIPPER_CLOSE)
        elif step < 760:
            target = pink.clone()
            target[:, 2] += 0.28
            gripper = _constant_gripper(env.num_envs, env.device, _GRIPPER_CLOSE)
        else:
            target = pink.clone()
            target[:, 2] += 0.30
            gripper = _constant_gripper(env.num_envs, env.device, _GRIPPER_OPEN)

        return self._relative_franka_action(env, target, gripper)

    def advance(self):
        self._step_count += 1
        if self._step_count >= self.MAX_STEPS:
            self._episode_done = True

    def reset(self):
        self._step_count = 0
        self._episode_done = False

    @property
    def is_episode_done(self):
        return self._episode_done
```

The production version should keep the code more factored than this skeleton by using phase methods and target helpers.

## 13. Register the State Machine for Datagen

The state-machine datagen script uses a registry in:

```text
scripts/datagen/state_machine/generate.py
```

To use cup stacking, first export the class in:

```text
source/leisaac/leisaac/datagen/state_machine/__init__.py
```

Example:

```python
from .cup_stacking import CupStackingStateMachine
from .pick_orange import PickOrangeStateMachine

__all__ = ["PickOrangeStateMachine", "CupStackingStateMachine"]
```

Then register the task in `scripts/datagen/state_machine/generate.py`:

```python
from leisaac.datagen.state_machine import CupStackingStateMachine, PickOrangeStateMachine

TASK_REGISTRY = {
    "LeIsaac-SO101-PickOrange-v0": (PickOrangeStateMachine, "so101_state_machine"),
    "LeIsaac-HCIS-CupStacking-SingleArm-v0": (CupStackingStateMachine, "keyboard"),
}
```

The `"keyboard"` device type is used because the Franka template currently defines the 8D relative action layout for keyboard/gamepad style control.

## 14. Tuning Checklist

If the gripper misses the blue cup:

- Tune `_GRASP_XY_OFFSET`.
- Tune `_GRASP_Z_OFFSET`.
- Add more steps to `_phase_lower_to_blue`.

If the cup slips:

- Keep the gripper closed longer.
- Increase the lift phase duration.
- Lower `_MAX_CARTESIAN_DELTA`.

If the blue cup is released too high:

- Lower `_PLACE_Z_OFFSET`.
- Lower `_STACK_HEIGHT`.

If the blue cup knocks over the pink cup:

- Increase `_LIFT_Z_OFFSET`.
- Add or extend a hover phase above the pink cup.
- Lower more slowly by extending `_phase_lower_to_stack`.

If the robot returns home too abruptly:

- Start the home phase earlier.
- Increase `MAX_STEPS`.
- Reduce direct joint blending speed in `pre_step()`.

## 15. Recommended Development Loop

1. Start with one environment.
2. Disable recording until the scripted policy works.
3. Print or visualize the blue cup, pink cup, and end-effector positions.
4. Tune xy offsets first.
5. Tune z offsets second.
6. Tune phase durations last.
7. Only enable recording after `check_success()` reliably returns `True`.

Typical command shape:

```bash
python scripts/datagen/state_machine/generate.py \
    --task LeIsaac-HCIS-CupStacking-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras
```

When the scripted policy is stable, add recording arguments:

```bash
python scripts/datagen/state_machine/generate.py \
    --task LeIsaac-HCIS-CupStacking-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --dataset_file ./datasets/cup_stacking.hdf5 \
    --num_demos 50
```

