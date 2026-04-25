# Franka Joint-Space Cup Stacking

This tutorial explains the change from relative end-effector actions to joint-space actions for the Franka cup-stacking task.

The goal was to make the Franka action vector mean:

```text
[panda_joint1, panda_joint2, panda_joint3, panda_joint4, panda_joint5, panda_joint6, panda_joint7, gripper]
```

instead of:

```text
[dx, dy, dz, drot_x, drot_y, drot_z, d_panda_joint1, gripper]
```

Two files needed to change:

- `source/leisaac/leisaac/tasks/template/single_arm_franka_cfg.py`
- `source/leisaac/leisaac/datagen/state_machine/cup_stacking.py`

## 1. The Action Space Changed

The environment config decides how Isaac Lab interprets the action tensor passed to `env.step(actions)`.

Before the change, Franka used two arm-related action terms:

```python
self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["panda_joint[2-7]"],
    body_name="panda_hand",
    controller=mdp.DifferentialIKControllerCfg(
        command_type="pose",
        ik_method="dls",
        use_relative_mode=True,
    ),
)
self.actions.base_action = mdp.RelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint1"],
    scale=1.0,
)
```

That meant:

- The first 6 action values were relative end-effector pose deltas.
- The 7th action value was a relative command for `panda_joint1`.
- The 8th action value controlled the gripper.

After the change, Franka uses one joint-position action for all seven arm joints:

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
```

Now the action manager reads the first 7 values as absolute position targets for `panda_joint1` through `panda_joint7`.

The important detail is `use_default_offset=False`. Isaac Lab joint actions compute:

```text
processed_action = raw_action * scale + offset
```

With `scale=1.0` and `use_default_offset=False`, the raw state-machine output is used directly as the joint target. This is what we want because the state machine now returns real joint positions in radians.

## 2. The State Machine Had to Match

The cup-stacking state machine still thinks in task-space waypoints:

- move above the blue cup
- move down to grasp
- close the gripper
- lift the blue cup
- move above the pink cup
- place and release
- retreat

That high-level logic did not change.

The change is only at the final step: instead of returning a relative end-effector pose command, the state machine converts the desired end-effector movement into seven joint-position targets.

The old output path was:

```text
target world pose
  -> relative pose delta in robot root frame
  -> [dx, dy, dz, drot_x, drot_y, drot_z, d_panda_joint1, gripper]
```

The new output path is:

```text
target world pose
  -> clipped pose delta in robot root frame
  -> damped least-squares IK
  -> current joint positions + IK delta
  -> clamp to joint limits
  -> [panda_joint1, ..., panda_joint7, gripper]
```

## 3. Resolve Joint and Jacobian Indices

In `CupStackingStateMachine.setup()`, the state machine now records the seven Franka arm joint ids:

```python
_FRANKA_ARM_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)
```

Then it checks that those joints exist on the robot:

```python
missing_joint_names = [
    joint_name for joint_name in _FRANKA_ARM_JOINT_NAMES if joint_name not in joint_names
]
if missing_joint_names:
    raise ValueError(...)

self._arm_joint_ids = [joint_names.index(joint_name) for joint_name in _FRANKA_ARM_JOINT_NAMES]
```

This matters because the action vector order must exactly match the action manager order:

```text
panda_joint1, panda_joint2, ..., panda_joint7
```

The setup code also resolves the Jacobian body index for `panda_hand`.

Isaac Lab stores fixed-base articulation Jacobians with the root body omitted, so the body index needs this adjustment:

```python
if robot.is_fixed_base:
    self._jacobi_body_idx = self._ee_body_idx - 1
    self._jacobi_joint_ids = self._arm_joint_ids
else:
    self._jacobi_body_idx = self._ee_body_idx
    self._jacobi_joint_ids = [joint_id + 6 for joint_id in self._arm_joint_ids]
```

This mirrors the indexing convention used by Isaac Lab task-space action code.

## 4. Keep the Waypoint Logic

The phase functions still return a target end-effector position and gripper command:

```python
def _phase_move_above_target(self, blue_cup_pos_w, num_envs, device):
    target_pos_w = blue_cup_pos_w.clone()
    target_pos_w[:, 2] += _HOVER_Z_OFFSET
    return target_pos_w, _constant_gripper(num_envs, device, _GRIPPER_OPEN)
```

The state machine still builds a target orientation with the gripper pointing down:

```python
target_quat_w = self._gripper_down_quat_w(...)
```

Then `get_action()` calls the new conversion function:

```python
return self._joint_position_franka_action(env, target_pos_w, target_quat_w, gripper_cmd)
```

## 5. Convert Target Pose to Root-Frame Error

The IK solve works in the robot root frame.

The target position and current end-effector position are transformed from world frame to root frame:

```python
root_quat_inv = quat_inv(root_quat_w)

target_pos_root = quat_apply(root_quat_inv, target_pos_w - root_pos_w)
ee_pos_root = quat_apply(root_quat_inv, self._ee_pos_w(robot) - root_pos_w)
delta_pos_root = _clamp_delta(target_pos_root - ee_pos_root)
```

The orientation error is computed as a quaternion difference, converted to axis-angle, transformed to root frame, and clipped:

```python
delta_quat_w = _shortest_quat(quat_mul(target_quat_w, quat_inv(self._ee_quat_w(robot))))
delta_rot_w = axis_angle_from_quat(delta_quat_w)
delta_rot_root = _clamp_delta(quat_apply(root_quat_inv, delta_rot_w), _MAX_ROT_DELTA)
```

The final task-space delta is:

```python
pose_delta_root = torch.cat([delta_pos_root, delta_rot_root], dim=-1)
```

This has shape:

```text
(num_envs, 6)
```

## 6. Compute a Root-Frame Jacobian

Isaac Lab gives the Jacobian in world frame:

```python
jacobian = robot.root_physx_view.get_jacobians()[
    :, self._jacobi_body_idx, :, self._jacobi_joint_ids
].clone()
```

The code rotates both the linear and angular rows into the robot root frame:

```python
root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
jacobian[:, :3, :] = torch.bmm(root_rot_matrix, jacobian[:, :3, :])
jacobian[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian[:, 3:, :])
```

The result has shape:

```text
(num_envs, 6, 7)
```

That means it maps seven arm joint changes to a six-dimensional end-effector pose change.

## 7. Solve Damped Least-Squares IK

The state machine uses the damped least-squares formula:

```text
delta_q = J^T * inverse(J * J^T + lambda^2 * I) * delta_x
```

In code:

```python
jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
lambda_matrix = (_IK_DLS_LAMBDA**2) * torch.eye(
    jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype
)
delta_joint_pos = (
    jacobian_t @ torch.inverse(jacobian @ jacobian_t + lambda_matrix) @ pose_delta.unsqueeze(-1)
)
```

This returns a joint-space delta with shape:

```text
(num_envs, 7)
```

Then the state machine turns it into an absolute joint target:

```python
joint_pos_target = self._arm_joint_pos(robot) + delta_joint_pos
```

## 8. Clamp to Joint Limits

Before returning the action, the joint targets are clamped:

```python
joint_pos_limits = getattr(robot.data, "soft_joint_pos_limits", None)
if joint_pos_limits is None:
    joint_pos_limits = getattr(robot.data, "joint_pos_limits", None)

arm_joint_pos_limits = joint_pos_limits[:, self._arm_joint_ids, :]
joint_pos_target = torch.clamp(
    joint_pos_target,
    arm_joint_pos_limits[..., 0],
    arm_joint_pos_limits[..., 1],
)
```

This prevents the scripted IK step from asking the robot to move outside the configured joint ranges.

## 9. Return the New Action

The final action is:

```python
return torch.cat([joint_pos_target, gripper_cmd], dim=-1)
```

So for one environment, the output shape is:

```text
(1, 8)
```

and the action values mean:

```text
0: panda_joint1 target position
1: panda_joint2 target position
2: panda_joint3 target position
3: panda_joint4 target position
4: panda_joint5 target position
5: panda_joint6 target position
6: panda_joint7 target position
7: gripper open or close command
```

## 10. How Data Generation Uses It

The state-machine generation script maps cup stacking to the Franka keyboard action setup:

```python
TASK_REGISTRY = {
    "LeIsaac-HCIS-CupStacking-SingleArm-v0": (CupStackingStateMachine, "keyboard"),
}
```

That means:

1. `env_cfg.use_teleop_device("keyboard")` installs the Franka joint-space action config.
2. `CupStackingStateMachine.get_action(env)` directly returns the 8D action tensor.
3. `env.step(actions)` sends those joint targets to Isaac Lab's action manager.

The state machine does not use keyboard input during data generation. `"keyboard"` is just the device type used to select the Franka action configuration.

## 11. How to Verify the Change

From the repository root, run:

```shell
PYTHONPYCACHEPREFIX=/tmp/leisaac-pycache python3 -m py_compile \
    source/leisaac/leisaac/datagen/state_machine/cup_stacking.py \
    source/leisaac/leisaac/tasks/template/single_arm_franka_cfg.py
```

Then check for whitespace issues:

```shell
git diff --check -- \
    source/leisaac/leisaac/datagen/state_machine/cup_stacking.py \
    source/leisaac/leisaac/tasks/template/single_arm_franka_cfg.py
```

To test in simulation, run a cup-stacking state-machine rollout:

```shell
python scripts/datagen/state_machine/generate.py \
    --task LeIsaac-HCIS-CupStacking-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --step_hz 60
```

To record successful demonstrations:

```shell
python scripts/datagen/state_machine/generate.py \
    --task LeIsaac-HCIS-CupStacking-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --dataset_file ./datasets/cup_stacking_joint_space.hdf5 \
    --num_demos 10
```

## 12. Common Problems

### Action dimension mismatch

If you see an error about action dimensions, check that the environment config and state machine agree on the action layout.

For this task, both sides must use:

```text
7 arm joints + 1 gripper = 8 action values
```

### Wrong joint order

If the robot moves strangely, check the action order. The state machine returns explicit joint order using `_FRANKA_ARM_JOINT_NAMES`, while the action manager resolves `panda_joint.*`.

For Panda, the resolved order should be:

```text
panda_joint1, panda_joint2, panda_joint3, panda_joint4, panda_joint5, panda_joint6, panda_joint7
```

### Jacobian body index errors

If IK fails with indexing errors, check the fixed-base adjustment:

```python
self._jacobi_body_idx = self._ee_body_idx - 1
```

This is required for fixed-base articulations because PhysX Jacobians omit the fixed root body.

### Large or unstable motions

The state machine clips task-space deltas:

```python
_MAX_CARTESIAN_DELTA = 0.018
_MAX_ROT_DELTA = 0.08
```

If the robot oscillates or moves too aggressively, reduce these values or increase `_IK_DLS_LAMBDA`.

### Gripper command confusion

The gripper is still binary:

```python
_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
```

Positive values open the gripper. Negative values close it.

## 13. Mental Model

Think of the new code as a small IK controller inside the state machine.

The high-level policy still says:

```text
"Move the hand toward this pose."
```

But the environment now expects:

```text
"Move each Franka joint to this position."
```

So the state machine bridges the gap:

```text
desired hand motion -> Jacobian IK -> joint targets -> env.step()
```

That is the core change.
