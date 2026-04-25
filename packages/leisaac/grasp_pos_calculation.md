# Beginner Guide: How To Calculate A Grasp Pose

This guide explains how to calculate a grasp pose for the cup-stacking state
machine in this repository. It assumes you have no background in robotics or
math.

The concrete example is the Franka cup-stacking task in:

```text
source/leisaac/leisaac/datagen/state_machine/cup_stacking.py
```

The goal is:

```text
Move the gripper above the blue cup, point the gripper down, close the gripper,
lift the cup, move it above the pink cup, lower it, and release it.
```

## 1. What Is A Grasp Pose?

A robot grasp pose is the pose you want the robot hand to reach before closing
the gripper.

A pose has two parts:

```text
pose = position + orientation
```

Position answers:

```text
Where should the gripper be?
```

Orientation answers:

```text
Which way should the gripper face?
```

For the cup task, a useful grasp pose is:

```text
position:    centered above the blue cup
orientation: gripper facing downward toward the cup
```

The state machine then sends an action to `env.step(actions)` that moves the
robot toward that pose.

## 2. The Three Numbers In A Position

In simulation, a position usually has three numbers:

```text
[x, y, z]
```

Think of them as directions in the room:

```text
x: forward/back
y: left/right
z: up/down
```

The exact meaning of positive `x` and positive `y` depends on the scene, but
positive `z` normally means upward.

Example:

```text
[0.36, -0.30, 0.12]
```

This means:

```text
x = 0.36 meters
y = -0.30 meters
z = 0.12 meters
```

IsaacLab uses meters, so:

```text
0.01 = 1 centimeter
0.10 = 10 centimeters
1.00 = 1 meter
```

## 3. Object Position: Where Is The Cup?

The cup-stacking state machine reads the blue cup position here:

```python
blue_cup_pos_w = env.scene[_BLUE_CUP_NAME].data.root_pos_w.clone()
```

The `_w` suffix means "world frame".

You can read that as:

```text
blue_cup_pos_w = the blue cup position in the simulation world
```

If the cup position is:

```text
blue_cup_pos_w = [0.36, -0.30, 0.12]
```

then the cup center/root is at:

```text
x = 0.36
y = -0.30
z = 0.12
```

Important detail: an object's root position is not always exactly the physical
top, bottom, or center of the visible mesh. It is the reference point stored in
the asset. That is why grasp offsets usually need some tuning in simulation.

## 4. Why We Add Offsets

The robot should not move its hand exactly to the cup root position.

If the cup root is at:

```text
[cup_x, cup_y, cup_z]
```

then a top grasp target should be:

```text
[cup_x + x_offset, cup_y + y_offset, cup_z + z_offset]
```

The offsets say:

```text
x_offset: move a little forward/back from the cup root
y_offset: move a little left/right from the cup root
z_offset: move above the cup root
```

For a centered top grasp, the current code uses:

```python
_GRASP_XY_OFFSET = (0.0, 0.0)
_GRASP_Z_OFFSET = 0.08
```

That means:

```text
Do not shift sideways.
Move the panda_hand target 0.08 meters above the cup root.
```

In code:

```python
target_pos_w = _offset_xy(blue_cup_pos_w.clone(), _GRASP_XY_OFFSET)
target_pos_w[:, 2] += _GRASP_Z_OFFSET
```

Plain English:

```text
1. Start with the blue cup position.
2. Add the x/y grasp offset.
3. Add the z grasp offset.
4. The result is the world-space grasp position.
```

## 5. Worked Position Example

Suppose the blue cup is at:

```text
blue_cup_pos_w = [0.36, -0.30, 0.12]
```

The current grasp offsets are:

```text
x_offset = 0.00
y_offset = 0.00
z_offset = 0.08
```

Calculate the target:

```text
target_x = 0.36 + 0.00 = 0.36
target_y = -0.30 + 0.00 = -0.30
target_z = 0.12 + 0.08 = 0.20
```

So the grasp position is:

```text
target_pos_w = [0.36, -0.30, 0.20]
```

This says:

```text
Put panda_hand directly above the cup root, 8 centimeters higher.
```

## 6. Hover, Grasp, Lift, And Place Positions

A real robot should not instantly jump to the grasp point. The state machine
uses several nearby target positions:

```python
_HOVER_Z_OFFSET = 0.15
_GRASP_Z_OFFSET = 0.08
_LIFT_Z_OFFSET = 0.34
_STACK_HEIGHT = 0.12
_PLACE_Z_OFFSET = _GRASP_Z_OFFSET + _STACK_HEIGHT
```

These describe a simple vertical motion plan.

### Hover

Hover is a safe point above the cup before lowering:

```text
hover_z = cup_z + _HOVER_Z_OFFSET
```

Using the same cup example:

```text
hover_z = 0.12 + 0.15 = 0.27
hover_pos_w = [0.36, -0.30, 0.27]
```

### Grasp

Grasp is lower, where the gripper should close:

```text
grasp_z = cup_z + _GRASP_Z_OFFSET
```

Example:

```text
grasp_z = 0.12 + 0.08 = 0.20
grasp_pos_w = [0.36, -0.30, 0.20]
```

### Lift

Lift raises the cup after closing the gripper:

```text
lift_z = cup_z + _LIFT_Z_OFFSET
```

Example:

```text
lift_z = 0.12 + 0.34 = 0.46
lift_pos_w = [0.36, -0.30, 0.46]
```

### Place

Place moves above the pink cup and lowers the blue cup onto it:

```text
place_z = pink_cup_z + _PLACE_Z_OFFSET
```

Because:

```python
_PLACE_Z_OFFSET = _GRASP_Z_OFFSET + _STACK_HEIGHT
```

the robot keeps the same hand-to-blue-cup grasp spacing while adding the height
needed to stack one cup on top of another.

## 7. What Is Orientation?

Position is where the gripper is.

Orientation is which way the gripper points.

For a top grasp, the gripper must face down.

In this state machine, the controlled body is:

```python
_EE_BODY_NAME = "panda_hand"
```

So the target orientation is the orientation for `panda_hand`, not the cup and
not the robot base.

The current code uses:

```python
_GRIPPER_DOWN_RPY_W = (math.pi, 0.0, 0.0)
```

This is a compact way to say:

```text
Rotate panda_hand so its local +Z direction points toward world -Z.
```

The important idea is simple:

```text
The hand has its own built-in forward/up direction.
We choose a rotation that makes that hand direction point downward at the cup.
```

## 8. Roll, Pitch, Yaw In Plain English

The code uses three orientation numbers called roll, pitch, and yaw:

```text
[roll, pitch, yaw]
```

You can think of them as three ways to turn an object:

```text
roll:  tip it around its x direction
pitch: tip it around its y direction
yaw:   spin it around its z direction
```

The code uses radians, not degrees.

The most important values are:

```text
0        = no rotation
math.pi = half turn = 180 degrees
```

So:

```python
_GRIPPER_DOWN_RPY_W = (math.pi, 0.0, 0.0)
```

means:

```text
roll by 180 degrees
pitch by 0 degrees
yaw by 0 degrees
```

This is the selected top-down hand orientation.

## 9. Why The Code Uses Quaternions

You do not need to understand quaternions deeply to use this code.

A quaternion is just another way to store an orientation.

The code starts with beginner-friendly roll, pitch, yaw:

```python
_GRIPPER_DOWN_RPY_W = (math.pi, 0.0, 0.0)
```

Then it converts that into a quaternion:

```python
target_quat_w = quat_from_euler_xyz(roll, pitch, yaw)
```

Why convert?

Because robotics software often uses quaternions internally. They are more
stable for rotation calculations than roll, pitch, yaw.

Beginner rule:

```text
Use roll/pitch/yaw to choose the orientation.
Use quaternions to compute with it.
```

## 10. Absolute Target Pose Vs Relative Action

A common source of confusion:

```text
The state machine calculates an absolute target pose.
The environment action expects a relative movement command.
```

Absolute target pose:

```text
Go to this exact place in the world.
```

Relative action:

```text
Move a little bit from where you are now.
```

The state machine does this:

```text
1. Calculate the desired world-space grasp pose.
2. Look at the current panda_hand pose.
3. Compute the small difference between current pose and desired pose.
4. Send that small difference to env.step(actions).
```

That is why `_relative_franka_action()` exists.

## 11. Position Difference In Plain English

The current code computes position movement like this:

```python
target_pos_root = quat_apply(root_quat_inv, target_pos_w - root_pos_w)
ee_pos_root = quat_apply(root_quat_inv, ee_pos_w - root_pos_w)
delta_pos_root = _clamp_delta(target_pos_root - ee_pos_root)
```

Plain English:

```text
1. Convert the target position from world coordinates into robot-root coordinates.
2. Convert the current hand position from world coordinates into robot-root coordinates.
3. Subtract current from target.
4. Clamp the result so the robot moves in small steps.
```

The subtraction idea is:

```text
delta = target - current
```

Example:

```text
target_z = 0.20
current_z = 0.25
delta_z = 0.20 - 0.25 = -0.05
```

That means:

```text
Move down by 0.05 meters.
```

But the code clamps movement, so it will not move the full 0.05 meters in one
step. It will move only up to:

```python
_MAX_CARTESIAN_DELTA = 0.018
```

That is 1.8 centimeters per control step.

## 12. Rotation Difference In Plain English

The current code computes rotation movement like this:

```python
ee_quat_w = self._ee_quat_w(robot)
delta_quat_w = _shortest_quat(quat_mul(target_quat_w, quat_inv(ee_quat_w)))
delta_rot_w = axis_angle_from_quat(delta_quat_w)
delta_rot_root = _clamp_delta(quat_apply(root_quat_inv, delta_rot_w), _MAX_ROT_DELTA)
```

Plain English:

```text
1. Read the current hand orientation.
2. Compare the target orientation to the current orientation.
3. Convert that difference into a small rotation command.
4. Express that command in robot-root coordinates.
5. Clamp it so the robot rotates smoothly.
```

The key idea is the same as position:

```text
rotation_delta = target_orientation - current_orientation
```

The code cannot literally subtract orientations with `-`, so it uses quaternion
operations to get the same idea.

The result goes into the action here:

```text
actions[:, 3:6]
```

Those three numbers are the relative rotation command.

## 13. What Goes Into env.step(actions)?

For this Franka task, each action has 8 numbers:

```text
[dx, dy, dz, drot_x, drot_y, drot_z, d_panda_joint1, gripper]
```

Meaning:

```text
dx, dy, dz:                move the hand position a little
drot_x, drot_y, drot_z:    rotate the hand a little
d_panda_joint1:            move the first Panda joint a little
gripper:                   open or close the gripper
```

The state machine sends:

```python
return torch.cat([delta_pos_root, delta_rot_root, base_joint_delta, gripper_cmd], dim=-1)
```

Where:

```text
delta_pos_root:   3 numbers
delta_rot_root:   3 numbers
base_joint_delta: 1 number
gripper_cmd:      1 number
```

Total:

```text
3 + 3 + 1 + 1 = 8 numbers
```

## 14. The Complete Top-Grasp Calculation

For each control step, the top grasp calculation is:

```text
1. Read blue cup position.
2. Choose the current phase: hover, lower, grasp, lift, move, place, release.
3. Calculate target position by adding offsets to the cup position.
4. Calculate target orientation so the gripper faces down.
5. Compare current panda_hand pose to target pose.
6. Send the small position and rotation difference to env.step(actions).
```

In simplified code:

```python
blue_cup_pos_w = env.scene[_BLUE_CUP_NAME].data.root_pos_w.clone()

target_pos_w = blue_cup_pos_w.clone()
target_pos_w[:, 0] += _GRASP_XY_OFFSET[0]
target_pos_w[:, 1] += _GRASP_XY_OFFSET[1]
target_pos_w[:, 2] += _GRASP_Z_OFFSET

target_quat_w = self._gripper_down_quat_w(
    env.num_envs,
    env.device,
    robot.data.root_quat_w.dtype,
)

actions = self._relative_franka_action(
    env,
    target_pos_w,
    target_quat_w,
    gripper_cmd,
)
```

## 15. How To Tune The Grasp Pose

The first version of a grasp pose is usually not perfect. Tune in this order.

### If The Gripper Is Too High

Decrease:

```python
_GRASP_Z_OFFSET
```

Example:

```text
0.08 -> 0.07
```

This moves the hand 1 centimeter lower.

### If The Gripper Hits The Cup Too Hard

Increase:

```python
_GRASP_Z_OFFSET
```

Example:

```text
0.08 -> 0.09
```

This moves the hand 1 centimeter higher.

### If The Gripper Is Sideways From The Cup

Adjust:

```python
_GRASP_XY_OFFSET
```

Example:

```python
_GRASP_XY_OFFSET = (0.01, 0.0)
```

This moves the grasp target 1 centimeter in positive x.

For a true top grasp, start with:

```python
_GRASP_XY_OFFSET = (0.0, 0.0)
```

Only add x/y offset if the asset root is not visually centered or the fingers
need to contact the cup rim in a specific way.

### If The Gripper Points Up Instead Of Down

The position is probably correct, but the target orientation is flipped.

Try one of these:

```python
_GRIPPER_DOWN_RPY_W = (0.0, 0.0, 0.0)
_GRIPPER_DOWN_RPY_W = (0.0, math.pi, 0.0)
_GRIPPER_DOWN_RPY_W = (math.pi, 0.0, 0.0)
```

Use the one that makes the gripper face toward the cup from above.

### If The Gripper Faces Down But The Fingers Are Rotated Wrong

Keep the downward direction, then change yaw:

```python
_GRIPPER_DOWN_RPY_W = (math.pi, 0.0, math.pi / 2.0)
```

Yaw spins the gripper around the vertical direction. This changes the finger
alignment without changing the fact that the gripper faces down.

## 16. How To Debug A Grasp Pose

Use this checklist.

### Step 1: Check The Target Position

Print:

```python
print("blue cup:", blue_cup_pos_w[0])
print("target:", target_pos_w[0])
```

Ask:

```text
Is target x/y centered over the cup?
Is target z above the cup, not below the table?
```

### Step 2: Check The Hand Position

Print:

```python
print("eef:", self._ee_pos_w(robot)[0])
```

Ask:

```text
Is the hand moving closer to the target each step?
```

### Step 3: Check The Rotation Command

Print:

```python
print("delta rot:", delta_rot_root[0])
```

Ask:

```text
Are the rotation numbers nonzero while the hand is not yet facing down?
Do the numbers become small after the hand reaches the target orientation?
```

### Step 4: Tune Only One Thing At A Time

Change one constant, run again, observe the result.

Good order:

```text
1. Fix orientation first.
2. Fix x/y centering second.
3. Fix z height third.
4. Fix phase timing last.
```

## 17. Common Mistakes

### Mistake: Using The Cup Position As The Hand Position

Bad idea:

```text
target_pos = cup_pos
```

The hand target is not the cup target. The hand needs clearance above the cup,
so use:

```text
target_pos = cup_pos + grasp_offset
```

### Mistake: Forgetting That panda_hand Is Not The Fingertip

The IK target body is:

```python
body_name="panda_hand"
```

That means the target position controls the `panda_hand` frame. The visible
finger contact point is lower/farther than the hand frame, so `_GRASP_Z_OFFSET`
does not mean "distance from cup top to fingertip". It means "distance from cup
root to panda_hand target".

### Mistake: Sending Absolute Position Directly To A Relative Controller

The controller is configured in relative mode. Do not send:

```text
[target_x, target_y, target_z]
```

as the action.

Instead, calculate:

```text
delta = target - current
```

and send the small delta.

The state machine already does this inside `_relative_franka_action()`.

### Mistake: Rotating In The Wrong Frame

The target orientation is chosen in world coordinates, but the action is sent in
robot-root coordinates.

That is why the code uses:

```python
quat_apply(root_quat_inv, delta_rot_w)
```

It converts the rotation command into the coordinate frame expected by the
Franka action.

## 18. Quick Reference

The grasp position calculation:

```python
target_pos_w = _offset_xy(blue_cup_pos_w.clone(), _GRASP_XY_OFFSET)
target_pos_w[:, 2] += _GRASP_Z_OFFSET
```

The top-down orientation calculation:

```python
roll = torch.full((num_envs,), math.pi, device=device, dtype=dtype)
pitch = torch.zeros(num_envs, device=device, dtype=dtype)
yaw = torch.zeros(num_envs, device=device, dtype=dtype)
target_quat_w = quat_from_euler_xyz(roll, pitch, yaw)
```

The relative position action:

```python
delta_pos_root = target_pos_root - ee_pos_root
```

The relative rotation action:

```python
delta_quat_w = quat_mul(target_quat_w, quat_inv(ee_quat_w))
delta_rot_w = axis_angle_from_quat(delta_quat_w)
delta_rot_root = quat_apply(root_quat_inv, delta_rot_w)
```

The final action shape:

```text
[dx, dy, dz, drot_x, drot_y, drot_z, d_panda_joint1, gripper]
```

## 19. Mental Model

You can understand the whole grasp pose calculation with this sentence:

```text
Put the robot hand a small tuned distance above the cup center, rotate the hand
so it points down, then send small movement commands until the hand reaches that
target.
```

Everything else in the code exists to make that sentence work reliably in the
simulation coordinate frames.
