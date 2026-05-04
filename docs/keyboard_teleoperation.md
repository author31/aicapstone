# Keyboard Teleoperation

Drive the Franka arm in Isaac Lab from the keyboard. Useful for sanity-checking a scene, debugging an environment config, or recording a small set of demonstrations without external hardware.

> **Run on: GPU machine, inside Docker container.** Same Isaac Lab environment as Step 3 in [Getting Started](getting_started.md).

## Launch

Start the Isaac Lab container (see [Getting Started → Step 3](getting_started.md#step-3-generate-synthetic-data-in-simulation)), then inside the container:

```bash
python scripts/teleop.py \
    --task HCIS-CupStacking-SingleArm-v0 \
    --teleop_device keyboard \
    --num_envs 1 \
    --device cuda \
    --enable_cameras
```

Available tasks:

- `HCIS-CupStacking-SingleArm-v0`
- `HCIS-CutleryArrangement-SingleArm-v0`
- `HCIS-ToyBlocksCollection-SingleArm-v0`

## Controls

End-effector deltas are expressed in the `panda_hand` frame. Differential IK runs device-side and emits joint-position targets; the gripper command is a latched scalar (open until you close it, and vice versa).

### Translation

| Key | Axis |
|-----|------|
| `W` / `S` | +x / -x |
| `A` / `D` | +y / -y |
| `J` / `K` | +z / -z |

### Rotation

| Key | Axis |
|-----|------|
| `H` / `L` | roll- / roll+ |
| `U` / `I` | pitch- / pitch+ |
| `Q` / `E` | yaw- / yaw+ |

### Gripper

| Key | Action |
|-----|--------|
| `C` | Open |
| `M` | Close |

### Episode controls

| Key | Action |
|-----|--------|
| `R` | Reset environment / advance to next replay episode |
| `N` | Mark current episode as success (when recording) |

Hold a pose key for continuous motion. Sensitivity scales with `--sensitivity` (default `1.0`); base step is `0.01 m` per tick for translation, `0.15 rad` per tick for rotation.

## Recording demonstrations

Add `--record` plus a recorder backend to capture the session as a dataset:

```bash
python scripts/teleop.py \
    --task HCIS-CupStacking-SingleArm-v0 \
    --teleop_device keyboard \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --lerobot_dataset_repo_id ${HF_USER}/<repo_id> \
    --num_demos 10
```

Workflow per episode:

1. Drive the arm to complete the task.
2. Press `N` to flag success and reset.
3. Or press `R` to discard and reset without saving.

Set `--num_demos 0` for unbounded recording (Ctrl+C to stop). Use `--resume` to append to an existing dataset.

## Replaying object poses

To match the scene layouts produced by the UMI pipeline (Step 2), pass `--object_poses`:

```bash
python scripts/teleop.py \
    --task HCIS-CupStacking-SingleArm-v0 \
    --teleop_device keyboard \
    --object_poses data/<demo_directory_name>/object_poses.json
```

Each `R` press loads the next episode's object poses. Episode count = number of `status=="full"` entries in the JSON.

## Useful flags

| Flag | Purpose |
|------|---------|
| `--sensitivity <float>` | Scale translation + rotation step sizes |
| `--step_hz <int>` | Environment stepping rate (default 60) |
| `--seed <int>` | Deterministic env seed |
| `--quality` | Enable high-quality render mode (FXAA) |

See `scripts/teleop.py --help` for the full list.

## Troubleshooting

- **Keys do nothing:** the Isaac Sim viewport must have keyboard focus — click on the 3D view.
- **Arm drifts away from target:** IK is operating near a singularity. Reset with `R` and approach the pose from a different configuration.
- **Gripper command ignored:** the action only applies on the next env step; if the sim is paused (focus lost, breakpoint), no progress.
