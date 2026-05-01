# LeRobot Rollout (Policy Evaluation)

Run a trained policy in the Isaac Lab simulator to evaluate robot performance.

## Prerequisites

1. **Linux machine with Nvidia GPU** — verify with `nvidia-smi`. Isaac Lab requires a Linux host with an Nvidia driver.
2. **Docker installed** — the simulator runs inside a container.
3. **Repository cloned** — if you haven't already:
   ```bash
   git clone https://github.com/HCIS-Lab/aicapstone.git
   cd aicapstone
   ```

## Step 1: Launch Isaac Lab Container

From the repository root:

```bash
make launch-isaaclab
```

This builds and starts the Isaac Lab Docker container. On success, your terminal drops into the container shell.

All remaining steps run **inside the container**.

## Step 2: Download Your Trained Policy

Download the pretrained model from Hugging Face Hub:

```bash
export HF_USER=<your-huggingface-username>
huggingface-cli download ${HF_USER}/<repo_id> --local-dir <download_path>
```

Replace:
- `<repo_id>` — the policy repository name you used during training (e.g., `my_policy`)
- `<download_path>` — where to save on disk (e.g., `checkpoints/my_policy`)

If you tagged a specific version during upload, add `--revision <tag>` (e.g., `--revision v1`).

For how to upload a trained policy, see [LeRobot Training — After Training](lerobot_training.md#after-training).

## Step 3: Run Rollout

```bash
python scripts/evaluation/rollout.py \
    --task=<task> \
    --policy_type=lerobot-<policy> \
    --policy_checkpoint_path=<download_path> \
    --device=cuda \
    --enable_cameras
```

### Flag Reference

| Flag | Description |
|------|-------------|
| `--task` | Isaac Lab task to evaluate. See available tasks below. |
| `--policy_type` | Policy type prefixed with `lerobot-`. For diffusion policy use `lerobot-diffusion`. For ACT use `lerobot-act`. |
| `--policy_checkpoint_path` | Path to the downloaded policy directory (same as `<download_path>` from Step 2). |
| `--device=cuda` | Run inference on GPU. |
| `--enable_cameras` | Enable camera rendering for visual observations. |

### Available Tasks

- `HCIS-CupStacking-SingleArm-v0`
- `HCIS-CutleryArrangement-SingleArm-v0`
- `HCIS-ToyBlocksCollection-SingleArm-v0`

### Example

```bash
python scripts/evaluation/rollout.py \
    --task=HCIS-CupStacking-SingleArm-v0 \
    --policy_type=lerobot-diffusion \
    --policy_checkpoint_path=checkpoints/my_policy \
    --device=cuda \
    --enable_cameras
```
