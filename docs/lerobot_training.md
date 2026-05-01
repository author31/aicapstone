# LeRobot Training Procedure

Guide for training a LeRobot imitation-learning policy on the host machine.

> **Why host?** Training inside the Docker container is significantly slower due to I/O and GPU passthrough overhead. Always run `lerobot-train` natively against the host `uv` environment.

## Prerequisites

- Host machine with Nvidia GPU and CUDA installed.
- `uv` workspace synced (`uv sync` at repo root).
- Dataset uploaded to Hugging Face Hub (see [LeRobot & Hugging Face Hub workflow](../README.md#lerobot--hugging-face-hub-workflow)).
- (Optional) Weights & Biases account — run `wandb login` before training.

For the official LeRobot training documentation, see:
<https://huggingface.co/docs/lerobot/il_robots#train-a-policy>

## Training Command

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/cupstacking \
  --policy.type=act \
  --output_dir=outputs/train/diffusion_policy_cupstacking_test \
  --job_name=cupstacking \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_policy
```

## Flag Reference

| Flag | Description |
|------|-------------|
| `--dataset.repo_id=${HF_USER}/cupstacking` | Hugging Face Hub dataset repository ID. The trainer downloads the dataset from the Hub. Replace `${HF_USER}` with your HF username or org. |
| `--policy.type=act` | Policy architecture type. Setting `act` loads `configuration_diffusion.py` and auto-adapts to the dataset's motor states, action space, and camera configuration (e.g. laptop + phone setup). |
| `--policy.device=cuda` | Device for training. `cuda` targets an Nvidia GPU. Omit or set to `cpu` only for debugging. |
| `--wandb.enable=true` | Enable Weights & Biases logging for training curves, loss plots, and video rollouts. Requires `wandb login` to be run beforehand. |
| `--output_dir` | Local directory where checkpoints, logs, and training config are written. |
| `--job_name` | Human-readable name for the training run (used in W&B and log output). |
| `--policy.repo_id=${HF_USER}/my_policy` | Target Hugging Face Hub repository for pushing the trained policy checkpoint. |

## Multi-GPU Training

For multi-GPU setups, wrap the command with `accelerate`:

```bash
accelerate launch --multi_gpu --num_processes=N $(which lerobot-train) <args>
```

## After Training

1. **Verify checkpoint** — check `--output_dir` for `pretrained_model/` and `train_config.json`.
2. **Upload to Hub** — `hf upload ${HF_USER}/my_policy <local-ckpt-dir> --revision <tag>`.
3. **Download for rollout** — inside the container: `hf download ${HF_USER}/my_policy --local-dir <dir> --revision <tag>`.
4. **Run rollout** — see [Rollout](../README.md#rollout-run-inside-the-container) in the README.
