# LeRobot Training Procedure

Guide for training a LeRobot imitation-learning policy on the host machine.

## Quick Start

```bash
# 1. Install all dependencies
uv sync

# 2. Activate the virtual environment
source .venv/bin/activate
```

After these two steps, the `lerobot-train` command is available in your terminal.

**What is a "policy"?** A policy is a trained model that tells the robot what to do next based on what it sees (camera images) and its current arm position. We train a policy by showing it many examples of a human doing the task (imitation learning), so the robot learns to copy that behavior.

**Default policy: Diffusion Policy.** This project uses **diffusion policy** (`--policy.type=diffusion`) as the default. Diffusion policy generates smooth, multi-step action sequences and works well for manipulation tasks like cup stacking.

> **Why train on the host machine (not inside Docker)?** Training inside the Docker container is significantly slower due to I/O and GPU passthrough overhead. Always run `lerobot-train` directly on the host machine using the `uv` environment.

## Prerequisites

Before you start, make sure you have:

1. **Nvidia GPU with CUDA** — training requires a GPU. Run `nvidia-smi` in your terminal to verify your GPU is detected. If you see an error, CUDA is not installed correctly.
2. **`uv` workspace synced** — run `uv sync` at the repo root. This installs all Python dependencies.
3. **Dataset on Hugging Face Hub** — your recorded demonstration data must be uploaded first. See [LeRobot & Hugging Face Hub workflow](../README.md#lerobot--hugging-face-hub-workflow).
4. **`HF_USER` environment variable set** — run `export HF_USER=<your-huggingface-username>` so the commands below work as-is.
5. **(Optional) Weights & Biases** — for tracking training progress with graphs. Run `wandb login` before training if you want this.

For the official LeRobot training documentation, see:
<https://huggingface.co/docs/lerobot/il_robots#train-a-policy>

## Training Command

Copy and paste the command below into your terminal to start training. Make sure you have set `HF_USER` first (see Prerequisites step 4).

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/<repo_id> \
  --policy.type=diffusion \
  --output_dir=<your-output-dir> \
  --job_name=cupstacking \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/my_policy
```

Training takes anywhere from a few hours to over a day depending on dataset size and GPU. You can monitor progress in the terminal output (or on the W&B dashboard if enabled).

## Flag Reference

| Flag | Description |
|------|-------------|
| `--dataset.repo_id=${HF_USER}/<repo_id>` | Which dataset to train on. Use the same `<repo_id>` you uploaded via `hf upload` in the [data generation step](../README.md#run-the-data-generation-pipeline). The trainer downloads it from Hugging Face Hub automatically. |
| `--policy.type=diffusion` | **Policy architecture (default: `diffusion`).** Diffusion policy generates smooth action sequences and is recommended for manipulation tasks. It auto-adapts to your dataset's robot arm configuration and cameras. Other options include `act` (Action Chunking with Transformers). |
| `--policy.device=cuda` | Train on GPU (`cuda`) for full speed. Only set to `cpu` for quick debugging — real training on CPU would take unreasonably long. |
| `--wandb.enable=true` | Turn on Weights & Biases logging so you can watch training progress (loss curves, etc.) from a web browser. Requires `wandb login` beforehand. Set to `false` if you don't need it. |
| `--output_dir=<your-output-dir>` | Folder on your computer where training results (checkpoints, logs, config) are saved. You must specify your own path (e.g., `outputs/train/diffusion_cupstacking_v1`). |
| `--job_name` | A name for this training run — shows up in logs and W&B dashboard. Use something descriptive like `cupstacking_v1`. |
| `--policy.repo_id=${HF_USER}/my_policy` | Where to upload the trained policy on Hugging Face Hub once training is done. This is what the robot will download later to run the task. |

## Multi-GPU Training (Advanced)

If your machine has multiple GPUs and you want faster training, wrap the command with `accelerate`. Replace `N` with the number of GPUs (e.g., `2`):

```bash
accelerate launch --multi_gpu --num_processes=N $(which lerobot-train) <args>
```

Most users with a single GPU can skip this section.

## After Training

Once training finishes (you'll see a completion message in the terminal):

1. **Verify checkpoint** — open your `--output_dir` folder and check that it contains a `pretrained_model/` subfolder and a `train_config.json` file. These are your trained policy files.
2. **Upload to Hub** — push the trained policy to Hugging Face so the robot can download it:
   ```bash
   hf upload ${HF_USER}/my_policy <your-output-dir>/pretrained_model --revision <tag>
   ```
   Replace `<your-output-dir>` with the `--output_dir` path you used during training, and `<tag>` with a version label (e.g., `v1`).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvidia-smi` not found or shows error | CUDA is not installed. Follow Nvidia's CUDA installation guide for your OS. |
| `lerobot-train: command not found` | Run `uv sync` at repo root, then try again. |
| Out of memory (OOM) error | Reduce batch size by adding `--training.batch_size=16` (or lower). |
| Training loss not decreasing | Check your dataset quality — ensure demonstrations are consistent and complete. |
