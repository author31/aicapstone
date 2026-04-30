# AI Capstone

## Introduction

Monorepo for AI Capstone. Two workflows under one uv workspace:

- **UMI** — real-world data collection.
- **Isaac Lab / Isaac Sim** — robot motion generation, synthetic data creation, policy training/rollout via LeRobot.

Repo root is uv workspace root. Install:

```bash
uv sync
```

### Where to run what

| Stage | Environment | Why |
|-------|-------------|-----|
| Data creation (teleop, FSM datagen, UMI SLAM) | **Docker container** | Isaac Sim / Isaac Lab / Vulkan stack pinned in image |
| Rollout (policy inference in sim) | **Docker container** | Same Isaac Sim runtime as datagen |
| Training (`lerobot-train`, `accelerate launch`) | **Host machine** | Container adds I/O + GPU passthrough overhead — train natively for throughput |

Use `make launch-isaaclab` for container work. Run training directly against the host `uv` env (`uv sync` then `lerobot-train ...`).

### Layout

- `packages/umi/` — UMI package
- `packages/simulator/` — simulator config layer over upstream Isaac Lab
- `scripts/` — teleoperation, datagen, evaluation scripts
- `umi_pipeline_configs/` — UMI SLAM pipeline configs
- `dependencies/` — vendored submodules (Isaac Lab, etc.)
- `data/`, `datasets/`, `checkpoints/` — runtime artifacts

## Documentation

Long-form guides live under [`docs/`](docs/):

| Document | Description |
|----------|-------------|
| [Isaac Lab configuration tutorial](docs/isaaclab_leisaac_tutorial.md) | Walkthrough of the single-arm Franka template, the cup-stacking task, UMI anchor pose loading, and how to add a new task. |
| [Exporting a self-implemented env config as a standalone file](docs/standalone_env_config_export.md) | Why and how to export an ad-hoc `ManagerBasedRLEnvCfg` subclass to a standalone config file before training / rollout. |
| [LeRobot checkpoint format](docs/lerobot-model-format.md) | On-disk layout of a LeRobot `pretrained_model/` directory: the seven files inside, what each one stores, and inference load order. |
| [LeRobot training procedure](docs/lerobot-training.md) | How to train a LeRobot imitation-learning policy on the host machine: prerequisites, `lerobot-train` flags, multi-GPU, and post-training upload/download. |
| [Synthetic data generation pipeline (cup_stacking walkthrough)](docs/synthetic-data-generation.md) | End-to-end walkthrough of generating synthetic demonstration data for the cup_stacking task. |

## Available Packages

### UMI (Universal Manipulation Interface)

Real-world data collection. SLAM reconstruction pipeline over recorded session.

```bash
uv run umi run-slam-pipeline <pipeline-config> --session-dir <session> --task <task>
```


### Isaac Lab / Isaac Sim

Robot motion generation, synthetic data creation. Wraps upstream Isaac Lab with project task configs in `packages/simulator/`.

New to the task config layout? See [Isaac Lab configuration tutorial](docs/isaaclab_leisaac_tutorial.md) — walks through the single-arm Franka template, the cup-stacking task, UMI anchor pose loading, and a recipe for adding a new task.

#### Docker installation

CUDA 12.8 / Ubuntu 22.04 image. Installs Isaac Sim 5.1.0, Isaac Lab (submodule), `simulator` package, LeRobot.

Driven by `Makefile`. Image tag set via `IMAGE` (default `leisaac-isaaclab:latest`), Dockerfile via `DOCKERFILE`.

| Target | Purpose |
|--------|---------|
| `make submodules` | Init/update git submodules (`dependencies/IsaacLab`, etc.) |
| `make submodules-pull` | Pull latest submodule revisions |
| `make install` | `submodules` + `uv sync` (host workspace install) |
| `make install-dev` | `submodules` + `uv sync --extra dev` |
| `make build-isaaclab` | Init submodules, build Docker image |
| `make launch-isaaclab` | Build, launch container with GPU + X11 + workspace bind-mount, NVIDIA Vulkan ICD probe |
| `make check-isaaclab-gpu` | Verify NVIDIA Vulkan ICD, `nvidia-smi`, GLU/Xt libs, torch CUDA visibility inside image |
| `make test` | Run repo layout tests |

Typical first-run flow:

```bash
make submodules
make build-isaaclab
make launch-isaaclab
```

Isaac Lab submodule must be initialized before build — `Dockerfile` fails fast otherwise.

#### Usage (run inside the container)

1. **Define task.** Task configs in `packages/simulator/`.
2. **Keyboard teleoperation.** Run `scripts/environments/teleoperation/teleop_se3_agent.py` with task ID, device, num envs.
3. **FSM planner datagen.** Run `scripts/datagen/state_machine/generate.py` with task, num demos, recorder flags, target dataset repo ID.

#### LeRobot & Hugging Face Hub workflow

Dataset transfer and training run on the **host machine** (training inside the container is significantly slower). Upload generated demos out of the container, then train on the host.

For the full training procedure, flag reference, and multi-GPU instructions, see [LeRobot training procedure](docs/lerobot-training.md).

Quick workflow:

1. **Upload generated dataset.** From inside the container after datagen: `hf upload <dataset-repo> <local-dataset-dir> --repo-type dataset --revision <tag>`.
2. **Download dataset on host.** `hf download <dataset-repo> --repo-type dataset --local-dir <dir> --revision <rev>`.
3. **Train (single GPU, host).** `lerobot-train` with `--policy.type`, `--dataset.repo_id`, `--output_dir`, `--policy.device`, etc.
4. **Train (multi-GPU, host).** `accelerate launch --multi_gpu --num_processes=N $(which lerobot-train) <args>`.
5. **Upload checkpoints.** `hf upload <model-repo> <local-ckpt-dir> --revision <tag>`.
6. **Download checkpoints (back into the container for rollout).** `hf download <model-repo> --local-dir <dir> --revision <tag>`.

## Rollout (run inside the container)

Run trained policy in sim. Entry: `scripts/rollout.py`. Flags: `--task`, `--policy_type`, `--policy_checkpoint_path`, `--policy_action_horizon`, `--device`, `--enable_cameras`.

Example:

```bash
python scripts/rollout.py \
    --task=LeIsaac-HCIS-CupStacking-SingleArm-v0 \
    --policy_type=lerobot-diffusion \
    --policy_checkpoint_path=tiny-diff \
    --policy_action_horizon=1 \
    --device=cuda \
    --enable_cameras
```

## License

MIT — see [LICENSE](LICENSE).
