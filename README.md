# AI Capstone

Monorepo for AI Capstone. Two workflows under one uv workspace:

- **UMI** — real-world data collection.
- **Isaac Lab / Isaac Sim** — robot motion generation, synthetic data creation, policy training/rollout via LeRobot.

Install:

```bash
uv sync
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer introduction](docs/dev/introduction.md) | Setup, layout, Docker install, container usage, LeRobot/HF workflow, rollout. Start here. |
| [Isaac Lab configuration tutorial](docs/isaaclab_leisaac_tutorial.md) | Walkthrough of the single-arm Franka template, the cup-stacking task, UMI anchor pose loading, and how to add a new task. |
| [Exporting a self-implemented env config as a standalone file](docs/standalone_env_config_export.md) | Why and how to export an ad-hoc `ManagerBasedRLEnvCfg` subclass to a standalone config file before training / rollout. |
| [LeRobot checkpoint format](docs/lerobot_model_format.md) | On-disk layout of a LeRobot `pretrained_model/` directory: the seven files inside, what each one stores, and inference load order. |
| [LeRobot training procedure](docs/lerobot_training.md) | How to train a LeRobot imitation-learning policy on the host machine: prerequisites, `lerobot-train` flags, multi-GPU, and post-training upload/download. |
| [Synthetic data generation pipeline (cup_stacking walkthrough)](docs/synthetic_data_generation.md) | End-to-end walkthrough of generating synthetic demonstration data for the cup_stacking task. |
| [UMI SLAM pipeline](docs/umi_pipeline.md) | Recording → verify → build_dataset workflow for UMI sessions. |

MIT — see [LICENSE](LICENSE).
