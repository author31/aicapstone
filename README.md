# aicapstone MVP

This repository is the single home for the MVP.

## Layout

- `umi/`: vendored from `voilab/packages/umi`
- `source/leisaac/leisaac/tasks/cup_stacking/`: MVP-owned task extension copied into `aicapstone`
- `source/leisaac/leisaac/tasks/template/single_arm_franka_cfg.py`: MVP-owned base config extension copied into `aicapstone`
- `scripts/`: vendored root-level workflow scripts from `voilab/scripts`
- `data/`: runtime dataset directory tracked with `.gitkeep`
- `checkpoints/`: runtime model directory tracked with `.gitkeep`

## Dependency Entry Point

Use the single top-level `pyproject.toml` as the dependency entrypoint for follow-on work:

```bash
uv sync
```

`leisaac` is consumed as a dependency from branch `feat/add-joint-position-action-space`, while the local extension sources live under `source/leisaac` and are loaded ahead of the installed package.
