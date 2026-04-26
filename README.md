# aicapstone MVP

This repository is the single home for the MVP.

## Layout

- `packages/umi/`: flat-vendored from `voilab/packages/umi`
- `packages/simulator/`: AICapstone simulator config layer on top of upstream `leisaac`, using `src/simulator/`
- `scripts/`: vendored root-level workflow scripts from `voilab/scripts`
- `data/`: runtime dataset directory tracked with `.gitkeep`
- `checkpoints/`: runtime model directory tracked with `.gitkeep`
- `Makefile`, `Dockerfile`, and `pyproject.toml`: root workspace scaffolding

## Workspace Root

The repository root is the uv workspace root. Install the workspace from the repo root:

```bash
uv sync
```

The root `pyproject.toml` wires `umi` and `simulator` through `[tool.uv.workspace]` and `[tool.uv.sources]`, while each workspace member keeps its own `pyproject.toml` under `packages/`. The `simulator` workspace member is intentionally limited to the simulator config extensions being developed in this repo and lives under `packages/simulator/src/simulator`. It depends on the upstream `leisaac` package as a runtime dependency.
