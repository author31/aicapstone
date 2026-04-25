# aicapstone MVP

This repository is the single home for the MVP.

## Layout

- `packages/umi/`: flat-vendored from `voilab/packages/umi`
- `packages/leisaac/`: AICapstone-specific `leisaac` extension package only, using `src/leisaac/`
- `scripts/`: vendored root-level workflow scripts from `voilab/scripts`
- `data/`: runtime dataset directory tracked with `.gitkeep`
- `checkpoints/`: runtime model directory tracked with `.gitkeep`
- `Makefile`, `Dockerfile`, and `pyproject.toml`: root workspace scaffolding

## Workspace Root

The repository root is the uv workspace root. Install the workspace from the repo root:

```bash
uv sync
```

The root `pyproject.toml` wires `umi` and `leisaac` through `[tool.uv.workspace]` and `[tool.uv.sources]`, while each workspace member keeps its own `pyproject.toml` under `packages/`. The `leisaac` workspace member is intentionally limited to the extensions being developed in this repo and lives under `packages/leisaac/src/leisaac`.
