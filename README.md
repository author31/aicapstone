# aicapstone MVP

This repository is the single home for the MVP.

## Layout

- `umi/`: vendored from `voilab/packages/umi`
- `leisaac/`: vendored from `author31/leisaac` branch `feat/add-joint-position-action-space`
- `scripts/`: vendored root-level workflow scripts from `voilab/scripts`
- `data/`: runtime dataset directory tracked with `.gitkeep`
- `checkpoints/`: runtime model directory tracked with `.gitkeep`

## Dependency Entry Point

Use the single top-level `pyproject.toml` as the dependency entrypoint for follow-on work:

```bash
uv sync
```

The vendored Python source trees live under `umi/src` and `leisaac/source/leisaac`.
