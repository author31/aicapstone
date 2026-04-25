# leisaac Overlay for aicapstone

Distribution name: `leisaac-overlay`. The Python module name remains `leisaac` —
this package merges into the upstream `leisaac` namespace via
`pkgutil.extend_path` (see `src/leisaac/__init__.py` and
`src/leisaac/tasks/__init__.py`).

Upstream `leisaac` is consumed as a `uv` git source in the root
`pyproject.toml`; this overlay only carries the AICapstone-specific extensions.

Current scope:

- `src/leisaac/tasks/cup_stacking/`
- `src/leisaac/tasks/template/single_arm_franka_cfg.py`
- `src/leisaac/utils/object_poses_loader.py`
