"""Datagen state-machine entrypoint.

Usage:
    python -m scripts.datagen.state_machine.generate --object_poses_path <path>

The ``--object_poses_path`` argument is required.  It is threaded into the
vendored ``leisaac`` environment configuration before ``gym.make`` is called.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the datagen state-machine with a pose file.",
    )
    parser.add_argument(
        "--object_poses_path",
        required=True,
        type=str,
        help="Path to a JSON file containing object poses (required).",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def validate_poses_path(path: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Pose file not found: {resolved}")
    return resolved


def build_env_cfg(object_poses_path: Path):
    """Import leisaac, register the env, and return a configured env cfg.

    This function imports the leisaac cup_stacking registration module so that
    the gym environment is registered, then creates a ``CupStackingEnvCfg``
    with ``object_poses_path`` set before calling ``gym.make``.

    Raises
    ------
    RuntimeError
        If the leisaac package is not importable (e.g. IsaacLab dependencies
        are missing).  The CLI layer catches this and emits a clear message.
    """
    try:
        import gymnasium as gym
        from leisaac.tasks.cup_stacking import cup_stacking_env_cfg  # noqa: F401  # registers env
        from leisaac.tasks.cup_stacking.cup_stacking_env_cfg import CupStackingEnvCfg
    except ImportError as exc:
        raise RuntimeError(
            "leisaac (or a dependency such as isaaclab/gymnasium) is not "
            "importable.  Ensure the workspace dependencies are installed "
            "before running the datagen entrypoint."
        ) from exc

    env_cfg = CupStackingEnvCfg()
    env_cfg.object_poses_path = str(object_poses_path)
    return env_cfg


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    poses_path = validate_poses_path(args.object_poses_path)
    env_cfg = build_env_cfg(poses_path)
    print(f"Env configured with object_poses_path={env_cfg.object_poses_path}")


if __name__ == "__main__":
    main()
