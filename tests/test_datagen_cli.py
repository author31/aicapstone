"""Smoke tests for the datagen CLI pose-injection flag and fixture.

No Isaac simulator is launched — keeps CI cheap and matches the no-sim
posture of the existing tests.
"""

import sys
from pathlib import Path

import pytest

from leisaac.utils.object_poses_loader import (
    ObjectPoseConfig,
    load_object_poses,
)
from scripts.datagen.state_machine.generate import build_parser

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "test_poses.json"


class TestCliParser:
    def test_object_poses_path_is_required(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_object_poses_path_accepted(self) -> None:
        args = parser = build_parser().parse_args(
            ["--object_poses_path", "/tmp/poses.json"]
        )
        assert args.object_poses_path == "/tmp/poses.json"


class TestFixture:
    def test_fixture_loads_via_object_poses_loader(self) -> None:
        """Verify the committed fixture validates against the loader schema."""
        pose_cfg = ObjectPoseConfig(
            tag_to_object={1: "blue_cup", 2: "pink_cup"},
            anchor_tag_id=0,
            anchor_world_pose=(0.0, 0.0, 0.0),
            object_z=0.12,
            object_roll=0.0,
            object_pitch=0.0,
        )
        loaded = load_object_poses(FIXTURE, pose_cfg)
        assert set(loaded) == {"blue_cup", "pink_cup"}

        for obj_name, (pos, quat) in loaded.items():
            assert len(pos) == 3
            assert len(quat) == 4
