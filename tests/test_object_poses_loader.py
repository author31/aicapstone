import json
import math
import os
import tempfile

import pytest

from leisaac.utils.object_poses_loader import (
    ObjectPoseConfig,
    ObjectPosesError,
    load_object_poses,
)


def _write_json(payload):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    return path


def _basic_config(**overrides):
    defaults = dict(
        tag_to_object={1: "blue_cup", 2: "pink_cup"},
        anchor_tag_id=0,
        anchor_world_pose=(0.0, 0.0, 0.0),
        object_z=0.12,
        object_roll=0.0,
        object_pitch=0.0,
    )
    defaults.update(overrides)
    return ObjectPoseConfig(**defaults)


# --- Identity anchor: anchor coords pass through to world coords ---


def test_identity_anchor_passthrough_positions():
    path = _write_json({
        "anchor_tag_id": 0,
        "objects": [
            {"tag_id": 1, "x": 0.36, "y": -0.4, "yaw": 0.0},
            {"tag_id": 2, "x": 0.46, "y": -0.4, "yaw": 0.0},
        ],
    })
    out = load_object_poses(path, _basic_config())

    assert set(out) == {"blue_cup", "pink_cup"}
    blue_pos, blue_quat = out["blue_cup"]
    assert blue_pos == pytest.approx((0.36, -0.4, 0.12))
    assert blue_quat == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_anchor_entry_in_objects_is_skipped():
    path = _write_json({
        "anchor_tag_id": 0,
        "objects": [
            {"tag_id": 0, "x": 0.0, "y": 0.0, "yaw": 0.0},
            {"tag_id": 1, "x": 0.1, "y": 0.2, "yaw": 0.0},
            {"tag_id": 2, "x": 0.3, "y": 0.4, "yaw": 0.0},
        ],
    })
    out = load_object_poses(path, _basic_config())
    assert set(out) == {"blue_cup", "pink_cup"}


# --- SE(2) transform from anchor frame to world frame ---


def test_translation_only_anchor():
    cfg = _basic_config(anchor_world_pose=(1.0, 2.0, 0.0))
    path = _write_json({
        "anchor_tag_id": 0,
        "objects": [
            {"tag_id": 1, "x": 0.5, "y": -0.25, "yaw": 0.0},
            {"tag_id": 2, "x": 0.0, "y": 0.0, "yaw": 0.0},
        ],
    })
    out = load_object_poses(path, cfg)
    assert out["blue_cup"][0] == pytest.approx((1.5, 1.75, 0.12))
    assert out["pink_cup"][0] == pytest.approx((1.0, 2.0, 0.12))


def test_yaw_rotation_anchor_rotates_offsets():
    # Anchor at origin rotated by +90° (pi/2): anchor +x maps to world +y.
    cfg = _basic_config(anchor_world_pose=(0.0, 0.0, math.pi / 2.0))
    path = _write_json({
        "anchor_tag_id": 0,
        "objects": [
            {"tag_id": 1, "x": 1.0, "y": 0.0, "yaw": 0.0},
            {"tag_id": 2, "x": 0.0, "y": 1.0, "yaw": 0.0},
        ],
    })
    out = load_object_poses(path, cfg)
    # (1, 0) anchor -> (0, 1) world
    assert out["blue_cup"][0] == pytest.approx((0.0, 1.0, 0.12), abs=1e-9)
    # (0, 1) anchor -> (-1, 0) world
    assert out["pink_cup"][0] == pytest.approx((-1.0, 0.0, 0.12), abs=1e-9)


def test_yaw_composes_anchor_and_object():
    # Anchor yaw=pi/4, object anchor-yaw=pi/4 -> world yaw=pi/2.
    cfg = _basic_config(anchor_world_pose=(0.0, 0.0, math.pi / 4.0))
    path = _write_json({
        "anchor_tag_id": 0,
        "objects": [
            {"tag_id": 1, "x": 0.0, "y": 0.0, "yaw": math.pi / 4.0},
            {"tag_id": 2, "x": 0.0, "y": 0.0, "yaw": 0.0},
        ],
    })
    _, q = load_object_poses(path, cfg)["blue_cup"]
    # quaternion for yaw = pi/2 (roll=pitch=0): (cos(pi/4), 0, 0, sin(pi/4))
    assert q == pytest.approx(
        (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)), abs=1e-9
    )


def test_fixed_z_and_roll_pitch_applied():
    cfg = _basic_config(object_z=0.5, object_roll=0.1, object_pitch=-0.2)
    path = _write_json({
        "anchor_tag_id": 0,
        "objects": [
            {"tag_id": 1, "x": 0.0, "y": 0.0, "yaw": 0.0},
            {"tag_id": 2, "x": 0.0, "y": 0.0, "yaw": 0.0},
        ],
    })
    out = load_object_poses(path, cfg)
    for pos, quat in out.values():
        assert pos[2] == pytest.approx(0.5)
        # Quaternion should be unit-norm.
        assert sum(c * c for c in quat) == pytest.approx(1.0)
        # Non-zero roll/pitch should yield a non-identity quaternion.
        assert quat[0] != pytest.approx(1.0)


# --- Error reporting: malformed JSON / missing tags / anchor mismatch ---


def test_error_missing_file():
    with pytest.raises(ObjectPosesError, match="not found"):
        load_object_poses("/nonexistent/object_poses.json", _basic_config())


def test_error_invalid_json_syntax():
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write("{not valid json")
    with pytest.raises(ObjectPosesError, match="Invalid JSON"):
        load_object_poses(path, _basic_config())


def test_error_top_level_not_object():
    path = _write_json([{"tag_id": 1}])
    with pytest.raises(ObjectPosesError, match="top-level JSON object"):
        load_object_poses(path, _basic_config())


def test_error_missing_anchor_field():
    path = _write_json({"objects": []})
    with pytest.raises(ObjectPosesError, match="anchor_tag_id"):
        load_object_poses(path, _basic_config())


def test_error_missing_objects_field():
    path = _write_json({"anchor_tag_id": 0})
    with pytest.raises(ObjectPosesError, match="objects"):
        load_object_poses(path, _basic_config())


def test_error_anchor_tag_mismatch_lists_both_ids():
    path = _write_json({
        "anchor_tag_id": 99,
        "objects": [
            {"tag_id": 1, "x": 0.0, "y": 0.0, "yaw": 0.0},
            {"tag_id": 2, "x": 0.0, "y": 0.0, "yaw": 0.0},
        ],
    })
    with pytest.raises(ObjectPosesError) as ctx:
        load_object_poses(path, _basic_config())
    msg = str(ctx.value)
    assert "anchor_tag_id mismatch" in msg
    assert "99" in msg
    assert "expects 0" in msg


def test_error_unknown_tag_id_in_objects():
    path = _write_json({
        "anchor_tag_id": 0,
        "objects": [
            {"tag_id": 1, "x": 0.0, "y": 0.0, "yaw": 0.0},
            {"tag_id": 2, "x": 0.0, "y": 0.0, "yaw": 0.0},
            {"tag_id": 7, "x": 0.0, "y": 0.0, "yaw": 0.0},
        ],
    })
    with pytest.raises(ObjectPosesError) as ctx:
        load_object_poses(path, _basic_config())
    msg = str(ctx.value)
    assert "tag_id 7" in msg
    assert "known tags" in msg


def test_error_missing_required_object_tag():
    path = _write_json({
        "anchor_tag_id": 0,
        "objects": [
            {"tag_id": 1, "x": 0.0, "y": 0.0, "yaw": 0.0},
        ],
    })
    with pytest.raises(ObjectPosesError) as ctx:
        load_object_poses(path, _basic_config())
    msg = str(ctx.value)
    assert "missing required tag" in msg
    assert "[2]" in msg


def test_error_object_missing_field():
    path = _write_json({
        "anchor_tag_id": 0,
        "objects": [
            {"tag_id": 1, "x": 0.0, "y": 0.0},  # missing yaw
            {"tag_id": 2, "x": 0.0, "y": 0.0, "yaw": 0.0},
        ],
    })
    with pytest.raises(ObjectPosesError, match="yaw"):
        load_object_poses(path, _basic_config())


def test_error_object_non_numeric_field():
    path = _write_json({
        "anchor_tag_id": 0,
        "objects": [
            {"tag_id": 1, "x": "not-a-number", "y": 0.0, "yaw": 0.0},
            {"tag_id": 2, "x": 0.0, "y": 0.0, "yaw": 0.0},
        ],
    })
    with pytest.raises(ObjectPosesError, match="numeric"):
        load_object_poses(path, _basic_config())


def test_error_duplicate_tag_id():
    path = _write_json({
        "anchor_tag_id": 0,
        "objects": [
            {"tag_id": 1, "x": 0.0, "y": 0.0, "yaw": 0.0},
            {"tag_id": 1, "x": 0.1, "y": 0.0, "yaw": 0.0},
            {"tag_id": 2, "x": 0.0, "y": 0.0, "yaw": 0.0},
        ],
    })
    with pytest.raises(ObjectPosesError, match="duplicate"):
        load_object_poses(path, _basic_config())


def test_error_objects_not_a_list():
    path = _write_json({"anchor_tag_id": 0, "objects": {"a": 1}})
    with pytest.raises(ObjectPosesError, match="'objects' must be a list"):
        load_object_poses(path, _basic_config())


def test_error_anchor_id_wrong_type():
    path = _write_json({"anchor_tag_id": "0", "objects": []})
    with pytest.raises(ObjectPosesError, match="must be an int"):
        load_object_poses(path, _basic_config())
