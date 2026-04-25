"""Convert UMI-style ``object_poses.json`` files into IsaacLab-ready world poses.

The MVP UMI pipeline reports per-object placements in the frame of a single
ArUco anchor tag observed in the same scene. Each task config knows where that
anchor sits in the IsaacLab world and which scene objects each ArUco tag maps
to. This loader applies the SE(2) anchor-to-world transform with the per-task
fixed ``z`` and roll/pitch conventions and returns ``(pos_xyz, quat_wxyz)``
tuples ready to drop into ``RigidObjectCfg.InitialStateCfg``.

The module is intentionally free of IsaacLab / numpy / torch dependencies so it
can be imported and unit-tested without spinning up the simulator.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# (pos_xyz, quat_wxyz) — quaternion is (w, x, y, z) to match IsaacLab convention.
WorldPose = tuple[tuple[float, float, float], tuple[float, float, float, float]]


class ObjectPosesError(ValueError):
    """Raised when ``object_poses.json`` is malformed or inconsistent with the task config."""


@dataclass(frozen=True)
class ObjectPoseConfig:
    """Per-task config describing how anchor-frame poses map into the IsaacLab world.

    Attributes:
        tag_to_object: ArUco tag id -> scene ``RigidObject`` name.
        anchor_tag_id: Tag id of the anchor; must match the JSON's ``anchor_tag_id``.
        anchor_world_pose: Anchor-frame pose in the IsaacLab world as ``(x, y, yaw_rad)``.
        object_z: Fixed world ``z`` height applied to every object.
        object_roll: Fixed world roll (radians) applied to every object.
        object_pitch: Fixed world pitch (radians) applied to every object.
    """

    tag_to_object: Mapping[int, str]
    anchor_tag_id: int
    anchor_world_pose: tuple[float, float, float]
    object_z: float
    object_roll: float = 0.0
    object_pitch: float = 0.0


def load_object_poses(path: str | Path, config: ObjectPoseConfig) -> dict[str, WorldPose]:
    """Read ``object_poses.json`` and return ``{object_name: (pos_xyz, quat_wxyz)}``.

    JSON schema::

        {
            "anchor_tag_id": <int>,
            "objects": [
                {"tag_id": <int>, "x": <float>, "y": <float>, "yaw": <float>},
                ...
            ]
        }

    ``x``/``y``/``yaw`` are expressed in the anchor frame (meters / radians).
    The anchor entry itself may appear in ``objects`` and is silently skipped.

    Raises:
        ObjectPosesError: If the file is missing, malformed, the anchor tag id
            disagrees with ``config.anchor_tag_id``, an object tag has no
            mapping in ``config.tag_to_object``, or any required mapped tag is
            absent from the JSON.
    """
    json_path = Path(path)
    data = _read_json(json_path)
    _validate_anchor(json_path, data, config.anchor_tag_id)

    objects = data.get("objects")
    if not isinstance(objects, list):
        raise ObjectPosesError(
            f"{json_path}: 'objects' must be a list, got {type(objects).__name__}"
        )

    anchor_x, anchor_y, anchor_yaw = config.anchor_world_pose
    cos_a = math.cos(anchor_yaw)
    sin_a = math.sin(anchor_yaw)

    result: dict[str, WorldPose] = {}
    seen_tags: set[int] = set()

    for idx, entry in enumerate(objects):
        tag_id, x_a, y_a, yaw_a = _parse_object_entry(json_path, idx, entry)

        if tag_id in seen_tags:
            raise ObjectPosesError(
                f"{json_path}: duplicate tag_id {tag_id} in objects (index {idx})"
            )
        seen_tags.add(tag_id)

        if tag_id == config.anchor_tag_id:
            continue

        if tag_id not in config.tag_to_object:
            raise ObjectPosesError(
                f"{json_path}: objects[{idx}] tag_id {tag_id} has no mapping in task "
                f"config (known tags: {sorted(config.tag_to_object)})"
            )

        x_w = anchor_x + cos_a * x_a - sin_a * y_a
        y_w = anchor_y + sin_a * x_a + cos_a * y_a
        yaw_w = anchor_yaw + yaw_a

        pos = (x_w, y_w, float(config.object_z))
        quat = _euler_xyz_to_quat_wxyz(
            float(config.object_roll), float(config.object_pitch), yaw_w
        )
        result[config.tag_to_object[tag_id]] = (pos, quat)

    expected = set(config.tag_to_object)
    missing = expected - seen_tags
    if missing:
        raise ObjectPosesError(
            f"{json_path}: object_poses.json is missing required tag(s) {sorted(missing)} "
            f"(task config maps tags {sorted(expected)} -> "
            f"{[config.tag_to_object[t] for t in sorted(expected)]})"
        )

    return result


def _read_json(json_path: Path) -> dict:
    try:
        raw = json_path.read_text()
    except FileNotFoundError as e:
        raise ObjectPosesError(f"object_poses.json not found at {json_path}") from e
    except OSError as e:
        raise ObjectPosesError(f"Failed to read object_poses.json at {json_path}: {e}") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ObjectPosesError(f"Invalid JSON in {json_path}: {e}") from e

    if not isinstance(data, dict):
        raise ObjectPosesError(
            f"{json_path}: expected top-level JSON object, got {type(data).__name__}"
        )
    return data


def _validate_anchor(json_path: Path, data: dict, expected_anchor_id: int) -> None:
    if "anchor_tag_id" not in data:
        raise ObjectPosesError(f"{json_path}: missing required field 'anchor_tag_id'")
    if "objects" not in data:
        raise ObjectPosesError(f"{json_path}: missing required field 'objects'")

    json_anchor_id = data["anchor_tag_id"]
    if isinstance(json_anchor_id, bool) or not isinstance(json_anchor_id, int):
        raise ObjectPosesError(
            f"{json_path}: 'anchor_tag_id' must be an int, "
            f"got {type(json_anchor_id).__name__}"
        )
    if json_anchor_id != expected_anchor_id:
        raise ObjectPosesError(
            f"{json_path}: anchor_tag_id mismatch — JSON has {json_anchor_id}, "
            f"task config expects {expected_anchor_id}"
        )


def _parse_object_entry(
    json_path: Path, idx: int, entry: object
) -> tuple[int, float, float, float]:
    if not isinstance(entry, dict):
        raise ObjectPosesError(
            f"{json_path}: objects[{idx}] must be a mapping, got {type(entry).__name__}"
        )

    for required in ("tag_id", "x", "y", "yaw"):
        if required not in entry:
            raise ObjectPosesError(
                f"{json_path}: objects[{idx}] missing required field '{required}'"
            )

    tag_id = entry["tag_id"]
    if isinstance(tag_id, bool) or not isinstance(tag_id, int):
        raise ObjectPosesError(
            f"{json_path}: objects[{idx}].tag_id must be int, got {type(tag_id).__name__}"
        )

    try:
        x_a = float(entry["x"])
        y_a = float(entry["y"])
        yaw_a = float(entry["yaw"])
    except (TypeError, ValueError) as e:
        raise ObjectPosesError(
            f"{json_path}: objects[{idx}] x/y/yaw must be numeric ({e})"
        ) from e

    return tag_id, x_a, y_a, yaw_a


def _euler_xyz_to_quat_wxyz(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    """Match ``isaaclab.utils.math.quat_from_euler_xyz`` (extrinsic XYZ, returns wxyz)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)
