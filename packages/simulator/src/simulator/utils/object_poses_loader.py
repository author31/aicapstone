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
from dataclasses import dataclass, field
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
        per_object_yaw_offset: Per-asset yaw (rad). When ``use_fixed_yaw`` is
            False, this is added to the detected world yaw to cancel each USD's
            local-frame mismatch with the ArUco tag. When ``use_fixed_yaw`` is
            True, the detected yaw is ignored entirely and this value is used
            as the object's world yaw (relative to the anchor) — every episode
            then spawns the object with the same orientation, so a single
            gripper-yaw tuning works across the whole dataset.
        use_fixed_yaw: If True, ignore the per-episode ``rvec`` and lock each
            object's world yaw to ``anchor_yaw + per_object_yaw_offset[name]``.
            Position still varies per episode; only the rotation is locked.
    """

    tag_to_object: Mapping[int, str]
    anchor_tag_id: int
    anchor_world_pose: tuple[float, float, float]
    object_z: float
    object_roll: float = 0.0
    object_pitch: float = 0.0
    per_object_yaw_offset: Mapping[str, float] = field(default_factory=dict)
    use_fixed_yaw: bool = False
    # Names present in the JSON that should be silently skipped by the loader.
    # Use this for objects detected by UMI that you want to spawn at a fixed
    # ``RigidObjectCfg.init_state`` position instead of per-episode poses.
    ignored_object_names: tuple[str, ...] = ()


def load_episode_poses(
    path: str | Path,
    config: ObjectPoseConfig,
) -> list[dict[str, WorldPose]]:
    """Parse the per-episode UMI ``object_poses.json`` schema.

    The UMI ``frame_to_pose`` service emits a list where each entry holds the
    anchor-frame pose of every detected object for one episode::

        [
            {
                "video_name": str,
                "episode_range": [start, end],
                "objects": [
                    {"object_name": str, "rvec": [rx, ry, rz], "tvec": [tx, ty, tz]},
                    ...
                ],
                "status": "full" | "partial" | "none",
            },
            ...
        ]

    Returns one ``{object_name: (pos_xyz, quat_wxyz)}`` dict per kept episode.

    Behavior:
        - Entries with ``status != "full"`` are skipped.
        - For each kept entry, the anchor-frame ``tvec[0]``/``tvec[1]`` are mapped
          to world via ``config.anchor_world_pose``; ``z`` is fixed to
          ``config.object_z`` (``tvec[2]`` is ignored). Yaw is extracted from
          ``rvec`` (axis-angle) and combined with ``config.anchor_world_pose[2]``;
          ``config.object_roll``/``config.object_pitch`` are applied as fixed
          world-frame roll/pitch.
        - Object names must appear in ``set(config.tag_to_object.values())``;
          unknown names raise ``ObjectPosesError``. Each kept episode must
          contain every required object name.

    Raises:
        ObjectPosesError: For missing files, malformed JSON, unknown object
            names, or kept episodes missing required objects.
    """
    json_path = Path(path)
    data = _read_json_any(json_path)
    if not isinstance(data, list):
        raise ObjectPosesError(
            f"{json_path}: expected top-level JSON list, got {type(data).__name__}"
        )

    expected_names = set(config.tag_to_object.values())
    anchor_x, anchor_y, anchor_yaw = config.anchor_world_pose
    cos_a = math.cos(anchor_yaw)
    sin_a = math.sin(anchor_yaw)

    episodes: list[dict[str, WorldPose]] = []
    for ep_idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ObjectPosesError(
                f"{json_path}: episodes[{ep_idx}] must be a mapping, got {type(entry).__name__}"
            )
        status = entry.get("status")
        if status != "full":
            continue

        objects = entry.get("objects")
        if not isinstance(objects, list):
            raise ObjectPosesError(
                f"{json_path}: episodes[{ep_idx}] 'objects' must be a list, "
                f"got {type(objects).__name__}"
            )

        episode_poses: dict[str, WorldPose] = {}
        for obj_idx, obj in enumerate(objects):
            name, rvec, tvec = _parse_episode_object(json_path, ep_idx, obj_idx, obj)
            if name in config.ignored_object_names:
                continue
            if name not in expected_names:
                raise ObjectPosesError(
                    f"{json_path}: episodes[{ep_idx}].objects[{obj_idx}] object_name "
                    f"{name!r} is not in task config (known names: {sorted(expected_names)})"
                )
            if name in episode_poses:
                raise ObjectPosesError(
                    f"{json_path}: episodes[{ep_idx}] duplicate object_name {name!r}"
                )

            x_a, y_a = tvec[0], tvec[1]

            x_w = anchor_x + cos_a * x_a - sin_a * y_a
            y_w = anchor_y + sin_a * x_a + cos_a * y_a
            per_object = float(config.per_object_yaw_offset.get(name, 0.0))
            if config.use_fixed_yaw:
                yaw_w = anchor_yaw + per_object
            else:
                yaw_w = anchor_yaw + _rotvec_to_yaw(rvec) + per_object

            pos = (x_w, y_w, float(config.object_z))
            quat = _euler_xyz_to_quat_wxyz(
                float(config.object_roll), float(config.object_pitch), yaw_w
            )
            episode_poses[name] = (pos, quat)

        missing = expected_names - set(episode_poses)
        if missing:
            raise ObjectPosesError(
                f"{json_path}: episodes[{ep_idx}] missing required object name(s) "
                f"{sorted(missing)}"
            )
        episodes.append(episode_poses)

    return episodes


def _parse_episode_object(
    json_path: Path, ep_idx: int, obj_idx: int, obj: object
) -> tuple[str, tuple[float, float, float], tuple[float, float, float]]:
    if not isinstance(obj, dict):
        raise ObjectPosesError(
            f"{json_path}: episodes[{ep_idx}].objects[{obj_idx}] must be a mapping, "
            f"got {type(obj).__name__}"
        )
    for required in ("object_name", "rvec", "tvec"):
        if required not in obj:
            raise ObjectPosesError(
                f"{json_path}: episodes[{ep_idx}].objects[{obj_idx}] missing field {required!r}"
            )
    name = obj["object_name"]
    if not isinstance(name, str):
        raise ObjectPosesError(
            f"{json_path}: episodes[{ep_idx}].objects[{obj_idx}] object_name must be str, "
            f"got {type(name).__name__}"
        )
    rvec = _parse_vec3(json_path, ep_idx, obj_idx, "rvec", obj["rvec"])
    tvec = _parse_vec3(json_path, ep_idx, obj_idx, "tvec", obj["tvec"])
    return name, rvec, tvec


def _parse_vec3(
    json_path: Path, ep_idx: int, obj_idx: int, field: str, value: object
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ObjectPosesError(
            f"{json_path}: episodes[{ep_idx}].objects[{obj_idx}].{field} must be a "
            f"length-3 list of numbers"
        )
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as e:
        raise ObjectPosesError(
            f"{json_path}: episodes[{ep_idx}].objects[{obj_idx}].{field} must be numeric ({e})"
        ) from e


def _rotvec_to_yaw(rvec: tuple[float, float, float]) -> float:
    """Axis-angle rotation vector → yaw (rotation about world z), in radians.

    Reproduces ``cv2.Rodrigues`` then ``atan2(R[1, 0], R[0, 0])`` without
    requiring numpy / cv2 — keeps this loader dependency-free per the module
    docstring.
    """
    rx, ry, rz = rvec
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-12:
        return 0.0
    kx = rx / theta
    ky = ry / theta
    kz = rz / theta
    c = math.cos(theta)
    s = math.sin(theta)
    one_minus_c = 1.0 - c
    # Rodrigues: R = I + sin(θ)·K + (1 − cos(θ))·K²
    r00 = c + kx * kx * one_minus_c
    r10 = ky * kx * one_minus_c + kz * s
    return math.atan2(r10, r00)


def _read_json_any(json_path: Path) -> object:
    try:
        raw = json_path.read_text()
    except FileNotFoundError as e:
        raise ObjectPosesError(f"object_poses.json not found at {json_path}") from e
    except OSError as e:
        raise ObjectPosesError(f"Failed to read object_poses.json at {json_path}: {e}") from e
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ObjectPosesError(f"Invalid JSON in {json_path}: {e}") from e


def _read_json(json_path: Path) -> dict:
    data = _read_json_any(json_path)
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
