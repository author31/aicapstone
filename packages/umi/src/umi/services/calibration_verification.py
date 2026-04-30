import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.transform import Rotation

from ..common.interpolation_util import get_gripper_calibration_interpolator
from ..common.pose_util import pose_to_mat
from .base_service import BaseService


class CalibrationVerificationService(BaseService):
    """Verify outputs of CalibrationService before downstream stages consume them."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.session_dir = self.config.get("session_dir")
        self.tag_id = self.config.get("tag_id", 13)
        self.dist_to_center_threshold = self.config.get("dist_to_center_threshold", 0)
        self.resolution = self.config.get("resolution")
        self.keyframe_only = self.config.get("keyframe_only", True)

        self.min_valid_detections = int(self.config.get("min_valid_detections", 30))
        self.max_translation_norm_m = float(self.config.get("max_translation_norm_m", 5.0))
        self.max_condition_number = float(self.config.get("max_condition_number", 1.0e6))
        self.min_gripper_width_m = float(self.config.get("min_gripper_width_m", 0.005))
        self.max_gripper_width_m = float(self.config.get("max_gripper_width_m", 0.5))
        self.min_tag_visibility_ratio = float(self.config.get("min_tag_visibility_ratio", 0.05))

    def execute(self) -> dict:
        logger.info("Starting calibration verification")
        assert self.session_dir, "Missing session_dir from configuration"
        assert self.resolution, "Missing resolution from configuration"

        session_path = Path(self.session_dir)
        mapping_dir = session_path / "demos/mapping"

        checks: list = []
        tx_slam_tag = self._verify_tx_slam_tag_matrix(mapping_dir, checks)
        if tx_slam_tag is not None:
            self._verify_tx_slam_tag_statistics(mapping_dir, checks)
            self._verify_inversion(tx_slam_tag, checks)

        self._verify_gripper_ranges(session_path / "demos", checks)

        passed = [c for c in checks if c["passed"]]
        failed = [c for c in checks if not c["passed"]]
        logger.info(f"Calibration verification: {len(passed)}/{len(checks)} checks passed")
        for c in checks:
            tag = "PASS" if c["passed"] else "FAIL"
            logger.info(
                f"  [{tag}] {c['name']}: value={c['value']!r} threshold={c['threshold']!r} :: {c['message']}"
            )

        if failed:
            lines = [
                f"- {c['name']}: value={c['value']!r} threshold={c['threshold']!r} :: {c['message']}"
                for c in failed
            ]
            raise RuntimeError(
                "Calibration verification failed:\n" + "\n".join(lines)
            )

        return {"checks": checks, "passed": len(passed), "failed": 0}

    def _record(self, checks, name, passed, value, threshold, message):
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "value": value,
                "threshold": threshold,
                "message": message,
            }
        )

    def _verify_tx_slam_tag_matrix(self, mapping_dir: Path, checks: list):
        path = mapping_dir / "tx_slam_tag.json"
        if not path.is_file():
            self._record(checks, "tx_slam_tag.exists", False, str(path), "file present",
                         "tx_slam_tag.json is missing")
            return None
        try:
            data = json.load(open(path, "r"))
            tx = np.array(data["tx_slam_tag"], dtype=np.float64).reshape(4, 4)
        except Exception as e:
            self._record(checks, "tx_slam_tag.parse", False, repr(e), "parseable 4x4",
                         f"failed to parse tx_slam_tag.json: {e}")
            return None
        self._record(checks, "tx_slam_tag.parse", True, "ok", "parseable 4x4", "loaded as 4x4")

        finite = bool(np.all(np.isfinite(tx)))
        self._record(checks, "tx_slam_tag.finite", finite, finite, True,
                     "all entries finite" if finite else "tx_slam_tag has non-finite entries")
        if not finite:
            return None

        bottom = tx[3]
        bottom_ok = bool(np.allclose(bottom, [0.0, 0.0, 0.0, 1.0], atol=1e-6))
        self._record(checks, "tx_slam_tag.bottom_row", bottom_ok, bottom.tolist(),
                     [0.0, 0.0, 0.0, 1.0],
                     "bottom row is [0,0,0,1]" if bottom_ok else "bottom row not [0,0,0,1]")

        R = tx[:3, :3]
        ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
        ortho_ok = ortho_err < 1e-3
        self._record(checks, "tx_slam_tag.rotation_orthogonal", ortho_ok, ortho_err, 1e-3,
                     f"||R^T R - I||_F = {ortho_err:.2e}")
        det = float(np.linalg.det(R))
        det_ok = abs(det - 1.0) < 1e-3
        self._record(checks, "tx_slam_tag.rotation_det", det_ok, det, 1.0,
                     f"det(R) = {det:.4f}")

        translation = tx[:3, 3]
        t_norm = float(np.linalg.norm(translation))
        t_ok = t_norm < self.max_translation_norm_m
        self._record(checks, "tx_slam_tag.translation_norm", t_ok, t_norm,
                     self.max_translation_norm_m,
                     f"||translation|| = {t_norm:.3f} m")

        cond = float(np.linalg.cond(tx))
        cond_ok = cond < self.max_condition_number
        self._record(checks, "tx_slam_tag.condition_number", cond_ok, cond,
                     self.max_condition_number,
                     f"cond(T) = {cond:.2e}")

        return tx

    def _verify_tx_slam_tag_statistics(self, mapping_dir: Path, checks: list):
        tag_path = mapping_dir / "tag_detection.pkl"
        if not tag_path.is_file():
            self._record(checks, "stats.tag_detection.exists", False, str(tag_path),
                         "file present", "tag_detection.pkl is missing")
            return

        csv_path = mapping_dir / "camera_trajectory.csv"
        if not csv_path.is_file():
            csv_path = mapping_dir / "mapping_camera_trajectory.csv"
        if not csv_path.is_file():
            self._record(checks, "stats.camera_trajectory.exists", False, str(csv_path),
                         "file present", "camera trajectory CSV is missing")
            return

        try:
            df = pd.read_csv(csv_path)
            tag_results = pickle.load(open(tag_path, "rb"))
        except Exception as e:
            self._record(checks, "stats.load", False, repr(e), "loadable",
                         f"failed to load trajectory or tag_detection: {e}")
            return

        is_valid = ~df["is_lost"]
        if self.keyframe_only:
            is_valid &= df["is_keyframe"]

        cam_pose_timestamps = df["timestamp"].loc[is_valid].to_numpy()
        cam_pos = df[["x", "y", "z"]].loc[is_valid].to_numpy()
        cam_rot_quat_xyzw = df[["q_x", "q_y", "q_z", "q_w"]].loc[is_valid].to_numpy()
        if cam_pos.shape[0] == 0:
            self._record(checks, "stats.valid_frames", False, 0, ">0",
                         "no valid (non-lost) frames in camera trajectory")
            return

        cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
        cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float64)
        cam_pose[:, 3, 3] = 1.0
        cam_pose[:, :3, 3] = cam_pos
        cam_pose[:, :3, :3] = cam_rot.as_matrix()

        video_timestamps = np.array([x["time"] for x in tag_results])
        if video_timestamps.size == 0:
            self._record(checks, "stats.tag_frames", False, 0, ">0",
                         "tag_detection.pkl has no entries")
            return

        tum_video_idxs = [int(np.argmin(np.abs(video_timestamps - t))) for t in cam_pose_timestamps]

        all_tx_slam_tag = []
        n_with_tag = 0
        skipped_distance = 0
        skipped_center = 0
        for tum_idx, video_idx in enumerate(tum_video_idxs):
            td = tag_results[video_idx]
            tag_dict = td["tag_dict"]
            if self.tag_id not in tag_dict:
                continue
            n_with_tag += 1
            tag = tag_dict[self.tag_id]
            pose = np.concatenate([tag["tvec"], tag["rvec"]])
            tx_cam_tag = pose_to_mat(pose)
            tx_slam_cam = cam_pose[tum_idx]
            dist_to_cam = float(np.linalg.norm(tx_cam_tag[:3, 3]))
            if dist_to_cam < 0.3 or dist_to_cam > 4:
                skipped_distance += 1
                continue
            corners = tag["corners"]
            tag_center_pix = corners.mean(axis=0)
            img_center = np.array(self.resolution, dtype=np.float64) / 2
            dist_to_center = float(np.linalg.norm(tag_center_pix - img_center) / img_center[0])
            if dist_to_center > self.dist_to_center_threshold:
                skipped_center += 1
                continue
            all_tx_slam_tag.append(tx_slam_cam @ tx_cam_tag)

        n_valid = len(all_tx_slam_tag)
        det_ok = n_valid >= self.min_valid_detections
        self._record(checks, "stats.valid_detections", det_ok, n_valid,
                     self.min_valid_detections,
                     f"{n_valid} detections survived filtering "
                     f"(skipped distance={skipped_distance}, center={skipped_center})")
        if not det_ok:
            return

        n_total_tag_frames = sum(1 for r in tag_results if self.tag_id in r["tag_dict"])
        ratio = n_total_tag_frames / max(len(tag_results), 1)
        ratio_ok = ratio >= self.min_tag_visibility_ratio
        self._record(checks, "stats.tag_visibility_ratio", ratio_ok, ratio,
                     self.min_tag_visibility_ratio,
                     f"tag {self.tag_id} visible in {n_total_tag_frames}/{len(tag_results)} frames")

    def _verify_inversion(self, tx, checks):
        try:
            inv = np.linalg.inv(tx)
            err = float(np.linalg.norm(inv @ tx - np.eye(4), ord="fro"))
        except Exception as e:
            self._record(checks, "downstream.inversion", False, repr(e), "<1e-4",
                         f"np.linalg.inv raised: {e}")
            return
        ok = err < 1e-4
        self._record(checks, "downstream.inversion", ok, err, 1e-4,
                     f"||inv@T - I||_F = {err:.2e}")

    def _verify_gripper_ranges(self, demos_dir: Path, checks: list):
        if not demos_dir.is_dir():
            self._record(checks, "gripper.demos_dir", False, str(demos_dir),
                         "directory present", "demos directory missing")
            return
        gripper_dirs = sorted(demos_dir.glob("gripper_calibration*"))
        if not gripper_dirs:
            self._record(checks, "gripper.dirs", False, 0, ">=1",
                         "no gripper_calibration_* directories found")
            return
        self._record(checks, "gripper.dirs", True, len(gripper_dirs), ">=1",
                     f"found {len(gripper_dirs)} gripper directories")

        for gripper_dir in gripper_dirs:
            name = gripper_dir.name
            range_path = gripper_dir / "gripper_range.json"
            if not range_path.is_file():
                self._record(checks, f"gripper.{name}.exists", False, str(range_path),
                             "file present", "gripper_range.json missing")
                continue
            try:
                data = json.load(open(range_path, "r"))
            except Exception as e:
                self._record(checks, f"gripper.{name}.parse", False, repr(e), "parseable",
                             f"failed to parse: {e}")
                continue

            required = {"gripper_id", "left_finger_tag_id", "right_finger_tag_id",
                        "min_width", "max_width"}
            missing = required - set(data.keys())
            schema_ok = not missing
            self._record(checks, f"gripper.{name}.schema", schema_ok, sorted(missing), [],
                         "all keys present" if schema_ok else f"missing keys: {sorted(missing)}")
            if not schema_ok:
                continue

            min_w = float(data["min_width"])
            max_w = float(data["max_width"])
            finite_ok = np.isfinite(min_w) and np.isfinite(max_w)
            self._record(checks, f"gripper.{name}.widths_finite", finite_ok, [min_w, max_w],
                         "finite", "widths finite" if finite_ok else "widths not finite")
            if not finite_ok:
                continue

            in_range = (self.min_gripper_width_m <= min_w <= self.max_gripper_width_m
                        and self.min_gripper_width_m <= max_w <= self.max_gripper_width_m)
            self._record(checks, f"gripper.{name}.widths_in_range", in_range, [min_w, max_w],
                         [self.min_gripper_width_m, self.max_gripper_width_m],
                         f"min={min_w:.4f} max={max_w:.4f}")

            ordering_ok = min_w < max_w and (max_w - min_w) > 0.005
            self._record(checks, f"gripper.{name}.widths_ordering", ordering_ok,
                         max_w - min_w, ">0.005",
                         f"max - min = {max_w - min_w:.4f}")

            gripper_id = int(data["gripper_id"])
            left = int(data["left_finger_tag_id"])
            right = int(data["right_finger_tag_id"])
            tag_layout_ok = (right == left + 1) and (left == gripper_id * 6)
            self._record(checks, f"gripper.{name}.tag_layout", tag_layout_ok,
                         {"gripper_id": gripper_id, "left": left, "right": right},
                         "right==left+1 and left==gripper_id*6",
                         "tag id layout matches calibration assumption")

            try:
                interp = get_gripper_calibration_interpolator(
                    aruco_measured_width=[min_w, max_w],
                    aruco_actual_width=[min_w, max_w],
                )
                interp([min_w, (min_w + max_w) / 2, max_w])
                interp_ok = True
                interp_msg = "interpolator constructed and evaluated"
            except Exception as e:
                interp_ok = False
                interp_msg = f"interpolator construction failed: {e}"
            self._record(checks, f"gripper.{name}.interpolator", interp_ok,
                         interp_msg if not interp_ok else "ok",
                         "constructible", interp_msg)
