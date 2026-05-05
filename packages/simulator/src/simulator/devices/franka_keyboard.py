from __future__ import annotations

import carb
import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from leisaac.devices.device_base import Device


class FrankaKeyboard(Device):
    """Keyboard teleop for Franka with device-side differential IK.

    Reads SE(3) deltas in the panda_hand frame from key presses, solves
    differential IK against the current ee pose, and emits an 8-D action
    matching the env's joint-position action space:
    ``[panda_joint_1..7 target, gripper_cmd]`` where ``gripper_cmd`` is a
    latched scalar (>= 0 open, < 0 close) consumed by
    ``BinaryJointPositionActionCfg``.
    """

    _GRIPPER_OPEN = 1.0
    _GRIPPER_CLOSE = -1.0

    def __init__(self, env, sensitivity: float = 1.0):
        super().__init__(env, "keyboard")

        self.pos_sensitivity = 0.01 * sensitivity
        self.rot_sensitivity = 0.15 * sensitivity

        self._create_key_bindings()

        # (dx, dy, dz, droll, dpitch, dyaw, gripper_latch)
        self._delta_action = np.zeros(7)
        self._delta_action[6] = self._GRIPPER_OPEN

        self.asset_name = "robot"
        self.robot_asset = self.env.scene[self.asset_name]

        self.target_frame = "panda_hand"
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self._body_idx = body_idxs[0]
        self.target_frame_idx = self._body_idx

        arm_joint_ids, _ = self.robot_asset.find_joints(["panda_joint.*"])
        self._arm_joint_ids = arm_joint_ids
        self._num_arm_joints = len(arm_joint_ids)

        if self.robot_asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = arm_joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in arm_joint_ids]

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", ik_method="dls", use_relative_mode=True
        )
        self._ik = DifferentialIKController(
            ik_cfg, num_envs=self.env.num_envs, device=self.env.device
        )

    def _add_device_control_description(self):
        rows = [
            ("W", "+x"), ("S", "-x"),
            ("A", "+y"), ("D", "-y"),
            ("J", "+z"), ("K", "-z"),
            ("H", "roll-"), ("L", "roll+"),
            ("U", "pitch-"), ("I", "pitch+"),
            ("Q", "yaw-"), ("E", "yaw+"),
            ("C", "gripper open"), ("M", "gripper close"),
        ]
        for key, desc in rows:
            self._display_controls_table.add_row([key, desc])

    def get_device_state(self):
        delta_b = self._convert_delta_from_frame(self._delta_action)
        gripper_cmd = float(delta_b[6])

        pose_delta = torch.tensor(delta_b[:6], device=self.env.device, dtype=torch.float32)
        pose_delta = pose_delta.unsqueeze(0).repeat(self.env.num_envs, 1)

        ee_pos_w = self.robot_asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self.robot_asset.data.body_quat_w[:, self._body_idx]
        root_pos_w = self.robot_asset.data.root_pos_w
        root_quat_w = self.robot_asset.data.root_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        jac_w = self.robot_asset.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ]
        base_R = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        jac_b = jac_w.clone()
        jac_b[:, :3, :] = torch.bmm(base_R, jac_b[:, :3, :])
        jac_b[:, 3:, :] = torch.bmm(base_R, jac_b[:, 3:, :])

        joint_pos = self.robot_asset.data.joint_pos[:, self._arm_joint_ids]

        self._ik.set_command(pose_delta, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        joint_pos_des = self._ik.compute(ee_pos_b, ee_quat_b, jac_b, joint_pos)

        action = torch.zeros(self._num_arm_joints + 1, device=self.env.device)
        action[: self._num_arm_joints] = joint_pos_des[0]
        action[self._num_arm_joints] = gripper_cmd
        return action.cpu().numpy()

    def reset(self):
        self._delta_action[:6] = 0.0
        self._ik.reset()

    def _on_keyboard_event(self, event, *args, **kwargs):
        super()._on_keyboard_event(event, *args, **kwargs)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            if key in self._POSE_KEY_DELTAS:
                self._delta_action[:6] += self._POSE_KEY_DELTAS[key]
            elif key == "C":
                self._delta_action[6] = self._GRIPPER_OPEN
            elif key == "M":
                self._delta_action[6] = self._GRIPPER_CLOSE
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            key = event.input.name
            if key in self._POSE_KEY_DELTAS:
                self._delta_action[:6] -= self._POSE_KEY_DELTAS[key]

    def _create_key_bindings(self):
        p = self.pos_sensitivity
        r = self.rot_sensitivity
        self._POSE_KEY_DELTAS: dict[str, np.ndarray] = {
            "W": np.array([+p, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "S": np.array([-p, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "A": np.array([0.0, +p, 0.0, 0.0, 0.0, 0.0]),
            "D": np.array([0.0, -p, 0.0, 0.0, 0.0, 0.0]),
            "J": np.array([0.0, 0.0, +p, 0.0, 0.0, 0.0]),
            "K": np.array([0.0, 0.0, -p, 0.0, 0.0, 0.0]),
            "H": np.array([0.0, 0.0, 0.0, -r, 0.0, 0.0]),
            "L": np.array([0.0, 0.0, 0.0, +r, 0.0, 0.0]),
            "U": np.array([0.0, 0.0, 0.0, 0.0, -r, 0.0]),
            "I": np.array([0.0, 0.0, 0.0, 0.0, +r, 0.0]),
            "Q": np.array([0.0, 0.0, 0.0, 0.0, 0.0, -r]),
            "E": np.array([0.0, 0.0, 0.0, 0.0, 0.0, +r]),
        }
