# This code is from "LightWheel/leisaac's scripts/environments/teleoperation/teleop_se3_agent.py".

"""Script to run a leisaac teleoperation with leisaac manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse
import signal

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac teleoperation for leisaac environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=[
        "keyboard",
        "gamepad",
        "so101leader",
        "bi-so101leader",
        "lekiwi-keyboard",
        "lekiwi-gamepad",
        "lekiwi-leader",
    ],
    help="Device for interacting with environment",
)
parser.add_argument(
    "--port", type=str, default="/dev/ttyACM0", help="Port for the teleop device:so101leader, default is /dev/ttyACM0"
)
parser.add_argument(
    "--left_arm_port",
    type=str,
    default="/dev/ttyACM0",
    help="Port for the left teleop device:bi-so101leader, default is /dev/ttyACM0",
)
parser.add_argument(
    "--right_arm_port",
    type=str,
    default="/dev/ttyACM1",
    help="Port for the right teleop device:bi-so101leader, default is /dev/ttyACM1",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# recorder_parameter
parser.add_argument("--record", action="store_true", help="whether to enable record function")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--resume", action="store_true", help="whether to resume recording in the existing dataset file")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)

parser.add_argument(
    "--object_poses",
    type=str,
    default=None,
    help="Path to the per-episode object_poses.json (UMI schema). Episode count = number of status=='full' entries.",
)
parser.add_argument("--recalibrate", action="store_true", help="recalibrate SO101-Leader or Bi-SO101Leader")
parser.add_argument("--quality", action="store_true", help="whether to enable quality render mode.")
parser.add_argument("--use_lerobot_recorder", action="store_true", help="whether to use lerobot recorder.")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None, help="Lerobot Dataset repository ID.")
parser.add_argument("--lerobot_dataset_fps", type=int, default=30, help="Lerobot Dataset frames per second.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import time

import numpy as np

import math as _math

import simulator.tasks  # noqa: F401
from simulator.tasks.external import resolve_task
from simulator.utils.object_poses_loader import load_episode_poses

import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def manual_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool):
    if hasattr(env, "termination_manager"):
        if success:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        else:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = success


def _apply_episode_poses(env, poses):
    """Write per-object root poses for the current episode into the sim."""
    device = env.device
    for name, (pos, quat) in poses.items():
        obj = env.scene[name]
        pose_tensor = torch.tensor(
            [[pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]],
            device=device,
            dtype=torch.float32,
        ).repeat(env.num_envs, 1)
        obj.write_root_pose_to_sim(pose_tensor)
        w, x, y, z = quat
        yaw_deg = _math.degrees(_math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        print(
            f"  [pose] {name}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
            f"yaw={yaw_deg:+6.1f}°"
        )


def main():  # noqa: C901
    """Running lerobot teleoperation with leisaac manipulation environment."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    task_name = resolve_task(args_cli.task)
    args_cli.task = task_name
    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())

    # load object poses if provided
    episodes = []
    if args_cli.object_poses:
        object_pose_cfg = getattr(env_cfg, "object_pose_cfg", None)
        if object_pose_cfg is None:
            raise ValueError(
                f"Task '{task_name}' env_cfg has no 'object_pose_cfg' attribute; "
                "cannot resolve anchor frame for --object_poses."
            )
        episodes = load_episode_poses(args_cli.object_poses, object_pose_cfg)
        if not episodes:
            raise ValueError(f"No 'status==full' episodes in {args_cli.object_poses}; nothing to replay.")
        print(f"Loaded {len(episodes)} replay episodes from {args_cli.object_poses}")

    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "quality"

    # precheck task and teleop device
    if "BiArm" in task_name:
        assert args_cli.teleop_device == "bi-so101leader", "only support bi-so101leader for bi-arm task"
    if "LeKiwi" in task_name:
        assert args_cli.teleop_device in [
            "lekiwi-leader",
            "lekiwi-keyboard",
            "lekiwi-gamepad",
        ], "only support lekiwi-leader, lekiwi-keyboard, lekiwi-gamepad for lekiwi task"
    is_direct_env = "Direct" in task_name
    if is_direct_env:
        assert args_cli.teleop_device in [
            "so101leader",
            "bi-so101leader",
        ], "only support so101leader or bi-so101leader for direct task"

    # timeout and terminate preprocess
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.manual_terminate = True
    else:
        # modify configuration
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None
    # recorder preprocess & manual success terminate preprocess
    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        else:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
                assert os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file does not exist, please don't use '--resume' if you want to record a new dataset"
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                assert not os.path.exists(
                    args_cli.dataset_file
                ), "the dataset file already exists, please use '--resume' to resume recording"
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if is_direct_env:
            env_cfg.return_success_status = False
        else:
            if not hasattr(env_cfg.terminations, "success"):
                setattr(env_cfg.terminations, "success", None)
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            )
    else:
        env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    # replace the original recorder manager with the streaming recorder manager or lerobot recorder manager
    if args_cli.record:
        del env.recorder_manager
        if args_cli.use_lerobot_recorder:
            from leisaac.enhance.datasets.lerobot_dataset_handler import (
                LeRobotDatasetCfg,
            )
            from leisaac.enhance.managers.lerobot_recorder_manager import (
                LeRobotRecorderManager,
            )

            dataset_cfg = LeRobotDatasetCfg(
                repo_id=args_cli.lerobot_dataset_repo_id,
                fps=args_cli.lerobot_dataset_fps,
            )
            env.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env)
        else:
            env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
            env.recorder_manager.flush_steps = 100
            env.recorder_manager.compression = "lzf"

    # create controller
    target_frame = getattr(env.cfg, "teleop_target_frame", "gripper")

    robot_name = getattr(env.cfg, "robot_name", "")

    if args_cli.teleop_device == "keyboard":
        if robot_name == "franka_panda":
            from simulator.devices import FrankaKeyboard

            teleop_interface = FrankaKeyboard(env, sensitivity=args_cli.sensitivity)
        else:
            from leisaac.devices import SO101Keyboard
            from leisaac.devices.device_base import Device

            if target_frame != "gripper":

                class _KeyboardAdapter(SO101Keyboard):
                    def __init__(self, env, sensitivity=1.0):
                        Device.__init__(self, env, "keyboard")
                        self.pos_sensitivity = 0.01 * sensitivity
                        self.joint_sensitivity = 0.15 * sensitivity
                        self.rot_sensitivity = 0.15 * sensitivity
                        self._create_key_bindings()
                        self._delta_action = np.zeros(8)
                        self.asset_name = "robot"
                        self.robot_asset = self.env.scene[self.asset_name]
                        self.target_frame = target_frame
                        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
                        self.target_frame_idx = body_idxs[0]

                teleop_interface = _KeyboardAdapter(env, sensitivity=args_cli.sensitivity)
            else:
                teleop_interface = SO101Keyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "gamepad":
        from leisaac.devices import SO101Gamepad
        from leisaac.devices.device_base import Device

        if target_frame != "gripper":

            class _GamepadAdapter(SO101Gamepad):
                def __init__(self, env, sensitivity=1.0):
                    Device.__init__(self, env, "gamepad")
                    self.pos_sensitivity = 0.01 * sensitivity
                    self.joint_sensitivity = 0.15 * sensitivity
                    self.rot_sensitivity = 0.15 * sensitivity
                    self._create_key_bindings()
                    self._delta_action = np.zeros(8)
                    self.asset_name = "robot"
                    self.robot_asset = self.env.scene[self.asset_name]
                    self.target_frame = target_frame
                    body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
                    self.target_frame_idx = body_idxs[0]

            teleop_interface = _GamepadAdapter(env, sensitivity=args_cli.sensitivity)
        else:
            teleop_interface = SO101Gamepad(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "so101leader":
        from leisaac.devices import SO101Leader

        teleop_interface = SO101Leader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
    elif args_cli.teleop_device == "bi-so101leader":
        from leisaac.devices import BiSO101Leader

        teleop_interface = BiSO101Leader(
            env, left_port=args_cli.left_arm_port, right_port=args_cli.right_arm_port, recalibrate=args_cli.recalibrate
        )
    elif args_cli.teleop_device == "lekiwi-keyboard":
        from leisaac.devices import LeKiwiKeyboard

        teleop_interface = LeKiwiKeyboard(env, sensitivity=args_cli.sensitivity)
    elif args_cli.teleop_device == "lekiwi-leader":
        from leisaac.devices import LeKiwiLeader

        teleop_interface = LeKiwiLeader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
    elif args_cli.teleop_device == "lekiwi-gamepad":
        from leisaac.devices import LeKiwiGamepad

        teleop_interface = LeKiwiGamepad(env, sensitivity=args_cli.sensitivity)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'gamepad', 'so101leader',"
            " 'bi-so101leader', 'lekiwi-keyboard', 'lekiwi-leader', 'lekiwi-gamepad'."
        )

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # add teleoperation key for task success
    should_reset_task_success = False

    def reset_task_success():
        nonlocal should_reset_task_success
        should_reset_task_success = True
        reset_recording_instance()

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("N", reset_task_success)
    teleop_interface.display_controls()
    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    teleop_interface.reset()

    # apply first episode poses if available
    next_episode_idx = 0
    if episodes:
        _apply_episode_poses(env, episodes[next_episode_idx])
        next_episode_idx = 1
        print(f"[teleop] Episode 1/{len(episodes)} poses applied.")

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count

    start_record_state = False

    interrupted = False

    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) signal."""
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt (Ctrl+C) detected. Cleaning up resources...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        while simulation_app.is_running() and not interrupted:
            # run everything in inference mode
            with torch.inference_mode():
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)
                actions = teleop_interface.advance()
                if should_reset_task_success:
                    print("Task Success!!!")
                    should_reset_task_success = False
                    if args_cli.record:
                        manual_terminate(env, True)
                if should_reset_recording_instance:
                    env.reset()
                    if episodes and next_episode_idx < len(episodes):
                        _apply_episode_poses(env, episodes[next_episode_idx])
                        next_episode_idx += 1
                        print(f"[teleop] Episode {next_episode_idx}/{len(episodes)} poses applied.")
                    elif episodes:
                        print(f"[teleop] All {len(episodes)} episodes exhausted.")
                    should_reset_recording_instance = False
                    if start_record_state:
                        if args_cli.record:
                            print("Stop Recording!!!")
                        start_record_state = False
                    if args_cli.record:
                        manual_terminate(env, False)
                    # print out the current demo count if it has changed
                    if (
                        args_cli.record
                        and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                        > current_recorded_demo_count
                    ):
                        current_recorded_demo_count = (
                            env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                        )
                        print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                    if (
                        args_cli.record
                        and args_cli.num_demos > 0
                        and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                        >= args_cli.num_demos
                    ):
                        print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                        break

                elif actions is None:
                    env.render()
                # apply actions
                else:
                    if not start_record_state:
                        if args_cli.record:
                            print("Start Recording!!!")
                        start_record_state = True
                    env.step(actions)
                if rate_limiter:
                    rate_limiter.sleep(env)
            if interrupted:
                break
    except Exception as e:
        import traceback

        print(f"\n[ERROR] An error occurred: {e}\n")
        traceback.print_exc()
        print("[INFO] Cleaning up resources...")
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)
        # finalize the recorder manager
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        # close the simulator
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
