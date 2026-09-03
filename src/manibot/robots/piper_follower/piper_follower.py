# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property

import numpy as np

from manibot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from lerobot.robots.robot import Robot
from .config_piper_follower import PiperFollowerConfig

logger = logging.getLogger(__name__)

# PiPER is a 6-DoF arm with a parallel-jaw gripper
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


class PiperFollower(Robot):
    """Agilex PiPER follower (slave) arm controlled via CAN bus using piper_sdk."""

    config_class = PiperFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PiperFollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._piper = None
        self._is_connected = False

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {f"{name}.pos": float for name in JOINT_NAMES}
        features["gripper.pos"] = float
        features.update(
            {cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras}
        )
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {f"{name}.pos": float for name in JOINT_NAMES}
        features["gripper.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        # PiPER uses absolute encoders, no manual calibration needed
        return True

    def calibrate(self) -> None:
        # PiPER uses absolute encoders, calibration is handled by the SDK
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect to the PiPER arm via CAN bus, enable motors, and optionally move to initial position."""
        from piper_sdk import C_PiperInterface_V2

        self._piper = C_PiperInterface_V2(self.config.can_name)
        self._piper.ConnectPort()
        self._piper.EnableArm(7)

        # Wait for all 6 motors to report enabled
        self._wait_for_enable()

        # Apply joint limits from the SDK defaults
        self._piper.SetSDKJointLimitParam("j4", -1.7977, 1.7977)
        self._piper.SetSDKJointLimitParam("j5", -1.4265, 1.4265)
        self._piper.SetSDKJointLimitParam("j6", -2.0071, 2.2166)

        # Move to start pose first, then teleop init pose
        if self.config.joints_start is not None:
            logger.info("Moving to start pose...")
            self._move_to_pose(self.config.joints_start)
            time.sleep(3)

        if self.config.joints_init is not None:
            logger.info("Moving to teleop init pose...")
            self._move_to_init()
            time.sleep(3)

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        self._is_connected = True
        logger.info(f"{self} connected.")

    def _wait_for_enable(self) -> None:
        """Block until all 6 motor drivers report enabled, or raise on timeout."""
        start = time.time()
        while time.time() - start < self.config.enable_timeout:
            msgs = self._piper.GetArmLowSpdInfoMsgs()
            all_enabled = all(
                getattr(msgs, f"motor_{i}").foc_status.driver_enable_status for i in range(1, 7)
            )
            if all_enabled:
                logger.info("All PiPER motors enabled.")
                return
            self._piper.EnableArm(7)
            self._piper.GripperCtrl(0, 5000, 0x01, 0)
            time.sleep(0.5)
        raise TimeoutError(f"Failed to enable PiPER arm within {self.config.enable_timeout}s")

    def _move_to_init(self) -> None:
        """Move the arm to the initial joint position at a safe speed."""
        if self.config.joints_init is None:
            return
        self._move_to_pose(self.config.joints_init)
        self._piper.GripperCtrl(0, 1000, 0x01, 0)

    def configure(self) -> None:
        # Set joint control mode
        self._piper.MotionCtrl_2(0x01, 0x01, self.config.max_speed_pct, 0x00)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()

        # Read joint positions from SDK
        # joint_state.joint_N values are in degrees * 1000
        js = self._piper.GetArmJointMsgs().joint_state
        raw_joints = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]

        obs_dict: RobotObservation = {}
        for name, raw in zip(JOINT_NAMES, raw_joints):
            obs_dict[f"{name}.pos"] = raw / 1000.0  # convert to degrees
        obs_dict["gripper.pos"] = self._piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command the arm to move to a target joint configuration.

        Args:
            action: Dictionary with keys like "joint1.pos", ..., "joint6.pos", "gripper.pos"
                    where joint values are in degrees and gripper is in normalized units.

        Returns:
            The action as sent to the robot.
        """
        # Extract joint values and convert to SDK format (degrees * 1000 as int)
        j1, j2, j3, j4, j5, j6 = (int(action[f"{name}.pos"] * 1e3) for name in JOINT_NAMES)

        # Send joint command
        self._piper.MotionCtrl_2(0x01, 0x01, self.config.max_speed_pct, 0x00)
        self._piper.JointCtrl(j1, j2, j3, j4, j5, j6)

        gripper_val = action["gripper.pos"]
        if self.config.gripper_mode == "direct":
            gripper_cmd = gripper_val
        else:
            # Scale master gripper range to slave gripper range (OR3DVP formula)
            # Master: ~0-72.75°, Slave: ~0-101.4°
            gripper_cmd = -2.4 + (101.4 / 72.75) * (gripper_val + 1.75)
        self._piper.GripperCtrl(int(gripper_cmd * 1e3), 1000, 0x01, 0)

        return action

    def _move_to_pose(self, joints_deg: list[float], speed_pct: int | None = None) -> None:
        """Move the arm to an arbitrary joint pose at given speed."""
        if speed_pct is None:
            speed_pct = self.config.init_move_speed_pct
        self._piper.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00)
        j1, j2, j3, j4, j5, j6 = (int(v * 1e3) for v in joints_deg)
        self._piper.JointCtrl(j1, j2, j3, j4, j5, j6)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._piper is not None:
            try:
                if self.config.return_to_init_on_disconnect:
                    disconnect_pose = self.config.joints_start or self.config.joints_init
                    if disconnect_pose is not None:
                        logger.info("Returning to start pose...")
                        self._move_to_pose(disconnect_pose)
                        time.sleep(3)
                if self.config.disable_on_disconnect:
                    self._piper.DisableArm(7)
                    time.sleep(1)
                self._piper.DisconnectPort()
            except Exception as e:
                logger.warning(f"Error disconnecting PiPER arm: {e}")

        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting camera: {e}")

        self._is_connected = False
        logger.info(f"{self} disconnected.")


# lerobot's make_robot_from_config() derives the device class name from the config
# class name by stripping "Config" (PiperFollowerRobotConfig -> PiperFollowerRobot);
# alias so that lookup finds this class without renaming it everywhere it's already
# used directly (bi_piper_follower.py, scripts, ...).
PiperFollowerRobot = PiperFollower
