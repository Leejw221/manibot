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
from typing import Any

import numpy as np

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from lerobot.teleoperators.teleoperator import Teleoperator
from .config_piper_leader import PiperLeaderConfig

logger = logging.getLogger(__name__)

# Must match the joint names used in piper_follower
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


class PiperLeader(Teleoperator):
    """Agilex PiPER leader (master) arm for teleoperation via CAN bus using piper_sdk."""

    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig):
        super().__init__(config)
        self.config = config
        self._piper = None
        self._is_connected = False

    @property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {f"{name}.pos": float for name in JOINT_NAMES}
        features["gripper.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        # PiPER uses absolute encoders, no manual calibration needed
        return True

    def calibrate(self) -> None:
        # PiPER uses absolute encoders, calibration is handled by the SDK
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect to the PiPER leader arm via CAN bus for reading teleoperation commands."""
        from piper_sdk import C_PiperInterface_V2

        self._piper = C_PiperInterface_V2(can_name=self.config.can_name)
        self._piper.ConnectPort()
        self._piper.EnableArm(7)

        self._is_connected = True
        logger.info(f"{self} connected.")

    def configure(self) -> None:
        # Leader arm does not require motor configuration; it is passively read
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()

        # Read joint positions from SDK
        # joint_state.joint_N values are in degrees * 1000
        js = self._piper.GetArmJointMsgs().joint_state
        raw_joints = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]

        action: RobotAction = {}
        for name, raw in zip(JOINT_NAMES, raw_joints):
            action[f"{name}.pos"] = raw / 1000.0  # convert to degrees
        action["gripper.pos"] = self._piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # TODO: Implement force feedback if supported by hardware
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._piper is not None:
            self._piper.DisconnectPort()
        self._is_connected = False
        logger.info(f"{self} disconnected.")
