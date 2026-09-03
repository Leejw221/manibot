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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig


@dataclass
class PiperFollowerConfig:
    """Base configuration class for PiPER follower robots."""

    # CAN interface name for the follower (slave) arm
    can_name: str = "can_slave"

    # Teleop init joint positions in degrees [j1, j2, j3, j4, j5, j6].
    # Captured from the master arm's physical pose (read_pose.py).
    joints_init: list[float] | None = field(default_factory=lambda: [-3.98, 13.53, -20.24, 4.9, 38.24, -6.47])

    # Start/rest joint positions in degrees [j1, j2, j3, j4, j5, j6].
    joints_start: list[float] | None = field(default_factory=lambda: [-0.51, 0.49, 1.6, 3.96, 4.83, -8.33])

    # Maximum speed percentage (0-100) for motion control
    max_speed_pct: int = 100

    # Speed percentage used when returning to initial position on disconnect
    init_move_speed_pct: int = 50

    # Whether to return to the initial position before disabling motors on disconnect
    return_to_init_on_disconnect: bool = True

    # Whether to disable the arm on disconnect
    disable_on_disconnect: bool = True

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Timeout in seconds to wait for arm motor enable
    enable_timeout: float = 5.0

    # Gripper mode: "scaled" = master-to-slave conversion, "direct" = use angle directly
    gripper_mode: str = "scaled"


@RobotConfig.register_subclass("piper_follower")
@dataclass
class PiperFollowerRobotConfig(RobotConfig, PiperFollowerConfig):
    pass
