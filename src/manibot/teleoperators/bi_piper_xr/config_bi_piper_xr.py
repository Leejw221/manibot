#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from pathlib import Path

from lerobot.teleoperators.config import TeleoperatorConfig

from .neck import NeckConfig

# Follower's physical "ready" pose (deg). Used as the XR IK seed so the arm holds
# there when idle. MUST match PiperFollowerConfig.joints_init: stock
# lerobot-teleoperate only calls send_feedback for unitree_g1, so without this
# the IK seed stays at zeros and the arm snaps to a flat pose once the loop starts.
_ARM_READY_DEG = [-3.98, 13.53, -20.24, 4.9, 38.24, -6.47]
_ARM_READY_RAD = [float(np.radians(d)) for d in _ARM_READY_DEG]

@dataclass
class MotionTrackerConfig:
    """Configuration for a motion tracker to retarget a specific link."""
    serial: str = ""
    link_target: str = "link3"

@dataclass
class PiperArmConfig:
    """Base configuration for a single Piper arm in BiPiperXR teleoperator."""
    side: str = "left"  # "left" or "right"
    
    # Pose source string for xrobotoolkit_sdk, typically "left_controller" or "right_controller"
    pose_source: str = "left_controller"
    control_trigger: str = "left_grip"
    gripper_trigger: str = "left_trigger"
    
    # IK scaling parameters
    position_scale: float = 1.0
    rotation_scale: float = 1.0

    # 1-Euro Filter
    euro_min_cutoff: float = 1.0
    euro_beta: float = 0.01
    euro_d_cutoff: float = 1.0
    
    # Motion tracker configuration
    motion_tracker: Optional[MotionTrackerConfig] = None
    
    # Placo solver and URDF
    urdf_path: str = str(Path(__file__).parents[2] / "assets" / "urdf" / "piper_description.urdf")
    link_name: str = "link6"  # Target link name for IK (end effector)
    
    # Initial joint positions
    joints_init: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    
    # Gripper limits (in degrees corresponding to Piper master gripper range)
    gripper_open_pos: float = 101.4
    gripper_close_pos: float = 0.0
    # Trigger deadzone: a resting / barely-touched trigger maps to fully open.
    gripper_trigger_deadzone: float = 0.1

@TeleoperatorConfig.register_subclass("bi_piper_xr")
@dataclass
class BiPiperXRTeleopConfig(TeleoperatorConfig):
    # Base configuration for dual arm xr teleoperation
    dt: float = 0.02  # Control loop time step (50 Hz)
    
    left_arm: PiperArmConfig = field(
        default_factory=lambda: PiperArmConfig(
            side="left",
            pose_source="left_controller",
            control_trigger="left_grip",
            gripper_trigger="left_trigger",
            # Calibrated to this arm's measured range (raw 10360..90020).
            gripper_open_pos=90.0,
            gripper_close_pos=10.4,
            joints_init=list(_ARM_READY_RAD),
            motion_tracker=MotionTrackerConfig(
                serial="PC2310MLKB041941G",
                link_target="link3"
            )
        )
    )
    right_arm: PiperArmConfig = field(
        default_factory=lambda: PiperArmConfig(
            side="right",
            pose_source="right_controller",
            control_trigger="right_grip",
            gripper_trigger="right_trigger",
            joints_init=list(_ARM_READY_RAD),
            motion_tracker=MotionTrackerConfig(
                serial="PC2310MLKB041978G",
                link_target="link3"
            )
        )
    )

    # 2-DOF Dynamixel pan/tilt neck driven by the headset pose (None = disabled)
    enable_neck: bool = True
    neck_write_hardware: bool = True
    neck: NeckConfig = field(default_factory=NeckConfig)

    # Headset to world transformation matrix (flattened row-major)
    R_headset_world: list[float] = field(
        default_factory=lambda: [
            0.0, 0.0, -1.0,
           -1.0, 0.0,  0.0,
            0.0, 1.0,  0.0
        ]
    )

    # get_action() always emits both left_/right_-prefixed keys (no single-arm
    # mode). Set this to "left"/"right" when driving a single-arm robot with
    # this teleop, so manibot.scripts.teleoperate knows which half to forward
    # (key-shape matching alone can't disambiguate — both halves look the same
    # after stripping their prefix).
    active_side: str | None = None

