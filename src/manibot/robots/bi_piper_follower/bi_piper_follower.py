"""Bimanual PiPER follower - wraps two PiperFollower instances with left/right prefixes."""

import logging
import math
from functools import cached_property

from manibot.robots.piper_follower import PiperFollower, PiperFollowerRobotConfig
from manibot.utils.neck import DynamixelNeckController
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from lerobot.robots.robot import Robot
from .config_bi_piper_follower import BiPiperFollowerConfig

logger = logging.getLogger(__name__)


class BiPiperFollower(Robot):
    """Bimanual PiPER follower controlling two arms via CAN bus."""

    config_class = BiPiperFollowerConfig
    name = "bi_piper_follower"

    def __init__(self, config: BiPiperFollowerConfig):
        super().__init__(config)
        self.config = config

        left_cfg = PiperFollowerRobotConfig(
            id=f"{config.id}_left" if config.id else None,
            can_name=config.left_arm_config.can_name,
            joints_init=config.left_arm_config.joints_init,
            joints_start=config.left_arm_config.joints_start,
            max_speed_pct=config.left_arm_config.max_speed_pct,
            init_move_speed_pct=config.left_arm_config.init_move_speed_pct,
            return_to_init_on_disconnect=config.left_arm_config.return_to_init_on_disconnect,
            disable_on_disconnect=config.left_arm_config.disable_on_disconnect,
            cameras=config.left_arm_config.cameras,
            enable_timeout=config.left_arm_config.enable_timeout,
            gripper_mode=config.left_arm_config.gripper_mode,
        )

        right_cfg = PiperFollowerRobotConfig(
            id=f"{config.id}_right" if config.id else None,
            can_name=config.right_arm_config.can_name,
            joints_init=config.right_arm_config.joints_init,
            joints_start=config.right_arm_config.joints_start,
            max_speed_pct=config.right_arm_config.max_speed_pct,
            init_move_speed_pct=config.right_arm_config.init_move_speed_pct,
            return_to_init_on_disconnect=config.right_arm_config.return_to_init_on_disconnect,
            disable_on_disconnect=config.right_arm_config.disable_on_disconnect,
            cameras=config.right_arm_config.cameras,
            enable_timeout=config.right_arm_config.enable_timeout,
            gripper_mode=config.right_arm_config.gripper_mode,
        )

        self.left_arm = PiperFollower(left_cfg)
        self.right_arm = PiperFollower(right_cfg)

        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

        # 2-DOF Dynamixel 목 — connect()에서 생성. 심 규약과 동일하게 features
        # 끝에 [neck_pitch, neck_yaw](deg)를 추가한다 (piper_mujoco_env 참조).
        self.neck_ctrl: DynamixelNeckController | None = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        left_ft = self.left_arm.observation_features
        right_ft = self.right_arm.observation_features
        features = {
            **{f"left_{k}": v for k, v in left_ft.items()},
            **{f"right_{k}": v for k, v in right_ft.items()},
        }
        if self.config.enable_neck:
            features["neck_pitch.pos"] = float
            features["neck_yaw.pos"] = float
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        left_ft = self.left_arm.action_features
        right_ft = self.right_arm.action_features
        features = {
            **{f"left_{k}": v for k, v in left_ft.items()},
            **{f"right_{k}": v for k, v in right_ft.items()},
        }
        if self.config.enable_neck:
            features["neck_pitch.pos"] = float
            features["neck_yaw.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        if self.config.enable_neck:
            self.neck_ctrl = DynamixelNeckController(
                self.config.neck, dry_run=self.config.neck_dry_run
            )
            logger.info(
                "Neck enabled on robot"
                + (" [dry-run: no Dynamixel]" if self.config.neck_dry_run else "")
            )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs = {}
        left_obs = self.left_arm.get_observation()
        obs.update({f"left_{k}": v for k, v in left_obs.items()})
        right_obs = self.right_arm.get_observation()
        obs.update({f"right_{k}": v for k, v in right_obs.items()})
        if self.neck_ctrl is not None:
            pitch, yaw = self.neck_ctrl.read()
            obs["neck_pitch.pos"] = math.degrees(pitch)
            obs["neck_yaw.pos"] = math.degrees(yaw)
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_action = {k.removeprefix("left_"): v for k, v in action.items() if k.startswith("left_")}
        right_action = {k.removeprefix("right_"): v for k, v in action.items() if k.startswith("right_")}

        sent_left = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)

        sent = {
            **{f"left_{k}": v for k, v in sent_left.items()},
            **{f"right_{k}": v for k, v in sent_right.items()},
        }

        if self.neck_ctrl is not None:
            missing = [k for k in ("neck_pitch.pos", "neck_yaw.pos") if k not in action]
            if missing:
                raise KeyError(
                    f"action에 없는 키: {missing} (텔레옵이 neck 키를 내보내는지 확인 — "
                    "--teleop.enable_neck=true --teleop.neck_write_hardware=false)"
                )
            pitch_deg = float(action["neck_pitch.pos"])
            yaw_deg = float(action["neck_yaw.pos"])
            self.neck_ctrl.write(math.radians(pitch_deg), math.radians(yaw_deg))
            sent["neck_pitch.pos"] = pitch_deg
            sent["neck_yaw.pos"] = yaw_deg

        return sent

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        if self.neck_ctrl is not None:
            self.neck_ctrl.close()
            self.neck_ctrl = None
