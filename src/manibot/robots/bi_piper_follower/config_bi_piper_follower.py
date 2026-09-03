from dataclasses import dataclass, field

from manibot.robots.piper_follower.config_piper_follower import PiperFollowerConfig
from manibot.utils.neck import NeckConfig

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("bi_piper_follower")
@dataclass
class BiPiperFollowerConfig(RobotConfig):
    """Configuration for bimanual PiPER follower (two arms)."""

    left_arm_config: PiperFollowerConfig
    right_arm_config: PiperFollowerConfig

    # 2-DOF Dynamixel 목. 켜면 observation/action features 끝에
    # neck_pitch.pos/neck_yaw.pos(deg)가 추가되어 데이터셋에 기록되고,
    # send_action의 neck 키를 robot이 Dynamixel로 구동한다.
    # 텔레옵은 --teleop.enable_neck=true --teleop.neck_write_hardware=false로
    # 실행해 이중 쓰기를 피할 것 (텔레옵은 neck 각도를 action으로만 방출).
    enable_neck: bool = False
    neck: NeckConfig = field(default_factory=NeckConfig)
    # 하드웨어 없이 테스트: 포트를 열지 않고 observation은 최근 명령값 echo.
    neck_dry_run: bool = False
