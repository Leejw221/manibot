import time
import math
import numpy as np
import torch
from torchvision.transforms import Resize

from manibot.robots.base_robot import BaseRobot
from manibot.robots.piper_robot import PiperRobot
from manibot.robots.dual_arm_robot import DualArmRobot
from manibot.cameras.base_camera import BaseCamera
from manibot.cameras.orbbec_camera import OrbbecCamera
from manibot.cameras.realsense_camera import RealSenseCamera
from manibot.utils.neck import DynamixelNeckController, NeckConfig


# Registry of single-arm robot classes. Adding a new robot (SoArm, Nero, ...)
# automatically enables its dual variant through DualArmRobot composition.
SINGLE_ROBOT_REGISTRY = {
    "piper": PiperRobot,
    # "soarm": SoarmRobot,
    # "nero": NeroRobot,
}

CAMERA_REGISTRY = {
    "orbbec": OrbbecCamera,
    "realsense": RealSenseCamera,
}


def create_robot(robot_type: str, robot_mode: str, config: dict) -> BaseRobot:
    """Build a robot instance from (type, mode, config).

    Args:
        robot_type: Key in SINGLE_ROBOT_REGISTRY ('piper', 'soarm', ...).
        robot_mode: 'single' or 'dual'.
        config:
            - single: kwargs for the single-arm class.
            - dual:   {'left': {...kwargs}, 'right': {...kwargs}}.
    """
    if robot_type not in SINGLE_ROBOT_REGISTRY:
        raise ValueError(
            f"Unknown robot_type='{robot_type}'. "
            f"Available: {list(SINGLE_ROBOT_REGISTRY.keys())}"
        )
    single_cls = SINGLE_ROBOT_REGISTRY[robot_type]

    if robot_mode == "single":
        return single_cls(**config)
    elif robot_mode == "dual":
        if not ({"left", "right"} <= set(config.keys())):
            raise ValueError("dual mode requires config['left'] and config['right'].")
        return DualArmRobot(single_cls, config["left"], config["right"])
    else:
        raise ValueError(f"Unknown robot_mode='{robot_mode}'. Use 'single' or 'dual'.")


def create_camera(config: dict) -> BaseCamera:
    camera_type = config.pop("type")
    return CAMERA_REGISTRY[camera_type](**config)


class PiperRealEnv:
    """Real-world environment for PiPER robot with multi-camera setup.

    Provides a similar interface to gym.Env for compatibility with
    the manibot eval pipeline.

    Observation format matches the training data:
        - observation.images.main: (C, H, W) float32 [0, 1]
        - observation.images.wrist: (C, H, W) float32 [0, 1]
        - observation.state: (7,) float32 [joint degrees + gripper]
    """

    def __init__(
        self,
        robot_type: str,
        robot_mode: str,
        robot_config: dict,
        camera_configs: dict[str, dict],
        image_size: tuple[int, int] = (240, 320),
        control_freq: float = 30.0,
        enable_neck: bool = False,
        neck_config: NeckConfig | None = None,
        use_neck: bool = True,
    ):
        """
        Args:
            robot_type: Key in SINGLE_ROBOT_REGISTRY (e.g. 'piper').
            robot_mode: 'single' or 'dual'.
            robot_config: Kwargs for single-arm class, OR
                {'left': {...}, 'right': {...}} for dual.
            camera_configs: Cameras keyed by name (e.g. 'main', 'left_wrist').
            image_size: (H, W) for image resize (must match training).
            control_freq: Control loop frequency in Hz.
            enable_neck: Torque-hold the Dynamixel pan/tilt neck at its home
                pose for the duration of the run. No head tracking here (eval
                has no headset input) -- this only keeps the camera from
                sagging out of the pose it was recorded at.
            neck_config: Overrides for the neck's home pose / ports. Defaults
                to NeckConfig()'s calibrated home_pitch_rad/home_yaw_rad.
            use_neck: enable_neck 그 자체와 별개 축(2026-08-06) — True(기본, 기존 동작
                유지)면 목이 관측/행동에 14+2=16차원으로 포함됨(그 차원으로 학습된 정책 전용).
                False면 enable_neck으로 토크홀드만 하고 관측/행동은 14차원 그대로 유지 —
                목 없이 학습된 기존 체크포인트(예: peg_in_hole)에 순수 안전 고정 목적으로만
                --enable-neck 쓸 때 이걸 함께 꺼야 정규화 차원이 안 어긋남.
        """
        # Deep-copy nested dicts in dual mode (PiperRobot constructor consumes kwargs).
        if robot_mode == "dual":
            rc = {k: v.copy() for k, v in robot_config.items()}
        else:
            rc = robot_config.copy()
        self.robot = create_robot(robot_type, robot_mode, rc)
        self.cameras = {
            name: create_camera(cfg.copy())
            for name, cfg in camera_configs.items()
        }
        self.image_size = image_size
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self._resize = Resize(image_size)
        self.enable_neck = enable_neck
        self.neck_config = neck_config or NeckConfig()
        self.neck_ctrl = None
        self.use_neck = use_neck

    def connect(self):
        """Connect robot and all cameras."""
        self.robot.connect()
        for name, cam in self.cameras.items():
            print(f"Connecting camera '{name}'...")
            cam.connect()
        if self.enable_neck:
            self.neck_ctrl = DynamixelNeckController(self.neck_config, dry_run=False)
            # Static hold at the recorded home pose -- no headset to track against here.
            self.neck_ctrl.write(self.neck_config.home_pitch_rad, self.neck_config.home_yaw_rad)
            print("Neck torqued and holding home pose.")
        print("PiperRealEnv ready.")

    def disconnect(self):
        """Disconnect all devices."""
        for name, cam in self.cameras.items():
            cam.disconnect()
        self.robot.disconnect()
        if self.neck_ctrl is not None:
            self.neck_ctrl.close()
        print("PiperRealEnv disconnected.")

    def get_obs(self) -> tuple[dict[str, torch.Tensor], dict[str, 'np.ndarray']]:
        """Get current observation matching training format.

        For every connected camera, emit obs[f"observation.images.{cam_name}"].
        Single-arm setup: cam_name ∈ {"main", "wrist"}.
        Dual-arm setup:  cam_name ∈ {"left_main", "left_wrist", "right_wrist"}.

        Returns:
            obs: dict with policy input tensors
            raw_images: dict with raw camera images {cam_name: (H, W, 3) uint8 RGB}
        """
        obs = {}
        raw_images = {}

        for cam_name, camera in self.cameras.items():
            image = camera.get_image()  # (H, W, 3) uint8 RGB
            raw_images[cam_name] = image
            image_t = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            image_t = self._resize(image_t)
            obs[f"observation.images.{cam_name}"] = image_t

        state = self.robot.get_state()
        if self.neck_ctrl is not None and self.use_neck:
            import math
            pitch, yaw = self.neck_ctrl.read()
            neck_state = np.array([math.degrees(pitch), math.degrees(yaw)], dtype=state.dtype)
            state = np.concatenate([state, neck_state])
        obs["observation.state"] = torch.from_numpy(state)

        return obs, raw_images

    def step(self, action: np.ndarray, next_action: np.ndarray = None, skip_sleep: bool = False):
        """Send action to robot and return new observation.

        Args:
            action: (7,) joint positions + gripper in degrees for single-arm,
                or (16,) [14 arm dims + neck_pitch_deg + neck_yaw_deg] when
                neck is enabled and included in the action space.
            next_action: matching shape, next action for velocity estimation (optional).

        Returns:
            obs: New observation dict.
        """
        if self.neck_ctrl is not None and self.use_neck and len(action) == 16:
            arm_action = action[:14]
            neck_action_deg = action[14:16]
            arm_next = next_action[:14] if next_action is not None else None
            self.robot.send_action(arm_action, next_action=arm_next)
            pitch_rad = math.radians(float(neck_action_deg[0]))
            yaw_rad = math.radians(float(neck_action_deg[1]))
            self.neck_ctrl.write(pitch_rad, yaw_rad)
        else:
            self.robot.send_action(action, next_action=next_action)
        if not skip_sleep:
            time.sleep(self.dt)
        return self.get_obs()

    def reset(self):
        """Get initial observation (no environment reset for real robot)."""
        return self.get_obs()
