"""NERO 를 robosuite 로봇으로 등록한다 (양팔 고정 상체 + 2지 그리퍼).

**왜 이렇게 두나**: robosuite 는 `register_robot` / `register_gripper` 로 **런타임 등록**을
지원하고 `xml_path_completion` 이 절대경로를 그대로 통과시킨다 [코드 확인 2026-09-05].
그래서 robosuite 패키지를 건드리지 않고 이 repo 안에서만 정의할 수 있다 — manibot 하나만
clone 해도 돌아야 한다는 요건과 맞는다.

자산(`assets/robot.xml`·`assets/gripper.xml`)은 `build_assets.py` 가 원본 URDF 에서 생성한다.
손으로 고치지 말 것 — 고쳐야 하면 생성기를 고친다.
"""

import os

import numpy as np
from robosuite.models.grippers import register_gripper
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.controllers import load_composite_controller_config
from robosuite.robots import register_robot_class

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# 실측값 [메시·MJCF 실측 2026-09-04~05]
GRIPPER_OPEN = 0.05        # 손가락 하나당 여는 거리 -> 개구 0.10 m
SHOULDER_Z = 1.08          # 어깨축 높이 (받침대 포함)
# 왼팔 관절값 = 오른팔 * MIRROR. 부호조합 전수 탐색으로 확정 [2026-09-05].
MIRROR = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0])


@register_gripper
class NeroGripperBase(GripperModel):
    """손가락 두 개를 각각 지령하는 원형(2 dof)."""

    def __init__(self, idn=0):
        super().__init__(os.path.join(ASSETS, "gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        # 절반쯤 벌린 상태에서 시작한다 (Panda 도 같은 관례).
        return np.array([GRIPPER_OPEN / 2, -GRIPPER_OPEN / 2])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["leftfinger_collision"],
            "right_finger": ["rightfinger_collision"],
            "left_fingerpad": ["leftfinger_collision"],
            "right_fingerpad": ["rightfinger_collision"],
        }


@register_gripper
class NeroGripper(NeroGripperBase):
    """robosuite 표준대로 1 dof 로 줄인 판. -1 = 열기 / +1 = 닫기.

    두 손가락이 서로 반대 방향으로 움직이므로 `[-1, 1]` 을 곱해 부호를 갈라준다
    (Panda 와 같은 방식). 이걸 빼먹으면 한쪽 손가락만 움직인다.
    """

    def format_action(self, action):
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.2

    @property
    def dof(self):
        return 1


# 모델 등록은 ManipulatorModel 의 메타클래스가 자동으로 한다 — `register_robot` 데코레이터는
# 클래스를 반환하지 않아(None) 다른 데코레이터와 겹쳐 쓰면 클래스가 사라진다.
# 제어 클래스 매핑은 따로 필요하다. 실물이 볼트 고정이라 Baxter 와 같은 FixedBaseRobot.
@register_robot_class("FixedBaseRobot")
class Nero(ManipulatorModel):
    """WeGo NERO — 기둥형 몸체에 7축 팔 두 대를 단 고정 상체."""

    arms = ["right", "left"]

    def __init__(self, idn=0):
        super().__init__(os.path.join(ASSETS, "robot.xml"), idn=idn)

    @property
    def default_base(self):
        # 실물이 책상·받침대에 볼트로 고정된다. 이동 베이스가 아니므로 마운트를 두지 않는다.
        return "NullMount"

    @property
    def default_gripper(self):
        return {"right": "NeroGripper", "left": "NeroGripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_panda", "left": "default_panda"}

    @property
    def init_qpos(self):
        """작업 자세 — 손이 몸 앞 x~0.40 · z~0.83 에 오게 잡았다.

        관절공간 4만 개를 훑어 목표 영역(x .30~.50 · z .70~1.00)에 드는 자세 중
        **관절 한계 여유가 가장 큰 것**을 골랐다(최소 여유 31°) [탐색 2026-09-05].
        이전 자세는 팔을 옆으로 내린 모양이라 손이 z 0.42~0.67 에 갇혀 작업대에 안 닿았다.

        왼팔 미러는 부호조합 128 가지를 전수 탐색해 확정했다(y 대칭 오차 0.00 cm) —
        side mount 라 좌우 roll 이 반대여서 전부 뒤집으면 맞지 않는다.
        """
        right = np.radians([122.2, 62.6, 49.8, 77.4, 103.8, -10.8, -50.6])
        return np.concatenate([right, right * MIRROR])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, SHOULDER_Z))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        # build_assets.py 가 link7 밑에 만들어 두는 그리퍼 부착점 body 이름.
        return {"right": "right_hand", "left": "left_hand"}


def controller_config(controller: str = "BASIC") -> dict:
    """NERO 용 컨트롤러 설정.

    ⚠️ robosuite 기본값은 `input_ref_frame="base"` 인데, NERO 는 팔이 몸통 옆면에 ±90°
    돌아 붙어 있어 그 프레임에서는 **축이 뒤바뀐다** — `x+` 를 명령하면 말단이 −y 로 간다
    [실측 2026-09-05]. 고정 설치 로봇이므로 world 기준으로 두는 게 맞다.

    ⚠️ 양팔 action 배치는 `[right 6, left 6, right_grip 1, left_grip 1]` 이다.
    단일팔(Panda)의 `[pos3, ori3, grip1]` 을 그대로 적용하면 왼팔 명령이 밀려 들어간다.
    """
    cfg = load_composite_controller_config(controller=controller, robot="Nero")
    for arm in ("right", "left"):
        cfg["body_parts"][arm]["input_ref_frame"] = "world"
    return cfg


# 양팔 action 벡터에서 각 팔의 위치 delta 가 놓이는 자리 [실측 2026-09-05]
ARM_POS_SLICE = {"right": slice(0, 3), "left": slice(6, 9)}
ARM_ORI_SLICE = {"right": slice(3, 6), "left": slice(9, 12)}
GRIPPER_INDEX = {"right": 12, "left": 13}
