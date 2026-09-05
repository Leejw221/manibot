"""NeroTabletop — NERO 양팔이 작업대 앞에 고정된 최소 환경.

**왜 만드나**: robosuite 기본 환경들은 작업대가 z=0.8 이고 로봇 배치도 Panda 기준이라
NERO 팔이 테이블을 뚫는다. 실물 NERO 는 어깨 1.08 m(휴머노이드 G1 기준 — 머리가 없어
어깨로 맞췄다 [사용자 2026-09-05])에 **작업대 640 mm** 를 쓴다. 그 배치를 그대로 옮긴
최소 환경을 두고, 여기에 전자레인지 같은 fixture 를 얹어 가사 task 로 키운다.

지금 용도는 두 가지다: 그리퍼 파지 검증, 그리고 도달 범위 안에서 물체를 다룰 수 있는지.
"""

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

from manibot.envs.microwave import MicrowaveTask

TABLE_HEIGHT = 0.64        # [사용자 2026-09-05] 실물이 쓰는 작업대 높이 (mm 640)
TABLE_SIZE = (0.8, 1.2, 0.05)
# 로봇을 작업대에서 얼마나 뒤로 물릴지. 손이 base 기준 x≈0.42 까지 가므로(도달 실측),
# 작업대 앞 가장자리가 그 안에 들어오도록 잡는다.
BASE_X = -0.55

class NeroTabletop(ManipulationEnv):
    """작업대 위 큐브 하나. 성공 판정은 "집어서 들어올렸는가"."""

    def __init__(self, robots="Nero", env_configuration="default",
                 table_height=TABLE_HEIGHT, cube_size=0.025, lift_height=0.05, **kwargs):
        self.table_full_size = TABLE_SIZE
        self.table_offset = np.array([0.0, 0.0, table_height])
        self.cube_size = cube_size
        self.lift_height = lift_height
        self.placement_initializer = kwargs.pop("placement_initializer", None)
        super().__init__(robots=robots, env_configuration=env_configuration, **kwargs)

    # ── 장면 ──────────────────────────────────────────────────────────────
    def _load_model(self):
        super()._load_model()
        # 배치는 Panda 기준 기본값(base_xpos_offset)을 쓰지 않는다 — 그 값은 작업대 0.8 을
        # 전제해서 NERO 팔이 테이블을 뚫는다 [실측 2026-09-05].
        self.robots[0].robot_model.set_base_xpos([BASE_X, 0.0, 0.0])

        arena = TableArena(table_full_size=self.table_full_size,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])

        tex = CustomMaterial(texture="WoodRed", tex_name="redwood", mat_name="redwood_mat",
                             tex_attrib={"type": "cube"},
                             mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"})
        self.cube = BoxObject(name="cube", size=[self.cube_size] * 3,
                              rgba=[1, 0, 0, 1], material=tex)

        # 물체는 로봇 앞 도달 범위 안에 둔다 (base 기준 x 0.30~0.45).
        lo, hi = BASE_X + 0.30, BASE_X + 0.45
        if self.placement_initializer is None:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler", mujoco_objects=self.cube,
                x_range=[lo, hi], y_range=[-0.10, 0.10],
                rotation=None, ensure_object_boundary_in_range=False,
                ensure_valid_placement=True, reference_pos=self.table_offset, z_offset=0.01)
        else:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)

        self.model = ManipulationTask(mujoco_arena=arena,
                                      mujoco_robots=[r.robot_model for r in self.robots],
                                      mujoco_objects=self.cube)

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        obs = super()._setup_observables()
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality="object")
        def cube_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.cube_body_id])

        obs["cube_pos"] = Observable(name="cube_pos", sensor=cube_pos,
                                     sampling_rate=self.control_freq)
        return obs

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            for pos, quat, obj in self.placement_initializer.sample().values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(pos), np.array(quat)]))

    # ── 판정 ──────────────────────────────────────────────────────────────
    def reward(self, action=None):
        return float(self._check_success())

    def _check_success(self):
        """큐브 밑면이 작업대에서 lift_height 이상 떠 있으면 성공."""
        z = self.sim.data.body_xpos[self.cube_body_id][2]
        return z > self.table_offset[2] + self.cube_size + self.lift_height


class NeroMicrowave(MicrowaveTask):
    """NERO 양팔 + 작업대 위 전자레인지. 환경 본체는 `microwave.py` 와 한 벌을 쓴다.

    Franka 로 먼저 세운 그 task 를 그대로 옮기는 것이 목적이라 배치 상수만 덮어쓴다.
    """

    table_full_size = TABLE_SIZE
    table_offset = (0.0, 0.0, TABLE_HEIGHT)
    base_xpos = (BASE_X, 0.0, 0.0)
    microwave_xy = (0.10, 0.05)
    cube_x_range = (-0.20, -0.12)
    cube_y_range = (-0.28, -0.20)

    def _set_ready_qpos(self):
        # NERO 는 양팔이고 초기 자세를 이미 실측으로 잡아 두었다 (robots/nero/sim.py).
        # 단일팔 IK 로 덮어쓰지 않는다.
        pass
