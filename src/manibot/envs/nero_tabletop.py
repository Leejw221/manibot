"""NeroTabletop — NERO 양팔이 작업대 앞에 고정된 최소 환경.

**왜 만드나**: robosuite 기본 환경들은 작업대가 z=0.8 이고 로봇 배치도 Panda 기준이라
NERO 팔이 테이블을 뚫는다. 실물 NERO 는 어깨 1.08 m(휴머노이드 G1 기준 — 머리가 없어
어깨로 맞췄다 [사용자 2026-09-05])에 **작업대 640 mm** 를 쓴다. 그 배치를 그대로 옮긴
최소 환경을 두고, 여기에 전자레인지 같은 fixture 를 얹어 가사 task 로 키운다.

지금 용도는 두 가지다: 그리퍼 파지 검증, 그리고 도달 범위 안에서 물체를 다룰 수 있는지.
"""

import os

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

TABLE_HEIGHT = 0.64        # [사용자 2026-09-05] 실물이 쓰는 작업대 높이 (mm 640)
TABLE_SIZE = (0.8, 1.2, 0.05)
# 로봇을 작업대에서 얼마나 뒤로 물릴지. 손이 base 기준 x≈0.42 까지 가므로(도달 실측),
# 작업대 앞 가장자리가 그 안에 들어오도록 잡는다.
BASE_X = -0.55

# 전자레인지 자산. robocasa 의 Microwave011 을 그대로 가져왔다(출처·라이선스는 README).
# robocasa 를 의존성으로 넣지 않고 MJCF 만 쓴다 — manibot 하나만 clone 해도 돌아야 하므로.
MICROWAVE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "assets", "microwave", "model.xml")


class Microwave(MujocoXMLObject):
    """문(`microjoint`, hinge -90°~0) + start/stop 버튼 geom + 회전 접시.

    작업대에 고정해 쓰므로 free joint 를 주지 않는다(joints=None).
    """

    def __init__(self, name="microwave"):
        super().__init__(MICROWAVE_XML, name=name, joints=None,
                         obj_type="all", duplicate_collision_geoms=False)


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


# 전자레인지 원점 기준 부품 위치 [MJCF 실측 2026-09-05]. 정면이 −y 를 향한다.
MW_HANDLE = np.array([0.162, -0.241, 0.001])
MW_START = np.array([0.209, -0.213, -0.125])
MW_STOP = np.array([0.265, -0.213, -0.124])
# 로봇은 −x 쪽에서 +x 를 본다. 정면(−y)을 로봇 쪽(−x)으로 돌리려면 z축 −90°.
MW_QUAT = np.array([0.70710678, 0.0, 0.0, -0.70710678])


class NeroMicrowave(NeroTabletop):
    """작업대 위 전자레인지 + 큐브. 가사 task(열기→넣기→닫기→버튼)의 바탕.

    전자레인지는 `microjoint`(hinge −90°~0)로 문이 열리고, start/stop 버튼은 geom
    접촉으로 눌림을 판정한다(robocasa 원래 방식).
    """

    def __init__(self, robots="Nero", microwave_pos=(0.04, -0.05, 0.151), **kwargs):
        self.microwave_pos = np.array(microwave_pos, dtype=float)
        super().__init__(robots=robots, **kwargs)

    def _load_model(self):
        super()._load_model()
        self.microwave = Microwave()
        pos = self.microwave_pos.copy()
        pos[2] += self.table_offset[2]          # 작업대 위에 올린다
        self.microwave.get_obj().set("pos", " ".join(f"{v:.4f}" for v in pos))
        self.microwave.get_obj().set("quat", " ".join(f"{v:.6f}" for v in MW_QUAT))
        self.model.merge_objects([self.microwave])

    def _setup_references(self):
        super()._setup_references()
        s = self.sim.model
        self.mw_door_joint = s.joint_name2id("microwave_microjoint")
        self.mw_handle_geom = s.geom_name2id("microwave_door_handle_main")
        self.mw_start_geom = s.geom_name2id("microwave_start_button")
        self.mw_stop_geom = s.geom_name2id("microwave_stop_button")

    def door_angle(self):
        return float(self.sim.data.qpos[self.sim.model.jnt_qposadr[self.mw_door_joint]])

    def is_open(self, th=-1.4):
        return self.door_angle() < th

    def is_closed(self, th=-0.05):
        return self.door_angle() > th

    def _check_success(self):
        return self.is_open()
