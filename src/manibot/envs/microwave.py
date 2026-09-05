"""MicrowaveTask — 전자레인지 가사 task: 문 열기 -> 물건 넣기 -> 문 닫기 -> 버튼 누르기.

**왜 만드나**: ETRI 자율성장 과제의 가사 task 를 시뮬로 먼저 돌려 하드웨어(NERO)로 수집·수행이
가능한지 본다. 다만 NERO 는 아직 파지 정밀도·자세가 검증 중이라, **같은 task 를 Franka 로 먼저**
세워 수집·학습 경로를 뚫는다. 그래서 이 환경은 로봇에 묶이지 않는다 — 배치 상수만 서브클래스가
덮어쓴다.

**자산**: robocasa 의 `Microwave052`(문 hinge · start/stop 버튼 · 회전 접시). robocasa 를
의존성으로 넣지 않고 MJCF 만 가져왔다 — manibot 하나만 clone 해도 돌아야 하므로.
50종 중 이걸 고른 이유는 실측 기준 두 가지다 [측정 2026-09-05]:
  - 깊이 0.329 m — 통과한 22종 중 가장 얕아 작업대에 공간이 남는다
  - start 버튼이 바닥에서 0.101 m — 낮으면(011 은 0.032 m) 그리퍼 몸통이 작업대에 걸려
    수평으로 누를 수 없다
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

from manibot.envs.ik import ArmIK, frame, roll, rot_z

MICROWAVE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "assets", "microwave", "model.xml")

# 전자레인지 정면은 자기 좌표계의 -y 를 향한다. 로봇은 -x 쪽에서 +x 를 보므로 z축 -90°.
FRONT_TO_MINUS_X = np.array([0.70710678, 0.0, 0.0, -0.70710678])

# 손잡이는 세로 막대라 개폐축이 수평이어야 잡힌다. 개폐축을 -y 로 두면 수평 접근이 닿고
# +y 면 손목 한계에 걸린다. roll -10° 는 문 여는 호 전체의 관절 여유를 재서 고른 값이다
# (여유 최소 7° -> 13.5°) [실측 2026-09-05].
HANDLE_ROLL = np.radians(-10.0)
HANDLE_FRAME = roll(frame((1, 0, 0), (0, -1, 0)), HANDLE_ROLL)
PICK_FRAME = roll(frame((0, 0, -1), (0, -1, 0)), HANDLE_ROLL)   # 개폐축을 그대로 두면
# 큐브를 집은 뒤 손목만 돌려 수평 삽입 자세로 갈 수 있다 (다시 잡을 필요 없음).


def _look_at(eye, target, up=(0, 0, 1)):
    """MuJoCo 카메라 자세(quat). 카메라는 자기 -z 를 보고 +y 가 위다."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-6:                       # 축각 경유(퇴화 방지)
        from robosuite.utils.transform_utils import mat2quat
        q = mat2quat(R)
        return np.array([q[3], q[0], q[1], q[2]])
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                     (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


def _rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Microwave(MujocoXMLObject):
    """문(`microjoint`, hinge -90°~0) + start/stop 버튼 + 회전 접시.

    작업대에 놓고 쓰므로 free joint 를 주지 않는다(joints=None) — 문을 당길 때 본체가
    끌려오면 task 가 성립하지 않는다.
    """

    def __init__(self, name="microwave"):
        super().__init__(MICROWAVE_XML, name=name, joints=None,
                         obj_type="all", duplicate_collision_geoms=False)


class MicrowaveTask(ManipulationEnv):
    """문 열기 -> 큐브 넣기 -> 문 닫기 -> start 버튼. 한 에피소드 안에 네 단계.

    성공 판정은 네 단계를 **순서대로** 통과했는지로 본다. 순서를 안 보면 문을 안 열고
    옆으로 큐브를 밀어 넣거나, 넣기 전에 버튼을 눌러도 성공이 된다.
    """

    # ── 배치 (서브클래스가 로봇에 맞춰 덮어쓴다) ─────────────────────────
    table_full_size = (0.8, 0.8, 0.05)
    table_offset = (0.0, 0.0, 0.8)
    base_xpos = None                 # None 이면 로봇의 table 기본 오프셋을 쓴다
    # 전자레인지 y 를 0.05 -> 0.10 으로 옮긴 이유는 팔이 문 여는 호를 지날 때의 관절 여유다
    # (최소 여유 7° -> 13.5°, IK 로 배치·roll 을 함께 훑어 고름 [2026-09-05]).
    microwave_xy = (0.16, 0.10)      # 작업대 중심 기준. z 는 상판 위에 얹는다
    cube_x_range = (-0.20, -0.12)
    cube_y_range = (-0.28, -0.20)
    cube_size = 0.022

    # 문 각도 [rad]. hinge 는 0(닫힘) ~ -pi/2(활짝).
    open_target = -1.45              # 스크립트 정책이 여는 목표
    open_threshold = -1.20
    closed_threshold = -0.10

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1500,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
    ):
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer
        self._ready_qpos = None
        super().__init__(
            robots=robots, env_configuration=env_configuration,
            controller_configs=controller_configs, base_types="default",
            gripper_types=gripper_types, initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs, has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer, render_camera=render_camera,
            render_collision_mesh=render_collision_mesh, render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id, control_freq=control_freq,
            lite_physics=lite_physics, horizon=horizon, ignore_done=ignore_done,
            hard_reset=hard_reset, camera_names=camera_names, camera_heights=camera_heights,
            camera_widths=camera_widths, camera_depths=camera_depths,
            camera_segmentations=camera_segmentations, renderer=renderer,
            renderer_config=renderer_config,
        )

    # ── 장면 ──────────────────────────────────────────────────────────────
    def _load_model(self):
        super()._load_model()
        model = self.robots[0].robot_model
        xpos = (self.base_xpos if self.base_xpos is not None
                else model.base_xpos_offset["table"](self.table_full_size[0]))
        model.set_base_xpos(xpos)

        arena = TableArena(table_full_size=self.table_full_size,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        # ⚠️ TableArena 의 기본 agentview 는 x=+0.5 에서 로봇을 마주 본다 — 그 자리는 지금
        # 전자레인지 **안쪽**이라 회색 벽만 찍힌다 [실측 2026-09-05]. 전자레인지 정면이
        # -x 를 향하므로 카메라는 옆(-y)에서 본다.
        for name, eye, tgt in (("agentview", (-0.72, -0.88, 1.50), (-0.02, 0.05, 0.92)),
                               ("frontview", (-0.85, -0.95, 1.60), (-0.05, 0.05, 0.93))):
            arena.set_camera(camera_name=name, pos=list(eye), quat=list(_look_at(eye, tgt)))

        tex = CustomMaterial(texture="WoodRed", tex_name="redwood", mat_name="redwood_mat",
                             tex_attrib={"type": "cube"},
                             mat_attrib={"texrepeat": "1 1", "specular": "0.4",
                                         "shininess": "0.1"})
        self.cube = BoxObject(name="cube", size=[self.cube_size] * 3,
                              rgba=[1, 0, 0, 1], material=tex)

        if self.placement_initializer is None:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler", mujoco_objects=self.cube,
                x_range=list(self.cube_x_range), y_range=list(self.cube_y_range),
                rotation=None, ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=np.array(self.table_offset), z_offset=0.01)
        else:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)

        self.microwave = Microwave()
        pos = np.array([self.microwave_xy[0], self.microwave_xy[1], self.table_offset[2]])
        # 전자레인지 원점은 몸통 중심이라 상판 위에 얹으려면 반높이만큼 올린다.
        pos[2] += self._microwave_half_height()
        self.microwave.get_obj().set("pos", " ".join(f"{v:.5f}" for v in pos))
        self.microwave.get_obj().set("quat", " ".join(f"{v:.6f}" for v in FRONT_TO_MINUS_X))

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=[self.cube, self.microwave])

    def _microwave_half_height(self):
        """MJCF 의 reg_main 영역에서 본체 반높이를 읽는다 (모델을 바꿔도 따라오게)."""
        for g in self.microwave.worldbody.iter("geom"):
            if g.get("name", "").endswith("reg_main"):
                return float(g.get("size").split()[2])
        raise RuntimeError("reg_main 영역이 없다 — 전자레인지 MJCF 를 확인할 것")

    # ── 참조 ──────────────────────────────────────────────────────────────
    def _setup_references(self):
        super()._setup_references()
        s = self.sim.model
        self.cube_body_id = s.body_name2id(self.cube.root_body)
        self.mw_door_jid = s.joint_name2id("microwave_microjoint")
        self.mw_door_qposadr = s.get_joint_qpos_addr("microwave_microjoint")
        self.mw_handle_gid = s.geom_name2id("microwave_door_handle_main")
        self.mw_start_gid = s.geom_name2id("microwave_start_button")
        self.mw_interior_gid = s.geom_name2id("microwave_reg_int")
        self._set_ready_qpos()
        # 버튼을 누른 주체가 그리퍼인지 확인하려면 로봇 geom 집합이 필요하다.
        self._robot_gids = {s.geom_name2id(n) for n in s.geom_names
                            if n.startswith(self.robots[0].robot_model.naming_prefix)
                            or n.startswith("gripper0")}

    def _set_ready_qpos(self):
        """리셋 자세를 "문 여는 호 전체를 여유 있게 지나는" 해로 바꾼다.

        robosuite OSC 는 영공간에서 리셋 자세를 목표로 당기므로, 여기를 바꾸면 손잡이로
        갈 때 관절이 한계로 끌려가는 것을 막을 수 있다. 기본 자세로는 joint5 가 하한
        -166° 에 붙어 문을 -24° 밖에 못 열었다 [실측 2026-09-05].
        """
        if self._ready_qpos is None:
            ik = ArmIK(self)
            poses = [(self.handle_at(t), rot_z(t) @ HANDLE_FRAME)
                     for t in np.linspace(0.0, self.open_target, 9)]
            found = ik.solve_path(poses)
            if found is None:
                return                       # 못 찾으면 로봇 기본 자세를 그대로 쓴다
            # 손잡이 바로 앞(대기점)을 리셋 자세로 — 첫 구간이 곧장 이어진다
            q, ep, _ = ik.solve(self.handle_at(0.0) - HANDLE_FRAME[:, 2] * 0.12,
                                HANDLE_FRAME, found[1][0])
            self._ready_qpos = q if ep < 0.01 else found[1][0]
        self.robots[0].init_qpos = np.array(self._ready_qpos)

    # ── 상태 질의 ─────────────────────────────────────────────────────────
    def door_angle(self):
        return float(self.sim.data.qpos[self.mw_door_qposadr])

    def is_open(self):
        return self.door_angle() < self.open_threshold

    def is_closed(self):
        return self.door_angle() > self.closed_threshold

    def handle_pos(self):
        return np.array(self.sim.data.geom_xpos[self.mw_handle_gid])

    def button_pos(self):
        return np.array(self.sim.data.geom_xpos[self.mw_start_gid])

    def cube_pos(self):
        return np.array(self.sim.data.body_xpos[self.cube_body_id])

    def hinge_pos(self):
        """경첩 축이 지나는 월드 좌표. 문이 이 점을 중심으로 도므로 각도와 무관하다."""
        b = self.sim.model.jnt_bodyid[self.mw_door_jid]
        return (np.array(self.sim.data.body_xpos[b])
                + np.array(self.sim.data.body_xmat[b]).reshape(3, 3)
                @ self.sim.model.jnt_pos[self.mw_door_jid])

    def handle_at(self, theta):
        """문이 각도 theta 일 때 손잡이가 갈 월드 좌표.

        스크립트 정책이 손잡이를 잡고 호를 따라 끌 때 쓴다. 지금 각도가 얼마든
        theta=0 기준 위치로 되돌린 뒤 다시 돌리므로 열려 있는 중에도 부를 수 있다.
        """
        h = self.hinge_pos()
        home = h + _rot_z(-self.door_angle()) @ (self.handle_pos() - h)
        return h + _rot_z(theta) @ (home - h)

    def interior_center(self):
        return np.array(self.sim.data.geom_xpos[self.mw_interior_gid])

    def interior_half(self):
        """내부 영역 반크기를 **월드 축** 기준으로 (전자레인지가 회전해 있으므로)."""
        R = self.sim.data.geom_xmat[self.mw_interior_gid].reshape(3, 3)
        return np.abs(R) @ self.sim.model.geom_size[self.mw_interior_gid]

    def cube_inside(self):
        """큐브 중심이 내부 영역(reg_int) 상자 안에 있는가.

        전자레인지는 회전해 놓으므로 월드 축으로 비교하면 안 된다 — geom 자세로 되돌린다.
        """
        c = self.sim.data.geom_xpos[self.mw_interior_gid]
        R = self.sim.data.geom_xmat[self.mw_interior_gid].reshape(3, 3)
        half = self.sim.model.geom_size[self.mw_interior_gid]
        local = R.T @ (self.cube_pos() - c)
        return bool(np.all(np.abs(local) < half))

    def _button_contact(self):
        d = self.sim.data
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if self.mw_start_gid == g1 and g2 in self._robot_gids:
                return True
            if self.mw_start_gid == g2 and g1 in self._robot_gids:
                return True
        return False

    # ── 진행·판정 ─────────────────────────────────────────────────────────
    def _reset_internal(self):
        super()._reset_internal()
        # robocasa 원본 문 관절에는 stiffness 0.05 스프링이 있어 놓으면 저절로 닫힌다
        # (실측: 열어 놓고 큐브를 집는 동안 -34° -> -19° 로 되돌아왔다 [2026-09-05]).
        # 실물 전자레인지 문은 놓은 자리에 서 있으므로 스프링을 없애고 마찰만 남긴다.
        self.sim.model.jnt_stiffness[self.mw_door_jid] = 0.0
        self.progress = dict(opened=False, inserted=False, closed=False, pressed=False)
        if not self.deterministic_reset:
            for pos, quat, obj in self.placement_initializer.sample().values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0], np.concatenate([np.array(pos), np.array(quat)]))

    def _update_progress(self):
        """단계는 **순서대로만** 올라간다 — 뒤 단계가 앞 단계를 건너뛰지 못하게."""
        p = self.progress
        if not p["opened"]:
            p["opened"] = self.is_open()
        elif not p["inserted"]:
            p["inserted"] = self.cube_inside()
        elif not p["closed"]:
            # 넣은 큐브가 문에 밀려 나오면 되돌린다 — "닫힘"만 만족시키는 편법 방지.
            p["inserted"] = self.cube_inside()
            p["closed"] = p["inserted"] and self.is_closed()
        elif not p["pressed"]:
            p["pressed"] = self._button_contact()

    def _post_action(self, action):
        self._update_progress()
        return super()._post_action(action)

    def reward(self, action=None):
        return self.reward_scale if self._check_success() else 0.0

    def _check_success(self):
        return bool(self.progress["pressed"])

    # ── 관측 ──────────────────────────────────────────────────────────────
    def _setup_observables(self):
        obs = super()._setup_observables()
        if not self.use_object_obs:
            return obs
        m = "object"

        @sensor(modality=m)
        def cube_pos(obs_cache):
            return self.cube_pos()

        @sensor(modality=m)
        def handle_pos(obs_cache):
            return self.handle_pos()

        @sensor(modality=m)
        def button_pos(obs_cache):
            return self.button_pos()

        @sensor(modality=m)
        def door_hinge(obs_cache):
            return np.array([self.door_angle()])

        @sensor(modality=m)
        def stage_flags(obs_cache):
            return np.array([float(v) for v in self.progress.values()])

        for s in (cube_pos, handle_pos, button_pos, door_hinge, stage_flags):
            obs[s.__name__] = Observable(name=s.__name__, sensor=s,
                                         sampling_rate=self.control_freq)
        return obs
