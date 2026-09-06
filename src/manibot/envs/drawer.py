"""DrawerTask — 서랍장 위의 블록을 서랍 안으로 옮긴다 (NERO 양팔 분업).

**왜 서랍인가**: 여닫이 문은 손잡이가 호를 그리며 좌우로 크게 움직여 NERO 가 못 연다
(전자레인지 실측: 손잡이가 x 0.345 m · y 0.244 m 이동, 한 팔의 쓸 만한 폭은 0.25 m).
서랍은 **직선 0.13 m** 이고 좌우 이동이 0 이다. 게다가 손잡이가 가로 막대라
**왼팔은 왼쪽 끝, 오른팔은 서랍 안 오른쪽**으로 갈라 쓸 수 있다 — NERO 는 팔이 몸 중앙을
못 넘으므로(오른팔 y<=+0.05) 이렇게 좌우로 갈리는 task 만 양팔로 성립한다.

단계: 왼팔 서랍 열기 -> 오른팔 블록 집기 -> 오른팔 서랍에 넣기 -> 왼팔 닫기 -> 복귀.
"""

import os

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor

from manibot.envs.ik import frame, rot_z
from manibot.envs.microwave import _look_at

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
# 서랍장 정면은 자기 좌표계 -y 를 향한다. 로봇이 -x 쪽이므로 z축 -90°.
FRONT_TO_MINUS_X = np.array([0.70710678, 0.0, 0.0, -0.70710678])

# 서랍장 국소 좌표의 부품 위치 [model.xml 과 같이 유지할 것]
HANDLE_LY, HANDLE_LZ = -0.225, 0.030     # 손잡이 (local y, z)
HANDLE_HALF_X = 0.200                    # 손잡이 반길이 (local x -> world y)
TOP_LZ = 0.110                           # 윗판 윗면 (local z)
INT_LZ, INT_HALF = -0.020, np.array([0.215, 0.130, 0.042])   # 서랍 내부 (local)
HALF_Z = 0.110                           # 서랍장 반높이


class Drawer(MujocoXMLObject):
    """서랍장. 작업대에 고정해 쓰므로 free joint 를 주지 않는다(당길 때 끌려오면 안 된다)."""

    def __init__(self, name="drawer"):
        super().__init__(os.path.join(ASSETS, "drawer", "model.xml"), name=name, joints=None,
                         obj_type="all", duplicate_collision_geoms=False)


class DrawerTask(ManipulationEnv):
    """서랍 열기 -> 블록 집기 -> 넣기 -> 닫기 -> 복귀. 순서대로 통과해야 성공."""

    table_full_size = (0.8, 1.2, 0.05)
    table_offset = (0.0, 0.0, 0.64)
    base_xpos = (-0.55, 0.0, 0.0)
    # [도달 탐색 2026-09-06] 인출 0.13 · (base+0.61, 0.02) 에서 최소 관절여유 15°
    drawer_xy = (0.06, 0.02)
    open_target = 0.13
    block_size = 0.022
    cameras = {"agentview": ((-0.95, -0.85, 1.35), (-0.05, 0.05, 0.85)),
               "frontview": ((-0.20, -1.15, 1.25), (-0.05, 0.05, 0.82))}

    open_threshold = 0.11
    closed_threshold = 0.02

    def __init__(self, robots, env_configuration="default", controller_configs=None,
                 gripper_types="default", initialization_noise="default",
                 use_camera_obs=True, use_object_obs=True, reward_scale=1.0,
                 reward_shaping=False, placement_initializer=None, has_renderer=False,
                 has_offscreen_renderer=True, render_camera="frontview",
                 render_collision_mesh=False, render_visual_mesh=True, render_gpu_device_id=-1,
                 control_freq=20, lite_physics=True, horizon=2000, ignore_done=False,
                 hard_reset=True, camera_names="agentview", camera_heights=256,
                 camera_widths=256, camera_depths=False, camera_segmentations=None,
                 renderer="mjviewer", renderer_config=None):
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer
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
            renderer_config=renderer_config)

    # ── 장면 ──────────────────────────────────────────────────────────────
    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(list(self.base_xpos))
        arena = TableArena(table_full_size=self.table_full_size, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        for name, (eye, tgt) in self.cameras.items():
            arena.set_camera(camera_name=name, pos=list(eye), quat=list(_look_at(eye, tgt)))

        tex = CustomMaterial(texture="WoodRed", tex_name="redwood", mat_name="redwood_mat",
                             tex_attrib={"type": "cube"},
                             mat_attrib={"texrepeat": "1 1", "specular": "0.4",
                                         "shininess": "0.1"})
        self.block = BoxObject(name="block", size=[self.block_size] * 3,
                               rgba=[1, 0, 0, 1], material=tex)
        self.drawer = Drawer()
        pos = np.array([self.drawer_xy[0], self.drawer_xy[1], self.table_offset[2] + HALF_Z])
        self.drawer.get_obj().set("pos", " ".join(f"{v:.5f}" for v in pos))
        self.drawer.get_obj().set("quat", " ".join(f"{v:.6f}" for v in FRONT_TO_MINUS_X))
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=[self.block, self.drawer])

    def _setup_references(self):
        super()._setup_references()
        s = self.sim.model
        self.block_body_id = s.body_name2id(self.block.root_body)
        self.slide_qposadr = s.get_joint_qpos_addr("drawer_slidejoint")
        self.handle_gid = s.geom_name2id("drawer_handle")
        self.int_gid = s.geom_name2id("drawer_reg_int")

    # ── 상태 질의 ─────────────────────────────────────────────────────────
    def slide(self):
        return float(self.sim.data.qpos[self.slide_qposadr])

    def is_open(self):
        return self.slide() > self.open_threshold

    def is_closed(self):
        return self.slide() < self.closed_threshold

    def handle_pos(self, along=0.0):
        """손잡이 위의 한 점. along 은 막대 방향(월드 y) 오프셋 — 왼팔은 +, 오른팔은 -."""
        return np.array(self.sim.data.geom_xpos[self.handle_gid]) + np.array([0.0, along, 0.0])

    def block_pos(self):
        return np.array(self.sim.data.body_xpos[self.block_body_id])

    def block_home(self):
        """리셋 때 블록을 놓는 자리 — 서랍장 윗면 오른쪽 (오른팔 영역)."""
        return np.array([self.drawer_xy[0], self.drawer_xy[1] - 0.18,
                         self.table_offset[2] + 2 * HALF_Z + self.block_size + 0.001])

    def drop_pos(self):
        """열린 서랍 안, 오른쪽에 넣을 자리."""
        c = np.array(self.sim.data.geom_xpos[self.int_gid])
        return np.array([c[0], self.drawer_xy[1] - 0.15, c[2] + 0.07])

    def block_inside(self):
        c = np.array(self.sim.data.geom_xpos[self.int_gid])
        R = self.sim.data.geom_xmat[self.int_gid].reshape(3, 3)
        half = self.sim.model.geom_size[self.int_gid]
        return bool(np.all(np.abs(R.T @ (self.block_pos() - c)) < half))

    def at_home(self, tol=0.10):
        """양팔이 초기 자세 근처로 돌아왔는가 (손 위치 기준)."""
        r = self.robots[0]
        return all(np.linalg.norm(np.array(self.sim.data.site_xpos[r.eef_site_id[a]])
                                  - self._home_eef[a]) < tol for a in r.arms)

    # ── 진행·판정 ─────────────────────────────────────────────────────────
    def _reset_internal(self):
        super()._reset_internal()
        self.sim.data.set_joint_qpos(
            self.block.joints[0], np.concatenate([self.block_home(), [1, 0, 0, 0]]))
        self.sim.forward()
        r = self.robots[0]
        self._home_eef = {a: np.array(self.sim.data.site_xpos[r.eef_site_id[a]]) for a in r.arms}
        self.progress = dict(opened=False, picked=False, inserted=False,
                             closed=False, returned=False)

    def _update_progress(self):
        p = self.progress
        if not p["opened"]:
            p["opened"] = self.is_open()
        elif not p["picked"]:
            p["picked"] = self.block_pos()[2] > self.block_home()[2] + 0.05
        elif not p["inserted"]:
            p["inserted"] = self.block_inside()
        elif not p["closed"]:
            p["inserted"] = self.block_inside()       # 닫다가 밀려 나오면 되돌린다
            p["closed"] = p["inserted"] and self.is_closed()
        elif not p["returned"]:
            p["returned"] = self.at_home()

    def _post_action(self, action):
        self._update_progress()
        return super()._post_action(action)

    def reward(self, action=None):
        return self.reward_scale if self._check_success() else 0.0

    def _check_success(self):
        return bool(self.progress["returned"])

    # ── 관측 ──────────────────────────────────────────────────────────────
    def _setup_observables(self):
        obs = super()._setup_observables()
        if not self.use_object_obs:
            return obs
        m = "object"

        @sensor(modality=m)
        def block_pos(obs_cache):
            return self.block_pos()

        @sensor(modality=m)
        def handle_pos(obs_cache):
            return self.handle_pos()

        @sensor(modality=m)
        def drawer_slide(obs_cache):
            return np.array([self.slide()])

        @sensor(modality=m)
        def stage_flags(obs_cache):
            return np.array([float(v) for v in self.progress.values()])

        for s in (block_pos, handle_pos, drawer_slide, stage_flags):
            obs[s.__name__] = Observable(name=s.__name__, sensor=s,
                                         sampling_rate=self.control_freq)
        return obs
