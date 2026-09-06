"""DrawerTask — 위층 서랍의 블록을 아래층 서랍으로 옮긴다 (NERO 양팔 분업).

순서 [사용자 2026-09-06]:
  ① 왼팔 위층 열기 → ② 오른팔 블록 꺼내기 → ③ 왼팔 위층 닫기
  → ④ 왼팔 아래층 열기 → ⑤ 오른팔 블록 넣기 → ⑥ 왼팔 아래층 닫기 → ⑦ 초기자세 복귀

**왜 서랍인가**: 여닫이 문은 손잡이가 호를 그리며 좌우로 크게 움직여 NERO 가 못 연다
(전자레인지 실측: 손잡이가 x 0.345 m · y 0.244 m 이동, 한 팔의 쓸 만한 폭은 0.25 m).
서랍은 **직선**이고 좌우 이동이 0 이다. 게다가 손잡이가 가로 막대라 왼팔은 왼쪽 끝을
잡고 오른팔은 서랍 안 오른쪽을 쓴다 — NERO 는 팔이 몸 중앙을 못 넘으므로(오른팔 y<=+0.05)
이렇게 좌우로 갈리는 task 만 양팔로 성립한다.
"""

import os

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

from manibot.envs.microwave import _look_at

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
# 서랍장 정면은 자기 좌표계 -y 를 향한다. 로봇이 -x 쪽이므로 z축 -90°.
FRONT_TO_MINUS_X = np.array([0.70710678, 0.0, 0.0, -0.70710678])

# 서랍장 국소 좌표 [model.xml 과 같이 유지할 것]
HALF_Z = 0.140                  # 서랍장 반높이
DRAWER_DZ = {"upper": 0.070, "lower": -0.070}   # 각 서랍 몸체의 국소 z
INT_DZ = 0.0                    # 서랍 몸체 기준 내부 중심 z
FLOOR_DZ = -0.037               # 서랍 몸체 기준 내부 바닥면 z
HANDLE_HALF_X = 0.220           # 손잡이 반길이 (local x -> world y)


class Block(MujocoXMLObject):
    """옮길 물체. 위에 좁은 목이 있어 **형상으로** 들린다 (자세한 이유는 model.xml)."""

    def __init__(self, name="block"):
        super().__init__(os.path.join(ASSETS, "block", "model.xml"), name=name,
                         joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)


class Drawer(MujocoXMLObject):
    """서랍장. 작업대에 고정해 쓰므로 free joint 를 주지 않는다(당길 때 끌려오면 안 된다)."""

    def __init__(self, name="drawer"):
        # duplicate_collision_geoms=True 여야 **보인다**. robosuite 는 group0=충돌 · group1=시각
        # 으로 갈라 렌더에서 group0 을 끄기 때문에, 끄면 서랍장이 통째로 안 그려진다
        # (사용자가 영상에서 "서랍이 없는데"로 잡아냈다) [2026-09-06].
        super().__init__(os.path.join(ASSETS, "drawer", "model.xml"), name=name, joints=None,
                         obj_type="all", duplicate_collision_geoms=True)


class DrawerTask(ManipulationEnv):
    """위층에서 꺼내 아래층에 넣기. 단계를 순서대로 통과해야 성공."""

    table_full_size = (0.8, 1.2, 0.05)
    table_offset = (0.0, 0.0, 0.64)
    base_xpos = (-0.55, 0.0, 0.0)
    # [도달 지도 실측 2026-09-06] 왼팔(y=+0.20) 손잡이 구간 x -0.28~-0.11 ·
    # 오른팔(y=-0.13) 블록 구간 x -0.09~0.00. 손잡이 닫힘(-0.135)·열림(-0.225) 과
    # 블록(-0.065) 이 모두 구간 안에 들려면 X=0.08 · 인출 0.09 다.
    drawer_xy = (0.08, 0.02)
    # 0.09 로 열면 물체가 x=-0.077 에 오는데, 그때 그리퍼 몸통(파지점 기준 x ±0.042)이
    # 서랍장 앞모서리(-0.035)와 **2 mm** 까지 붙어 물체를 17 mm 들다 막혔다
    # [실측 2026-09-06]. 0.12 면 여유 18 mm.
    open_target = 0.12
    open_target_lower = 0.15   # 아래층은 끝까지 — 블록 놓을 자리를 앞으로 뺀다
    # 아령 모양 물체(assets/block/model.xml): 몸통 45x45x20 · 목 30x30x44 · 머리 45x45x10.
    block_half_z = 0.030            # 물체 원점에서 바닥까지
    grip_up = 0.010                 # 물체 원점에서 **목 중앙**까지 (여기를 문다)
    # 서랍 안 **앞쪽 끝**(내부 앞벽에서 블록 반크기만큼만 안쪽). 가운데에 두면, 위에서
    # 내려오는 그리퍼 몸통(파지점 기준 x -0.033~+0.042)이 서랍장 앞모서리와 3mm 밖에
    # 안 떨어져 긁고, 그 힘으로 열어둔 서랍이 도로 닫혔다 [실측 2026-09-06].
    block_local_y = -0.055
    # 오른팔 도달 창 안쪽이면서 **서랍 오른벽(-0.202)에서 충분히 떨어진** 값.
    # -0.14 면 활짝 벌린 손가락 바깥면(파지 중심 67mm)이 벽과 15mm 겹쳐, 미리 좁히고
    # 내려가야 했다 — 그러면 "정렬보다 닫기가 먼저"인 부자연스러운 동작이 된다
    # [사용자 지적 2026-09-06]. -0.12 면 15mm 여유라 덜 좁히고 들어갈 수 있다.
    block_world_y = -0.12
    # 에피소드마다 블록을 이만큼 흔든다 [m]. 시연 데이터가 전부 같은 좌표면 정책이
    # 관측이 아니라 순서를 외운다 — 그래도 "성공"하므로 학습이 됐는지 알 수가 없다.
    # 범위는 좁게 잡았다: +x 는 블록이 서랍 앞턱으로 나가고, -x 는 위에서 내려오는
    # 그리퍼 몸통이 서랍장 앞모서리를 긁는다(둘 다 여유가 20 mm 이내) [실측 2026-09-06].
    # y 는 서랍 오른벽(-0.202)과 오른팔 도달 한계(+0.05) 사이라 더 여유가 있다.
    block_xy_noise = (0.010, 0.015)
    cameras = {"agentview": ((-0.95, -0.85, 1.35), (-0.05, 0.05, 0.85)),
               "frontview": ((-0.20, -1.15, 1.25), (-0.05, 0.05, 0.82))}

    # **정책 입력 카메라** [사용자 2026-09-06]: 사람 시점 + 양 손목. 로봇에 붙어 있어
    # 실물로 옮길 때 그대로 재현된다 (agentview 처럼 외부에 세운 카메라는 못 그런다).
    POLICY_CAMERAS = ("robot0_ego", "robot0_right_eye_in_hand", "robot0_left_eye_in_hand")
    # 수집 때 같이 저장만 하는 관찰용. 진행 상황을 사람이 보기 위한 것이지 정책 입력이 아니다.
    MONITOR_CAMERAS = ("agentview",)
    ALL_CAMERAS = POLICY_CAMERAS + MONITOR_CAMERAS

    open_threshold = 0.075
    closed_threshold = 0.02

    def __init__(self, robots, env_configuration="default", controller_configs=None,
                 gripper_types="default", initialization_noise="default",
                 use_camera_obs=True, use_object_obs=True, reward_scale=1.0,
                 reward_shaping=False, placement_initializer=None, has_renderer=False,
                 has_offscreen_renderer=True, render_camera="frontview",
                 render_collision_mesh=False, render_visual_mesh=True, render_gpu_device_id=-1,
                 control_freq=20, lite_physics=True, horizon=2000, ignore_done=False,
                 hard_reset=True, camera_names=None, camera_heights=256,
                 camera_widths=256, camera_depths=False, camera_segmentations=None,
                 renderer="mjviewer", renderer_config=None):
        # 기본값을 "정책 3대 + 관찰 1대" 로 둔다. 부르는 쪽이 매번 나열하지 않아도
        # 수집·평가가 같은 카메라 구성을 쓰게 하기 위해서다.
        if camera_names is None:
            camera_names = list(self.ALL_CAMERAS)
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

        self.block = Block()
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
        self.slide_adr = {k: s.get_joint_qpos_addr(f"drawer_{k}_slide")
                          for k in ("upper", "lower")}
        self.handle_gid = {"upper": s.geom_name2id("drawer_up_handle"),
                           "lower": s.geom_name2id("drawer_lo_handle")}
        self.int_gid = {"upper": s.geom_name2id("drawer_up_int"),
                        "lower": s.geom_name2id("drawer_lo_int")}

    # ── 상태 질의 ─────────────────────────────────────────────────────────
    def slide(self, which):
        return float(self.sim.data.qpos[self.slide_adr[which]])

    def is_open(self, which):
        return self.slide(which) > self.open_threshold

    def is_closed(self, which):
        return self.slide(which) < self.closed_threshold

    def handle_pos(self, which, along=0.0):
        """손잡이 위의 한 점. along 은 막대 방향(월드 y) 오프셋 — 왼팔은 +."""
        return np.array(self.sim.data.geom_xpos[self.handle_gid[which]]) + np.array([0, along, 0])

    def block_pos(self):
        return np.array(self.sim.data.body_xpos[self.block_body_id])

    def floor_z(self, which):
        """서랍 안 바닥면의 월드 z."""
        return self.table_offset[2] + HALF_Z + DRAWER_DZ[which] + FLOOR_DZ

    def rest_z(self, which):
        """그 서랍 안에 놓인 블록의 중심 높이."""
        return self.floor_z(which) + self.block_half_z

    def slot_pos(self, which):
        """그 서랍 안에서 블록이 놓이는 자리 (지금 인출량 기준)."""
        return np.array([self.drawer_xy[0] + self.block_local_y - self.slide(which),
                         self.block_world_y, self.rest_z(which)])

    def block_in(self, which):
        c = np.array(self.sim.data.geom_xpos[self.int_gid[which]])
        R = self.sim.data.geom_xmat[self.int_gid[which]].reshape(3, 3)
        half = self.sim.model.geom_size[self.int_gid[which]]
        return bool(np.all(np.abs(R.T @ (self.block_pos() - c)) < half))

    def at_home(self, tol=0.10):
        r = self.robots[0]
        return all(np.linalg.norm(np.array(self.sim.data.site_xpos[r.eef_site_id[a]])
                                  - self._home_eef[a]) < tol for a in r.arms)

    # ── 진행·판정 ─────────────────────────────────────────────────────────
    # MuJoCo 의 미끄럼 방지 후처리 반복 횟수. 기본 0 이면 그리퍼가 물체를 물고 있어도
    # 마찰이 거의 전달되지 않는다 — 손잡이를 15N 으로 물고 당겨도 당김 힘이 0.25N 밖에
    # 안 걸리고(마찰 한계는 55N) 패드가 막대 위를 미끄러졌다 [실측 2026-09-06].
    # MuJoCo 이슈 #786 에서도 같은 증상에 이 값을 권한다. 2 면 10.45 -> 11.84cm.
    # ⚠️ 같이 시험한 다른 처방은 오히려 나빴다: timestep 5e-4 · solimp "0.99 0.999 0.001"
    # 둘 다 서랍이 아예 안 열렸다.
    noslip_iterations = 2

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.model._model.opt.noslip_iterations = self.noslip_iterations
        d = np.zeros(3)
        if not self.deterministic_reset:
            nx, ny = self.block_xy_noise
            d[:2] = np.random.uniform([-nx, -ny], [nx, ny])
        self.block_offset = d
        self.sim.data.set_joint_qpos(
            self.block.joints[0],
            np.concatenate([self.slot_pos("upper") + d + [0, 0, 0.001], [1, 0, 0, 0]]))
        self.sim.forward()
        r = self.robots[0]
        self._home_eef = {a: np.array(self.sim.data.site_xpos[r.eef_site_id[a]]) for a in r.arms}
        self.progress = dict(opened_u=False, picked=False, closed_u=False, opened_l=False,
                             inserted=False, closed_l=False, returned=False)

    def _update_progress(self):
        p = self.progress
        if not p["opened_u"]:
            p["opened_u"] = self.is_open("upper")
        elif not p["picked"]:
            p["picked"] = self.block_pos()[2] > self.rest_z("upper") + 0.04
        elif not p["closed_u"]:
            p["closed_u"] = self.is_closed("upper")
        elif not p["opened_l"]:
            p["opened_l"] = self.is_open("lower")
        elif not p["inserted"]:
            p["inserted"] = self.block_in("lower")
        elif not p["closed_l"]:
            p["inserted"] = self.block_in("lower")      # 닫다가 밀려 나오면 되돌린다
            p["closed_l"] = p["inserted"] and self.is_closed("lower")
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
        def handle_pos_upper(obs_cache):
            return self.handle_pos("upper")

        @sensor(modality=m)
        def handle_pos_lower(obs_cache):
            return self.handle_pos("lower")

        @sensor(modality=m)
        def drawer_slide(obs_cache):
            return np.array([self.slide("upper"), self.slide("lower")])

        @sensor(modality=m)
        def stage_flags(obs_cache):
            return np.array([float(v) for v in self.progress.values()])

        for s in (block_pos, handle_pos_upper, handle_pos_lower, drawer_slide, stage_flags):
            obs[s.__name__] = Observable(name=s.__name__, sensor=s,
                                         sampling_rate=self.control_freq)
        return obs
