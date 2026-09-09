"""NutAssemblySquare 를 푸는 스크립트 정책 — 개입 데이터 수집의 "사람" 역할.

**왜 스크립트인가**: 개입 시점은 사람이 트리거하되 교정 행동은 스크립트가 낸다.
PICO 텔레오퍼 없이도 배포 데이터를 모을 수 있고, 시드를 붙일 수 있어 재현된다.

**시간 파라미터화 궤적을 따른다** [2026-09-08 재작성]. 처음엔 목표를 고정해두고 P 제어로
쫓아가게 했는데, 수렴이 점근적이라 목표 근처에서 흔들리고 언제 끝날지 몰라 budget·settle·
glide 같은 임시방편이 붙었다. 그 흔들림이 그대로 시연에 남아 모방을 어렵게 한다.
지금은 **현재 자세에서 목표까지의 궤적을 먼저 만들고 그 위의 점을 매 스텝 명령**한다:
  - 연속한 명령점이 가까워서 OSC 가 바짝 따라온다 -> 흔들림이 안 생긴다
  - **코사인 이징**이라 시작·끝 속도가 0 이다 -> 급출발·급정지가 없다
  - 구간마다 **스텝 수로 속도를 직접 정한다** -> 이득(kp)을 올려 속도를 내지 않는다

**기하는 전부 실측으로 정했다** [2026-09-07, Panda + NutAssemblySquare]:
  - 그리퍼 좌표계: 접근축 = +eef_z, 개폐축 = eef_x (Panda). `envs/ik.frame(..., "x")`
  - 손잡이 geom(g4) 반치수 [0.0253, 0.0159, 0.010] -> **폭 31.8 mm**. Panda 최대 개구가
    41.6 mm 라 폭 방향으로만 물린다. 길이(50.6 mm) 방향으로 물면 안 들어간다.
  - ⭐ **손목 카메라는 eef +z 를 보고 eef +y 로 50 mm 치우쳐 있다** [실측]. 그래서
    **eef +y 가 손잡이->너트중심 을 향하게** 잡으면 너트 사각 구멍이 손목 화면 중앙에 오고
    **구멍 너머로 봉이 보인다** — 정렬이 어긋나면 봉이 구멍 한쪽에 치우쳐 보이고 맞으면
    정중앙에 온다.  이 자세가 동시에 개폐축을 손잡이 폭에 맞춘다.  `_grasp_frame` 참조.
  - 성공 판정은 너트 **body 원점** 기준이고 peg1 은 yaw 0 이다. 사각 구멍이라 너트 yaw 를
    90° 배수로 맞춰야 들어간다 (최대 45° 만 굴리면 된다).
  - 놓고 **4.24 cm 이상 물러나야** 성공으로 친다 (`_check_success` 의 r_reach < 0.6).

⚠ `hard_reset=True` 라 리셋마다 `env.sim` 객체가 교체된다. 절대 캐시하지 말 것.
"""

import numpy as np
from robosuite.utils.transform_utils import axisangle2quat, mat2quat, quat2axisangle, quat2mat

from manibot.envs.ik import frame, roll, rot_z

# ── 기하 ────────────────────────────────────────────────────────────────────
PRE = 0.09          # 손잡이 위 대기 높이 [m]
GRASP_DZ = 0.005    # 손잡이 site 보다 **얼마나 더 내려가서** 물나 [m].  손잡이 막대는 높이
                    # 20 mm(반치수 0.010)라 5 mm 는 막대 안이다.  site 높이 그대로 물면
                    # 손가락이 막대 윗변에 걸쳐 헐겁게 잡히고, 이송 중 너트가 그리퍼 안에서
                    # 미끄러져 삽입 직전 수평오차가 20 mm 대로 남는다 (2026-09-08 실측)
NUT_CLEAR = 0.06    # 이송 중 **너트**가 봉 꼭대기 위로 띄울 높이 [m].  eef 가 아니라 너트 기준이다
XY_TOL = 0.008      # 삽입 전에 맞춰야 할 너트-봉 수평 오차 [m]
YAW_TOL = np.radians(3.0)   # 삽입 전에 맞춰야 할 **너트 안쪽 사각형 - 봉 사각형** 각 오차 [rad].
                    # 사각 구멍이라 수평이 맞아도 각이 틀어지면 모서리에 걸린다 (사용자 지시
                    # 2026-09-08).  봉 yaw 는 0 이므로 너트 yaw 가 90° 배수면 맞는 것이다.
                    # ⚠ 3° 는 "정상 시행을 건드리지 않는 관문" 으로 고른 값이다 — 실제로 안
                    # 들어가기 시작하는 각은 안 쟀다.  [실측] 성공 시행의 삽입 직전 yaw 오차는
                    # 0.1~0.5° 라 이 관문은 통과하고, 그리퍼 안에서 돈 경우만 걸린다
ENGAGE = 0.04       # 봉 꼭대기 아래로 **너트 두께(0.02)의 2배**만 내려가면 물린 것으로 본다.
                    # 거기서 놓으면 나머지는 중력이 내려준다 — 끝까지 내릴 필요가 없다
RETREAT = 0.13      # 놓은 뒤 물러나는 높이 [m] (4.24 cm 이상이어야 성공 판정이 난다)
SIDE_TOL = np.sin(np.radians(15.0))   # 금지 배치 판정의 여유 — `u.(봉-베이스) < -SIDE_TOL`
                    # 일 때만 기각한다.  여유 없이 `< 0` 으로 자르면 거의 수직인 배치까지
                    # 걸러 90° 를 더 돌게 되는데, 그건 "너트가 베이스 쪽" 이라 할 배치가 아니다
J7_LIMIT = np.radians(164.0)   # 손목 마지막 관절(joint7, 범위 ±166°) 을 여기까지만 쓴다.
                    # ⚠ 여유를 크게 두면 작은 회전 후보가 잘려 크게 감아 도는 해가 뽑힌다 —
                    # 한계에 붙는 것만 막을 만큼(2°)이면 된다

# ── 구간별 스텝 수 = 속도. 이득이 아니라 여기로 속도를 정한다 ──────────────────
# **공개 square-ph 통계에 맞춘다** [실측 2026-09-08]: 길이 150.8 · 행동 std(위치) 0.42 ·
# 포화 11.3%.  우리 원래 값(316.8 / 0.24 / 0.0%)으로 학습하면 성공률이 6.7% 인데
# 공개 50개로는 63.3% 였다 — 개수가 아니라 **시연의 성질**이 갈랐다.
# 스텝 수를 줄이면 같은 거리를 짧은 시간에 가므로 행동 크기·포화가 같이 따라온다.
# ⚠ kp 를 올려 속도를 내면 안 된다 — 목표 주변에서 진동한다(2026-09-08 실측).
T_PRE = 25          # 홈 -> 손잡이 위
T_DOWN = 14         # 수직 하강 (짧고 정확하게)
T_SETTLE = 4        # OSC 추종 오차가 가라앉기를 기다린다 — 파지 직전에만
T_LIFT = 18         # 들어올리며 yaw 정렬까지 한 동작으로
T_PEG = 28          # 봉 위로 이송
T_INS = 25          # 삽입 하강 (물리면 조기 종료)
T_OUT = 14          # 물러나기
GRIP_BUDGET, GRIP_MIN, GRIP_EPS = 60, 15, 2e-4   # 파지 대기도 줄인다 (원래 25 는 여유가 컸다)


def _slerp(R0, R1, a):
    if a >= 1.0:
        return R1
    return quat2mat(axisangle2quat(a * quat2axisangle(mat2quat(R1 @ R0.T)))) @ R0


def _ease(a):
    """코사인 이징 — 시작·끝 속도가 0 이라 급출발·급정지가 없다."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(a, 0.0, 1.0))


def _yaw(R):
    return np.arctan2(R[1, 0], R[0, 0])


def _rot_dist(R, Rc):
    """현재 자세 Rc 에서 R 까지 돌아야 하는 각 [rad]."""
    return float(np.linalg.norm(quat2axisangle(mat2quat(R @ Rc.T)))) 


def _val(x):
    """값이면 그대로, 함수면 지금 계산해서 — 폐루프 목표를 값처럼 쓰기 위한 것."""
    return x() if callable(x) else x



class SquareExpert:
    """손잡이 정렬·파지 -> 들며 yaw 정렬 -> 봉 위 -> 물릴 때까지만 삽입 -> 놓고 물러나기."""

    def __init__(self, env, kp=6.0, kr=2.0, jitter=True):
        self.env = env
        self.kp, self.kr = kp, kr
        self.jitter = jitter
        self.reset()

    def reset(self):
        """에피소드마다 경유점·시간·파지지점을 흔든다 — **여전히 시연이다**(교란 주입이 아니다).

        [실측 2026-09-08] 흔들지 않으면 같은 진행률에서 eef 산포(관 굵기)가 24.9 mm 로
        공개 square-ph(43.9 mm)의 절반이고 길이 편차가 ±5.9(공개 ±20.3)다. 궤적이 가는 관
        하나만 덮으면 조금만 벗어나도 돌아올 정보가 데이터에 없다 — 일반화 오차 46.9% 의 원인.
        ⚠ **파지 프레임은 흔들지 않는다** — 개폐축이 손잡이 폭(31.8 mm)에 맞아야 물린다.
           봉에 맞출 방향은 `_choose_yaw` 가 너트·손잡이 방향으로 고른다.
        ⚠ 집는 구간만 흔들면 봉 접근·삽입이 그대로 과적합되므로 **전 구간에 넣는다**.
        """
        u = np.random.uniform
        j = self.jitter
        self.pre = u(0.07, 0.12) if j else PRE            # 손잡이 위 대기 높이
        self.clear = u(0.05, 0.10) if j else NUT_CLEAR      # 이송 시 너트 여유 (너트 기준)
        self.engage = u(0.030, 0.055) if j else ENGAGE      # 얼마나 물리면 놓나
        self.grab = u(-0.008, 0.008) if j else 0.0          # 손잡이 축 위 파지 지점 (막대 반길이 25mm)
        self.gz = u(0.003, 0.008) if j else GRASP_DZ        # 손잡이 site 아래로 내려가 무는 깊이
        self.tscale = u(0.85, 1.35) if j else 1.0           # 구간 시간 배율 -> 길이·속도 편차
        self.via = u(-0.05, 0.05, size=2) if j else np.zeros(2)  # 봉으로 가는 경유점 흔들기
        self._gen = self._plan()
        self.stage = "init"
        self.aborted = False       # 자세가 못 쓸 것이라 홀로 가지 않고 끊었나
        self.done = False

    def _T(self, t):
        return max(6, int(round(t * self.tscale)))

    # ── 상태 (sim 을 캐시하지 않는다 — hard_reset 이 객체를 갈아치운다) ──────────
    @property
    def sim(self):
        return self.env.sim

    def _eef(self):
        r = self.env.robots[0]
        s = r.eef_site_id[r.arms[0]]
        return (np.array(self.sim.data.site_xpos[s]),
                np.array(self.sim.data.site_xmat[s]).reshape(3, 3))

    def _handle(self):
        return np.array(self.sim.data.site_xpos[self.sim.model.site_name2id("SquareNut_handle_site")])

    def _nut(self):
        b = self.sim.model.body_name2id("SquareNut_main")
        q = np.array(self.sim.data.body_xquat[b])
        return np.array(self.sim.data.body_xpos[b]), quat2mat(np.r_[q[1:], q[0]])

    def _peg(self):
        return np.array(self.sim.data.body_xpos[self.sim.model.body_name2id("peg1")])

    def _peg_top(self):
        m, b = self.sim.model, self.sim.model.body_name2id("peg1")
        zs = [self.sim.data.geom_xpos[g][2] + m.geom_size[g][2]
              for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        return max(zs) if zs else 0.95

    def holding(self):
        """지금 손잡이를 물고 있나 — **개입 재진입**을 위한 판정.

        [실측 2026-09-08] 열린 상태 개구 0.078~0.080 · 손잡이를 문 상태 0.031,
        그때 eef-손잡이 거리는 3 mm 미만이다.
        """
        r = self.env.robots[0]
        q = np.array(self.sim.data.qpos[r._ref_gripper_joint_pos_indexes[r.arms[0]]])
        opening = float(q[0] - q[1])
        near = np.linalg.norm(self._eef()[0] - self._handle()) < 0.02
        return opening < 0.05 and near

    def _grasp_frame(self):
        """하향 파지 자세 — 개폐축(Panda 는 eef x)을 손잡이 폭에 맞춘다.

        손잡이는 u(손잡이->너트중심) 방향 막대고 물어야 할 폭 31.8 mm 는 u 에 수직이다
        (길이 50.6 mm 는 개구 41.6 mm 에 안 들어간다). 그런 자세는 접근축 둘레 180° 차이로
        **두 개**이고 파지에는 동등하다 — **지금 손목에서 가까운 쪽**을 골라 회전을 90° 이내로
        줄인다. 큰 회전을 명령하면 joint7 이 한계에 붙어 자세가 모자란 채 내려가 파지가
        깨진다 [실측 2026-09-08: 요구회전 135~180° 구간 파지 성공률 20%].

        ⚠ 손목 카메라가 너트 쪽을 보는 자세로 고정하지 않는다 — 그러면 회전이 최대 180° 가
        되어 파지가 무너진다.
        [검증 2026-09-08] 이 규칙으로 20/20 파지, 자세 잔차 평균 0.32°.
        """
        nut, _ = self._nut()
        u = nut - self._handle()
        u[2] = 0.0
        u /= np.linalg.norm(u)
        z = np.array([0.0, 0.0, -1.0])
        R = frame(z, np.cross(u, z), close_axis="x")
        _, Rc = self._eef()
        flip = roll(R, np.pi)
        return R if _rot_dist(R, Rc) <= _rot_dist(flip, Rc) else flip

    def _base(self):
        return np.array(self.sim.data.body_xpos[self.sim.model.body_name2id("robot0_base")])

    def _j7(self):
        r = self.env.robots[0]
        return float(self.sim.data.qpos[r._ref_joint_pos_indexes[6]])

    def _choose_yaw(self):
        """봉에 맞출 너트 yaw 를 고른다 — **금지 배치를 빼고** 회전이 가장 작은 것.

        사각 구멍이 4회 대칭이라 너트 yaw 는 90° 배수 넷 중 아무거나 맞으면 들어간다
        (봉 yaw 는 0 이다). 어느 배수를 고르냐에 따라 손잡이의 월드 방향이 90° 씩 달라진다.

        ⭐ **금지 배치** (사용자 지시 2026-09-08): 너트가 로봇 베이스 쪽이고 손잡이가 봉 쪽인
        정렬. 그러면 팔이 너트를 넘어 뻗어 agentview 에서 그리퍼가 화면 밖으로 나가고
        손목뷰도 막힌다. 판정: `u_end . (봉 - 베이스) <= 0` 이면 금지.

        같은 너트 방향은 d 로도 d-360° 로도 도달하므로 **감는 방향 셋을 모두** 본다 —
        한쪽만 보면 joint7 한계에 걸리는 것처럼 보인다.
        [검증 2026-09-09] 20 시행에서 20/20 이 금지 아닌 방향을 찾고 파지도 20/20 유지.
        """
        _, Rn = self._nut()
        phi = _yaw(Rn)
        nut, _ = self._nut()
        u = nut - self._handle()
        u[2] = 0.0
        u /= np.linalg.norm(u)
        want = self._peg()[:2] - self._base()[:2]
        j7 = self._j7()
        best = None
        for k in range(4):
            d0 = (k * np.pi / 2 - phi + np.pi) % (2 * np.pi) - np.pi
            c, sn = np.cos(d0), np.sin(d0)
            u_end = np.array([c * u[0] - sn * u[1], sn * u[0] + c * u[1]])
            if float(u_end @ want) < -SIDE_TOL * np.linalg.norm(want):
                continue                      # 명백히 베이스 쪽일 때만 뺀다 (SIDE_TOL 참조)
            for d in (d0, d0 - 2 * np.pi, d0 + 2 * np.pi):
                if abs(j7 + d) < J7_LIMIT and (best is None or abs(d) < abs(best)):
                    best = d
        if best is not None:
            return best
        # 금지 아닌 방향이 하나도 안 닿으면 가장 가까운 배수로 간다 (여기 오면 드문 경우다)
        d0 = -(phi % (np.pi / 2))
        return d0 + np.pi / 2 if d0 < -np.pi / 4 else d0

    # ── 저수준 ────────────────────────────────────────────────────────────
    def _action(self, pos, R, grip):
        p, Rc = self._eef()
        a = np.zeros(self.env.action_dim)
        a[:3] = np.clip(self.kp * (pos - p) / 0.05, -1.0, 1.0)
        a[3:6] = np.clip(self.kr * quat2axisangle(mat2quat(R @ Rc.T)) / 0.5, -1, 1)
        a[-1] = grip
        return a

    def _goto(self, pos, R, grip, name, steps):
        """현재 자세 -> (pos, R) 를 steps 에 걸쳐 매끄럽게 잇는다.

        명령점이 매 스텝 조금씩만 움직이므로 OSC 가 바짝 따라오고, 목표 근처에서
        헤매는 구간 자체가 생기지 않는다. 속도는 steps 로만 정한다.
        """
        self.stage = name
        p0, R0 = self._eef()
        pos = np.asarray(pos, float)
        for i in range(1, steps + 1):
            a = _ease(i / steps)
            yield self._action(p0 + (pos - p0) * a, _slerp(R0, R, a), grip)

    def _hold(self, pos, R, grip, n, name):
        self.stage = name
        for _ in range(n):
            yield self._action(pos, R, grip)

    def _descend_until(self, pos_fn, R, grip, name, steps, done_fn):
        """아래로 내리다 조건이 서면 끊는다 — 끝까지 내리지 않는다."""
        self.stage = name
        p0, _ = self._eef()
        for i in range(1, steps + 1):
            if done_fn():
                return
            tgt = np.asarray(pos_fn(), float)
            yield self._action(p0 + (tgt - p0) * _ease(i / steps), _val(R), grip)

    def _push_until(self, pos, R, grip, name, done_fn, budget=60):
        """pos·R 은 값이거나 매 스텝 다시 계산되는 함수다 (폐루프 보정용)."""
        """목표를 계속 명령하며 조건이 설 때까지 기다린다.

        ⚠ _goto 는 정해진 스텝 수 안에 명령점을 목표까지 옮기지만, 팔이 그 속도를 못 내면
        **낮은 데서 끝난다**. 이송 높이처럼 "실제로 도달했는가" 가 중요한 구간에는 관문을 둔다
        (2026-09-08: 개입 때 T_LIFT 18 스텝으로는 못 올라가 너트가 봉에 걸렸다).
        """
        self.stage = name
        for _ in range(budget):
            if done_fn():
                return
            yield self._action(np.asarray(_val(pos), float), _val(R), grip)

    def _grip(self, pos, R, grip, name, budget=GRIP_BUDGET):
        """손가락이 멈출 때까지 기다린다 — 고정 스텝으로 기다리면 덜 물린 채 끌게 된다."""
        self.stage = name
        r = self.env.robots[0]
        idx = r._ref_gripper_joint_pos_indexes[r.arms[0]]
        prev, still = None, 0
        for i in range(budget):
            yield self._action(pos, R, grip)
            q = np.array(self.sim.data.qpos[idx])
            still = still + 1 if prev is not None and np.abs(q - prev).max() < GRIP_EPS else 0
            prev = q
            if still >= 5 and i >= GRIP_MIN:
                break

    # ── 계획 ──────────────────────────────────────────────────────────────
    def _plan(self):
        """현재 상태를 보고 어디서부터 시작할지 고른다.

        개입은 **정책이 망쳐놓은 상태**에서 시작하므로 리셋 자세를 전제하면 안 된다.
        이미 물고 있으면 잡으러 가는 구간을 통째로 건너뛴다.
        """
        if self.holding():
            yield from self._place()                 # 이미 잡고 있다 — 그대로 이어서 꽂는다
            return
        yield from self._pick()
        yield from self._place()

    def _pick(self):
        up = np.array([0.0, 0.0, 1.0])
        R_grasp = self._grasp_frame()
        h = self._handle()

        # ① 손잡이 바로 위로 — 이동과 자세 맞춤을 한 궤적에 담는다
        nut, _ = self._nut()
        uu = nut - h; uu[2] = 0.0; uu /= np.linalg.norm(uu)      # 손잡이 축 방향
        grab = h + uu * self.grab                                 # 막대 위 파지 지점을 흔든다
        yield from self._goto(grab + up * self.pre, R_grasp, -1, "pre_grasp", self._T(T_PRE))
        # ② 수직 하강 후 정착 — 손잡이 폭이 31.8 mm 라 몇 mm 어긋나면 미끄러진다.
        #    site 높이가 아니라 **그보다 gz 만큼 아래**를 문다 (사용자 지시 2026-09-08):
        #    막대 윗변에 걸치지 않고 손가락 면 전체가 막대를 물어 이송 중 안 미끄러진다
        yield from self._goto(self._handle() + uu * self.grab - up * self.gz, R_grasp, -1,
                              "descend", self._T(T_DOWN))
        p, _ = self._eef()
        yield from self._hold(p, R_grasp, -1, self._T(T_SETTLE), "descend")
        # ③ 파지
        p, _ = self._eef()
        yield from self._grip(p, R_grasp, 1, "grasp")

    def _place(self):
        """⚠ 이송 높이는 **너트 기준**으로 잡는다.

        예전엔 eef 를 transit_z 로 올렸는데, 시연에서는 늘 같은 자세로 잡아 eef ~ 너트라
        (실측 eef 1.063 / 너트 1.061) 문제가 없었다. **개입에서는 정책이 잡은 자세라
        eef-너트 상대 높이가 다르다** — 너트가 5 cm 아래 매달려 있으면 eef 1.02 여도 너트는
        0.97 이라 봉 꼭대기 0.95 를 겨우 넘고 접근하다 걸린다 (2026-09-08 사용자 관찰).
        그래서 파지 직후 오프셋을 재서 **너트가 NUT_CLEAR 만큼 뜨도록** eef 목표를 올린다.
        """
        # 잡은 게 없으면 봉으로 갈 이유가 없다. **자세 판단은 여기서 하지 않는다** —
        # 어느 방향으로 정렬할지는 _choose_yaw 가 너트·손잡이 방향으로 고른다.
        if not self.holding():
            self.stage = "abort_no_grasp"
            self.aborted = True
            return

        up = np.array([0.0, 0.0, 1.0])
        # ④ 들면서 yaw 정렬을 **한 동작으로**. 궤적이 매끄러워 회전을 겹쳐도 헤매지 않는다.
        #    어느 90° 배수로 갈지는 _choose_yaw 가 금지 배치를 빼고 고른다.
        d = self._choose_yaw()
        # ⭐ **고른 목표를 절대각으로 잡아둔다.** 아래 정렬 폐루프가 "가장 가까운 90° 배수"
        # 를 쓰면, 드는 동안 목표까지 다 못 돌았을 때 **다른 배수로 끌려가** _choose_yaw 의
        # 선택(금지 배치 회피)이 무효가 된다 [실측 2026-09-09: 이송 시점에 u·ŵ 가 -0.45 까지].
        target_yaw = _yaw(self._nut()[1]) + d
        p, Rc = self._eef()
        R_hold = rot_z(d) @ Rc
        # 잡은 뒤엔 너트가 그리퍼에 고정이라 eef-너트 오프셋이 상수다 — 지금 재두고 계속 쓴다
        nut, _ = self._nut()
        off = p - nut
        top = self._peg_top()
        z_eef = top + self.clear + off[2]        # **너트**가 clear 만큼 뜨는 eef 높이 — 그 이상 안 올린다
        # 회전이 크면 스텝을 늘린다 (6°/스텝 이하). 이득을 올려 속도를 내지 않는다는 방침대로다
        lift_steps = max(self._T(T_LIFT), int(np.degrees(abs(d)) / 6))
        yield from self._goto([p[0], p[1], z_eef], R_hold, 1, "lift_align", lift_steps)
        # 진짜로 떴는지 확인하고 안 떴으면 더 올린다 — 여기서 못 뜨면 이송 중 봉에 걸린다
        yield from self._push_until([p[0], p[1], z_eef], R_hold, 1, "lift_align",
                                    lambda: self._nut()[0][2] > top + self.clear * 0.8, budget=60)

        # ⑤ 봉 위로
        peg = self._peg()
        p, _ = self._eef()
        # 봉으로 곧장 가지 않고 경유점을 하나 둔다 — 매번 같은 직선이면 그 구간이 과적합된다
        mid = [(p[0] + peg[0] + off[0]) / 2 + self.via[0],
               (p[1] + peg[1] + off[1]) / 2 + self.via[1], z_eef]
        yield from self._goto(mid, R_hold, 1, "to_peg", self._T(T_PEG) // 2)
        yield from self._goto([peg[0] + off[0], peg[1] + off[1], z_eef], R_hold, 1,
                              "to_peg", self._T(T_PEG) - self._T(T_PEG) // 2)
        # ⚠ **정렬을 확인하고 내려간다.** _goto 는 정해진 스텝만 쓰고 끝나서 수렴 전에
        # 삽입으로 넘어갈 수 있고, 그러면 너트가 봉 옆에 걸린다 (2026-09-08 사용자 관찰).
        # ⭐ **너트 실제 위치를 보고** 맞춘다. 파지 직후 잰 off 로 목표를 만들면 너트가 그리퍼
        # 안에서 조금만 미끄러져도 목표가 낡아 영영 못 맞춘다 (2026-09-08: 오차 0.0317m 로
        # 예산 소진). eef 를 "지금 남은 오차"만큼 옮기는 폐루프로 바꾼다.
        # ⭐ **수평만이 아니라 각도 관문도 둔다** (사용자 지시 2026-09-08): 너트 안쪽 사각형과
        # 봉 사각형이 맞아야 들어간다. yaw 는 지금까지 파지 직후 한 번만 맞춰 놓았을 뿐이라,
        # 이송 중 너트가 그리퍼 안에서 돌면 고칠 방법이 아예 없었다 — 여기서 닫는다.
        def _yaw_err():
            """**고른 목표각** 에서 얼마나 벗어났나 [rad]. 가장 가까운 배수가 아니다."""
            _, Rn = self._nut()
            return (_yaw(Rn) - target_yaw + np.pi) % (2 * np.pi) - np.pi

        def _aligned():
            return (np.linalg.norm(self._nut()[0][:2] - peg[:2]) < XY_TOL
                    and abs(_yaw_err()) < YAW_TOL)

        def _align_tgt():
            e, _ = self._eef()
            return np.r_[e[:2] + (peg[:2] - self._nut()[0][:2]), z_eef]

        def _align_R():
            # 남은 각오차만큼 손목을 되돌린다 — 명령을 쌓지 않고 **지금 너트 각**을 보고 낸다
            return rot_z(-_yaw_err()) @ self._eef()[1]

        yield from self._push_until(_align_tgt, _align_R, 1, "align_peg", _aligned, budget=90)
        # 관문을 지난 자세를 그대로 굳혀 내려간다 — 접촉 중에 손목을 더 돌리면 걸린다
        _, R_ins = self._eef()

        # ⑥ 물릴 때까지만 내린다
        # ⑥ **정렬을 유지한 채 수직으로만** 내린다. 여기서도 너트 실제 위치로 수평을 계속 보정한다
        yield from self._descend_until(
            lambda: np.r_[self._eef()[0][:2] + (peg[:2] - self._nut()[0][:2]),
                          top - self.engage - 0.02 + off[2]],
            R_ins, 1, "insert", self._T(T_INS),
            done_fn=lambda: self._nut()[0][2] < top - self.engage)

        # ⑦ 놓고 물러난다
        p, _ = self._eef()
        yield from self._grip(p, R_ins, -1, "release")
        p, _ = self._eef()
        yield from self._goto(p + up * RETREAT, R_ins, -1, "retreat", self._T(T_OUT))

    def act(self):
        try:
            return next(self._gen)
        except StopIteration:
            self.done = True
            self.stage = "done"
            p, R = self._eef()
            return self._action(p, R, -1)
