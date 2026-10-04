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
from manibot.envs.scripted_expert import ScriptedExpert, fit_T, qeval, quintic

# ── 기하 ────────────────────────────────────────────────────────────────────
PRE = 0.075         # 손잡이 위 대기 높이 [m].  0.09 에서 낮췄다 — 같은 스텝으로 내려오면
                    # 거리가 길수록 빨라져 책상에 부딪혔다 [영상 관찰 2026-09-19]
GRAB_OUT = -0.0075  # 파지 지점을 손잡이 site 에서 **너트 반대쪽**으로 옮기는 거리 [m].
                    # 고리 밖 자유 구간(35.6mm)의 중심에 손가락을 놓기 위한 값이다
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
T_DOWN = 18         # 수직 하강.  14 -> 18. 대기 높이를 0.095 -> 0.075 로 낮춘 것과 합쳐
                    # 하강 속도가 약 0.61 배가 된다 (거리 x0.79 / 스텝 x1.29)
T_SETTLE = 4        # OSC 추종 오차가 가라앉기를 기다린다 — 파지 직전에만
T_LIFT = 18         # 들어올리며 yaw 정렬까지 한 동작으로
T_PEG = 28          # 봉 위로 이송
T_INS = 25          # 삽입 하강 (물리면 조기 종료)
T_OUT = 14          # 물러나기
GRIP_BUDGET, GRIP_MIN, GRIP_EPS = 60, 15, 2e-4   # 파지 대기도 줄인다 (원래 25 는 여유가 컸다)
# 폐루프 보정(_servo_until)의 속도 상한 — 남은 오차를 한 번에 명령하면 kp 비례라 오차 8 mm ·
# 회전 14° 만 넘어도 행동이 ±1 에 붙어 정렬이 튄다 [사용자 관찰 2026-09-17]
# ⚠ 구간 경계의 급변(직전 궤적 끝에서 팔이 뒤처진 만큼 명령점이 되돌아감)은 남아 있다.
#   새 구간을 직전 명령점에서 이어가게 해봤으나 추종 지연이 누적돼 포화가 늘었고(정렬 27%),
#   정지·파지까지 이으면 파지가 깨졌다(성공 30 -> 6) [측정 2026-09-17]. 그래서 되돌렸다.
V_POS, V_ROT, SERVO_MIN = 0.002, np.radians(2.0), 4

# ── 연속 실행 (continuous=True) [사용자 지시 2026-10-01] ─────────────────────────
# 위 방식은 구간마다 시작·끝 속도가 0 이고(코사인 이징) 시간이 스텝 수로 고정이라 개입이
# 경유점마다 서서 평균 149 스텝이 걸렸다(사람 시연 전체 150). 최고 속도는 사람과 비슷하다
# [실측 2026-10-01: eef p90 사람 13 · 스크립트 12 mm/step] — 느린 건 멈춤과 고정 대기다.
# 그래서 **궤적을 만들고 C_PERIOD 스텝 실행한 뒤 실제 상태에서 다시 만든다** (정책이 16 을
# 예측하고 8 을 실행하는 것과 같은 꼴). 경유점은 멈추지 않고 지나가고(끝 속도를 다음 경유점
# 쪽으로), 시간은 거리 / 속도 상한으로 정한다. 5차 다항식은 실물 Piper 보간과 같은 식이다
# (Cashier_policy-dp `robots/piper_robot.py:113` _quintic_coeffs, 가속도 양끝 0).
C_PERIOD = 6                    # 다시 계획하는 주기 [step] — 실물과 제어 방식을 맞춘다 [사용자 2026-10-01]:
                                # 실물은 request period 5~6 으로 돌린다. 추론 지연까지 더하면 새 청크가
                                # 들어오기 전에 이전 청크를 8 개 정도 쓰게 되는 값이다
C_V_MAX = 0.013                 # 위치 속도 상한 [m/step] — 사람 시연 eef p90 13 mm/step (중앙 4.8)
C_W_MAX = np.radians(5.0)       # 회전 속도 상한 [rad/step] — 기존 들기 구간 평균 6°/step 보다 약간 낮게
C_V_INS = 0.010                 # 삽입 하강 속도 상한 [m/step]
C_PASS = 0.015                  # 경유점을 지난 것으로 보는 거리 [m]
C_STOP_V = 0.001                # 마지막 경유점 도착 = C_PASS 안에서 이만큼 느려졌다 [m/step].
                                # 거리만 보면 안 된다 — 손가락이 손잡이에 닿으면 목표 3 mm 안으로 못 들어가
                                # 하강에서 예산을 다 썼다 [측정 2026-10-01]
C_T_MIN = 3                     # 한 계획의 최소 길이 [step]
C_GRIP_MOVED, C_GRIP_STILL = 0.003, 3   # 그리퍼: 이만큼 움직였다가 이 스텝 동안 멈추면 끝


def _slerp(R0, R1, a):
    if a >= 1.0:
        return R1
    return quat2mat(axisangle2quat(a * quat2axisangle(mat2quat(R1 @ R0.T)))) @ R0


def _ease(a):
    """코사인 이징 — 시작·끝 속도가 0 이라 급출발·급정지가 없다."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(a, 0.0, 1.0))


# 5차 다항식 · 시간 맞춤은 공통 엔진(`scripted_expert`)으로 옮겼다 — 이름은 그대로 둔다(연속 모드가 쓴다)
_quintic, _qeval, _fit_T = quintic, qeval, fit_T


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

    def __init__(self, env, kp=6.0, kr=2.0, jitter=True, continuous=False, v_max=C_V_MAX):
        self.env = env
        self.kp, self.kr = kp, kr
        self.jitter = jitter
        self.continuous = continuous     # True = 멈추지 않는 연속 실행 (C_* 주석 참고)
        self.v_max = v_max
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
        self.pre = u(0.06, 0.09) if j else PRE            # 손잡이 위 대기 높이
        self.clear = u(0.05, 0.10) if j else NUT_CLEAR      # 이송 시 너트 여유 (너트 기준)
        self.engage = u(0.030, 0.055) if j else ENGAGE      # 얼마나 물리면 놓나
        # 손잡이 site 는 막대 **전체** 중심(몸체 x=0.054)이라 사각 고리 끝(x=0.0437)에서
        # 10.3mm 밖에 안 떨어져 있다. 여기서 +8mm 흔들면 고리에서 2.3mm — 손가락이 너트를
        # 누른다 [영상 관찰 2026-09-19]. 고리 밖 자유 구간은 x [0.0437, 0.0793] 이고 그
        # 중심이 0.0615 이므로, site 에서 **너트 반대쪽으로 7.5mm** 옮긴 자리를 기준으로 삼는다.
        self.grab = u(-0.012, -0.003) if j else GRAB_OUT   # 손잡이 축 위 파지 지점 (+ 가 너트 쪽)
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

    def _servo_until(self, pos, R, grip, name, done_fn, budget=60):
        """_push_until 과 같은 폐루프지만 남은 오차를 **이징 궤적으로 나눠** 따라간다.

        구간마다 목표를 다시 재고(너트 실제 위치 보정은 그대로), 오차 크기로 스텝 수를 정해
        V_POS·V_ROT 을 넘지 않게 한다. 조건이 서면 구간 중간이라도 끊는다.
        """
        self.stage = name
        used = 0
        while used < budget:
            if done_fn():
                return
            p0, R0 = self._eef()
            tgt, Rt = np.asarray(_val(pos), float), _val(R)
            n = int(np.ceil(max(np.linalg.norm(tgt - p0) / V_POS, _rot_dist(Rt, R0) / V_ROT)))
            n = min(max(n, SERVO_MIN), budget - used)
            for i in range(1, n + 1):
                if done_fn():
                    return
                a = _ease(i / n)
                yield self._action(p0 + (tgt - p0) * a, _slerp(R0, Rt, a), grip)
                used += 1

    def _grip(self, pos, R, grip, name, budget=GRIP_BUDGET):
        """손가락이 멈출 때까지 기다린다 — 고정 스텝으로 기다리면 덜 물린 채 끌게 된다."""
        self.stage = name
        r = self.env.robots[0]
        idx = r._ref_gripper_joint_pos_indexes[r.arms[0]]
        q0 = np.array(self.sim.data.qpos[idx])
        prev, still = None, 0
        for i in range(budget):
            yield self._action(pos, R, grip)
            q = np.array(self.sim.data.qpos[idx])
            still = still + 1 if prev is not None and np.abs(q - prev).max() < GRIP_EPS else 0
            prev = q
            if self.continuous:
                # 최소 대기(GRIP_MIN) 대신 "움직였다가 멈췄다" — 명령 직후엔 손가락이 아직
                # 안 움직여서 멈춤만 보면 너무 일찍 끝난다
                if np.abs(q - q0).max() > C_GRIP_MOVED and still >= C_GRIP_STILL:
                    break
            elif still >= 5 and i >= GRIP_MIN:
                break

    def _cfollow(self, wps, grip, final=None, budget=200, v_max=None):
        """경유점들을 멈추지 않고 지나간다 — 궤적을 만들고 C_PERIOD 스텝 실행한 뒤 다시 만든다.

        wps: [(이름, pos, R, 통과판정 | None), ...]. pos·R 은 값이거나 함수다(계획마다 실제
        상태로 다시 잰다 — 너트 위치 보정이 여기로 들어온다). 통과판정이 없으면 C_PASS 안에
        들면 다음 경유점으로 넘어간다. 마지막 경유점은 final() 이 참이 될 때까지, final 이
        없으면 C_PASS 안에서 느려질 때까지 붙든다. 통과·도착은 주기 중간이라도 바로 처리한다.

        ⚠ 다시 계획할 때 출발점은 **실제 eef 의 위치·속도**다. 직전 명령점에서 이어가면 추종
        지연이 쌓여 포화가 늘고 파지가 깨졌다 [측정 2026-09-17, V_POS 주석].
        """
        v_max = v_max or self.v_max
        k, used = 0, 0
        p_prev, R_prev = self._eef()
        while used < budget:
            name, pos, R, passed = wps[k]
            self.stage = name
            last = k == len(wps) - 1
            p, Rc = self._eef()
            tgt, Rt = np.asarray(_val(pos), float), _val(R)
            if not last and (passed() if passed else np.linalg.norm(tgt - p) < C_PASS):
                k += 1
                continue
            v = p - p_prev
            if last and (final() if final else (np.linalg.norm(tgt - p) < C_PASS
                                                and np.linalg.norm(v) < C_STOP_V)):
                return
            vf = np.zeros(3)
            if not last:                       # 다음 경유점 쪽으로 지나간다
                d = np.asarray(_val(wps[k + 1][1]), float) - tgt
                n = np.linalg.norm(d)
                if n > 1e-6:
                    vf = d / n * min(0.5 * v_max, n / C_T_MIN)
            w = quat2axisangle(mat2quat(Rc @ R_prev.T))
            r = quat2axisangle(mat2quat(Rt @ Rc.T))
            T = max(_fit_T(p, v, tgt, vf, v_max), _fit_T(np.zeros(3), w, r, np.zeros(3), C_W_MAX))
            cp = _quintic(p, v, tgt, vf, T)
            cr = _quintic(np.zeros(3), w, r, np.zeros(3), T)
            for i in range(1, C_PERIOD + 1):
                t = min(i, T)
                p_prev, R_prev = self._eef()
                yield self._action(_qeval(cp, t), quat2mat(axisangle2quat(_qeval(cr, t))) @ Rc, grip)
                used += 1
                pe, _ = self._eef()
                if (t >= T or (last and final and final())
                        or (not last and (passed() if passed else np.linalg.norm(tgt - pe) < C_PASS))):
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
        if self.continuous:
            # 손잡이 위 -> 수직 하강을 멈추지 않고 잇는다. 정착 대기(T_SETTLE) 대신 "가깝고
            # 느려지면" 문다. 무는 높이(gz)는 아래 ② 와 같다
            yield from self._cfollow([
                ("pre_grasp", lambda: self._handle() + uu * self.grab + up * self.pre, R_grasp, None),
                ("descend", lambda: self._handle() + uu * self.grab - up * self.gz, R_grasp, None),
            ], -1)
        else:
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
        lifted = lambda: self._nut()[0][2] > top + self.clear * 0.8   # noqa: E731
        lift_pt = [p[0], p[1], z_eef]
        peg = self._peg()
        if not self.continuous:
            lift_steps = max(self._T(T_LIFT), int(np.degrees(abs(d)) / 6))
            yield from self._goto(lift_pt, R_hold, 1, "lift_align", lift_steps)
            # 진짜로 떴는지 확인하고 안 떴으면 더 올린다 — 여기서 못 뜨면 이송 중 봉에 걸린다
            yield from self._servo_until(lift_pt, R_hold, 1, "lift_align", lifted, budget=60)

            # ⑤ 봉 위로
            p, _ = self._eef()
        # 봉으로 곧장 가지 않고 경유점을 하나 둔다 — 매번 같은 직선이면 그 구간이 과적합된다
        mid = [(p[0] + peg[0] + off[0]) / 2 + self.via[0],
               (p[1] + peg[1] + off[1]) / 2 + self.via[1], z_eef]
        if not self.continuous:
            # 앞 40% 로 멀리 가고 뒤 60% 로 천천히 붙는다 — 반반으로 나누면 봉 근처에서도 속도가
            # 남아 팔이 명령점을 못 따라잡고, 다음 정렬 구간이 **뒤처진 실제 위치에서 다시 시작**해
            # 화면에서 멈췄다가 튀는 것으로 보인다 [영상 관찰 2026-09-19]
            n_far = int(self._T(T_PEG) * 0.4)
            yield from self._goto(mid, R_hold, 1, "to_peg", n_far)
            yield from self._goto([peg[0] + off[0], peg[1] + off[1], z_eef], R_hold, 1,
                                  "to_peg", self._T(T_PEG) - n_far)
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

        if self.continuous:
            # 들기 -> 경유점 -> 봉 위 정렬을 멈추지 않고 잇는다. 마지막 경유점은 너트 실제 위치로
            # 매 계획마다 다시 잰다(= 아래 정렬 폐루프). 들기는 너트가 실제로 떠야 지난다
            yield from self._cfollow([
                ("lift_align", lift_pt, R_hold, lifted),
                ("to_peg", mid, R_hold, None),
                ("align_peg", _align_tgt, _align_R, None),
            ], 1, final=_aligned, budget=300)
        else:
            yield from self._servo_until(_align_tgt, _align_R, 1, "align_peg", _aligned, budget=90)
        # 관문을 지난 자세를 그대로 굳혀 내려간다 — 접촉 중에 손목을 더 돌리면 걸린다
        _, R_ins = self._eef()

        # ⑥ 물릴 때까지만 내린다
        # ⑥ **정렬을 유지한 채 수직으로만** 내린다. 여기서도 너트 실제 위치로 수평을 계속 보정한다
        ins_tgt = lambda: np.r_[self._eef()[0][:2] + (peg[:2] - self._nut()[0][:2]),  # noqa: E731
                                top - self.engage - 0.02 + off[2]]
        engaged = lambda: self._nut()[0][2] < top - self.engage   # noqa: E731
        if self.continuous:
            yield from self._cfollow([("insert", ins_tgt, R_ins, None)], 1, final=engaged,
                                     budget=80, v_max=C_V_INS)
        else:
            yield from self._descend_until(ins_tgt, R_ins, 1, "insert", self._T(T_INS),
                                           done_fn=engaged)

        # ⑦ 놓고 물러난다
        p, _ = self._eef()
        yield from self._grip(p, R_ins, -1, "release")
        p, _ = self._eef()
        if self.continuous:
            yield from self._cfollow([("retreat", p + up * RETREAT, R_ins, None)], -1)
        else:
            yield from self._goto(p + up * RETREAT, R_ins, -1, "retreat", self._T(T_OUT))

    def act(self):
        try:
            return next(self._gen)
        except StopIteration:
            self.done = True
            self.stage = "done"
            p, R = self._eef()
            return self._action(p, R, -1)


# ── 개입용: 궤적을 만들어 두고 따라가기 [사용자 2026-10-04] ─────────────────────────
HANDLE_MOVED = 0.005    # 집으러 가는 중 손잡이가 이만큼 움직이면 다시 만든다 [m]. 저장된 초기 상태는
                        # 너트가 z 0.89 에 떠 있다가 첫 몇 스텝에 0.83 으로 떨어져 놓인다 [실측 2026-10-03]
ALIGN_TRIES = 4         # 봉 위 정렬 보정 횟수 상한 — 넘으면 그대로 삽입한다


class SquareScriptedExpert(ScriptedExpert):
    """square 에서 **어떤 궤적을 언제 만드나** — 움직이는 방식은 `scripted_expert.ScriptedExpert`.

    읽는 정보: 손잡이 위치 · 너트 위치·방향 · 봉 위치·꼭대기 높이 · 그리퍼 개구 · eef 자세.
    경유점·판정은 위 `SquareExpert` 의 기하를 그대로 쓴다(_grasp_frame · _choose_yaw · holding ·
    PRE · GRAB_OUT · GRASP_DZ · NUT_CLEAR · ENGAGE · RETREAT · XY_TOL · YAW_TOL).

    단계: pick(손잡이 위 -> 파지점) -> grasp(닫기) -> transport(들기 + 봉 위로, 순항) -> [align_peg]
          -> insert -> release -> retreat. 못 잡으면 reopen -> pick.
    [측정 2026-10-04] 배포 때 저장한 개입 시작 상태 87개: 87/87 · 길이 111 (SquareExpert 146) · 포화 39% (26%).
    고정 50 위치 처음부터: 50/50 · 길이 134 · 포화 30% (공개 시연 149 · 11%).
    """

    def __init__(self, env, jitter=False, **_):
        super().__init__(env, v_max=C_V_MAX, w_max=C_W_MAX, t_min=C_T_MIN,
                         grip_moved=C_GRIP_MOVED, grip_still=C_GRIP_STILL, grip_eps=GRIP_EPS)
        self.geo = SquareExpert(env, jitter=False, continuous=True)     # 기하·판정 함수만 쓴다

    def start(self, n):
        """개입은 정책이 망쳐 놓은 상태에서 시작한다 — 이미 물었으면 놓으러, 닫혔는데 못 물었으면 열고 집으러."""
        q = self.gripper_q()
        if self.geo.holding():
            self._place(n)
        elif q[0] - q[1] < 0.05:
            self.hold(n, "reopen", -1, steps=10)
        else:
            self._pick(n)

    def update(self, n):
        g, ph = self.geo, self.phase
        if ph == "pick" and not self.motion_done(n) and np.linalg.norm(g._handle() - self.h_plan) > HANDLE_MOVED:
            self._pick(n, cont=n > self.t0)
        elif ph == "pick" and self.motion_done(n):
            self.hold(n, "grasp", 1)
        elif ph in ("grasp", "release") and self.motion_done(n):
            if self.grip_done() or n - self.t0 >= len(self.traj):
                if ph == "release":
                    p, Rc = self.eef()
                    self.set(self.segments(p, Rc, [(p + np.array([0.0, 0.0, RETREAT]), Rc, None)], -1), n, "retreat")
                elif g.holding():
                    self._place(n)
                else:                                              # 못 잡았다 — 열고 다시 집는다
                    self.hold(n, "reopen", -1, steps=10)
        elif ph == "reopen" and self.motion_done(n):
            self._pick(n)
        elif ph in ("transport", "align_peg") and self.motion_done(n):
            if self._aligned() or self.tries >= ALIGN_TRIES:
                self._insert(n)
            else:
                self._align(n)
        elif ph == "insert" and (g._nut()[0][2] < g._peg_top() - ENGAGE or self.motion_done(n)):
            self.hold(n, "release", -1)
        elif ph == "retreat" and self.motion_done(n):
            self.done = True

    def _pick(self, n, cont=False):
        """cont=True: 지금 따라가던 궤적의 위치·속도를 이어받아 다시 만든다(손잡이가 움직였을 때)."""
        g = self.geo
        R = g._grasp_frame()
        h = g._handle()
        self.h_plan = h.copy()
        u = g._nut()[0] - h
        u[2] = 0.0
        u /= np.linalg.norm(u)
        grab = h + u * GRAB_OUT
        if cont:
            (p0, R0, _), (p1, _, _) = self.target(n, -1), self.target(n, 0)
            v0 = p1 - p0
        else:
            (p0, R0), v0 = self.eef(), None
        up = np.array([0.0, 0.0, 1.0])
        self.set(self.segments(p0, R0, [(grab + up * PRE, R, None), (grab - up * GRASP_DZ, R, None)], -1, v0),
                 n, "pick")

    def _place(self, n):
        g = self.geo
        d = g._choose_yaw()
        self.target_yaw = _yaw(g._nut()[1]) + d
        p, Rc = self.eef()
        nut, _ = g._nut()
        self.off = p - nut
        z = g._peg_top() + NUT_CLEAR + self.off[2]             # 너트가 NUT_CLEAR 만큼 뜨는 eef 높이
        # 손목을 eef 둘레로 d 만큼 돌리면 너트의 수평 오프셋도 같이 돈다 -> 봉 위에서 eef 가 있을 자리
        o = rot_z(d)[:2, :2] @ (nut - p)[:2]
        above = np.r_[g._peg()[:2] - o, z]
        self.set(self.cruise(p, Rc, np.r_[p[:2], z], above, rot_z(d) @ Rc, 1), n, "transport")
        self.tries = 0

    def _yaw_err(self):
        return (_yaw(self.geo._nut()[1]) - self.target_yaw + np.pi) % (2 * np.pi) - np.pi

    def _aligned(self):
        return (np.linalg.norm(self.geo._nut()[0][:2] - self.geo._peg()[:2]) < XY_TOL
                and abs(self._yaw_err()) < YAW_TOL)

    def _align(self, n):
        g = self.geo
        p, Rc = self.eef()
        tgt = np.r_[p[:2] + (g._peg()[:2] - g._nut()[0][:2]), p[2]]
        self.set(self.segments(p, Rc, [(tgt, rot_z(-self._yaw_err()) @ Rc, C_V_INS)], 1), n, "align_peg")
        self.tries += 1

    def _insert(self, n):
        g = self.geo
        p, Rc = self.eef()
        tgt = np.r_[p[:2] + (g._peg()[:2] - g._nut()[0][:2]), g._peg_top() - ENGAGE - 0.02 + self.off[2]]
        self.set(self.segments(p, Rc, [(tgt, Rc, C_V_INS)], 1), n, "insert")

