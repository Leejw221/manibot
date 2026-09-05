"""MicrowaveTask 를 푸는 스크립트 정책 — 학습 데이터를 만드는 시연자.

**왜 스크립트인가**: 사람이 텔레오퍼레이션으로 모으기 전에, 이 하드웨어·이 배치에서 task 가
성립하는지부터 확인해야 한다. 성공률이 재현되면 그대로 대량 수집해 정책 학습의 바탕으로 쓴다.

**기하는 전부 실측으로 정했다** [2026-09-05, Panda + Microwave052]:
  - 그리퍼 좌표계: 접근축 = +eef_z, 개폐축 = eef_x (손가락 body 오프셋에서 확인)
  - 손잡이는 세로 막대(길이 175 mm, 개폐 방향 두께 17.8 mm)라 개폐축이 수평이어야 한다.
    개폐축을 **-y** 로 두면 수평 접근(t=0°)이 닿고, +y 로 두면 손목 한계에 걸려 못 닿는다.
  - 문을 여는 동안 그리퍼는 문과 함께 돈다 — 목표 자세를 Rz(theta) @ R0 로 주면 된다.
"""

import numpy as np
from robosuite.utils.transform_utils import (axisangle2quat, mat2quat, quat2axisangle,
                                             quat2mat)


from manibot.envs.ik import rot_z
from manibot.envs.microwave import HANDLE_FRAME, PICK_FRAME


def _slerp(R0, R1, a):
    """R0 -> R1 을 a(0~1) 만큼 간 회전. 축각으로 선형 보간한다."""
    if a >= 1.0:
        return R1
    return quat2mat(axisangle2quat(a * quat2axisangle(mat2quat(R1 @ R0.T)))) @ R0


R_HANDLE, R_PICK = HANDLE_FRAME, PICK_FRAME

PRE = 0.12              # 손잡이·버튼 앞 대기 거리 [m]
# 활짝 열린 문의 손잡이는 팔에서 가장 먼 곳이다. 대기 거리를 그대로 두면 대기점이 y=0.52 로
# 밀려 도달 범위를 벗어난다 — 열린 쪽만 더 짧게 잡는다 [실측 2026-09-05].
PRE_OPEN = 0.08
# 활짝 열린 문은 y=0.31~0.42 · x=0.09~-0.32 를 가로막는다. 손잡이를 놓고 큐브를 집으러
# 갈 때 그 사이를 직선으로 지나면 문을 밀어 닫아 버린다 (실측: -81° -> -27° [2026-09-05]).
# 문 윗면이 z~1.11 이라 그 위로 넘어간다.
OVER_Z = 1.28
GRIP_BUDGET = 70        # 그리퍼 개폐를 기다리는 최대 스텝 수
GRIP_EPS = 2e-4         # 손가락이 이만큼도 안 움직이면 다 닫힌(또는 물린) 것으로 본다
GRIP_MIN = 32           # 최소 대기 (명령이 10 스텝에 걸쳐 차오르고 damping 이 100 이다)


class MicrowaveExpert:
    """열기 -> 집기 -> 넣기 -> 닫기 -> 버튼. 각 구간은 waypoint 수렴 또는 제한시간으로 넘어간다."""

    def __init__(self, env, kp=6.0, kr=2.0, tol=0.012, ang_tol=0.15):
        self.env = env
        self.kp, self.kr, self.tol, self.ang_tol = kp, kr, tol, ang_tol
        self.reset()

    def reset(self):
        self._gen = self._plan()
        self.stage = "init"
        self.done = False

    # ── 저수준 ────────────────────────────────────────────────────────────
    def _eef(self):
        r = self.env.robots[0]
        s = r.eef_site_id[r.arms[0]]
        return (np.array(self.env.sim.data.site_xpos[s]),
                np.array(self.env.sim.data.site_xmat[s]).reshape(3, 3))

    def _action(self, pos, R, grip, max_step=1.0):
        p, Rc = self._eef()
        a = np.zeros(self.env.action_dim)
        a[:3] = np.clip(self.kp * (pos - p) / 0.05, -max_step, max_step)
        a[3:6] = np.clip(self.kr * quat2axisangle(mat2quat(R @ Rc.T)) / 0.5, -1, 1)
        a[-1] = grip
        return a

    def _move(self, target, rot, grip, name, budget=90, settle=0, ramp=30,
              max_step=1.0, tol=None, glide=0):
        """target·rot 은 값이거나 매 스텝 다시 계산되는 함수(움직이는 손잡이 추종용).

        ⚠️ 목표 자세를 한 번에 주면 자세 오차가 클 때 OSC 가 팔을 위로 튕겨 올린다
        (집은 뒤 위보기->앞보기로 90° 돌릴 때 실제로 z=1.45 까지 올라갔다 [2026-09-05]).
        그래서 시작 자세에서 목표 자세로 `ramp` 스텝에 걸쳐 나눠 준다. 큰 자세 변화는
        아예 `_reorient` 로 제자리에서 먼저 끝낸다.
        """
        self.stage = name
        p_start, R_start = self._eef()
        for i in range(budget):
            pos = target() if callable(target) else target
            R = rot() if callable(rot) else rot
            # ⚠️ 먼 목표를 한 번에 주면 OSC 가 팔꿈치를 펴며 위로 달아난다 (z=1.38~1.45 로
            # 튀는 실패를 여러 번 봤다 [2026-09-05]). 긴 이동은 목표점 자체를 직선으로
            # 흘려 보내 매 스텝 오차를 작게 유지한다.
            pos_cmd = (p_start + (np.asarray(pos, float) - p_start) * min(1.0, (i + 1) / glide)
                       if glide else pos)
            R_cmd = _slerp(R_start, R, min(1.0, (i + 1) / ramp)) if ramp else R
            yield self._action(pos_cmd, R_cmd, grip, max_step)
            p, Rc = self._eef()
            close = (np.linalg.norm(p - pos) < (tol or self.tol)
                     and np.linalg.norm(quat2axisangle(mat2quat(R @ Rc.T))) < self.ang_tol)
            if close and i >= settle:
                break
        for _ in range(settle):
            pos = target() if callable(target) else target
            yield self._action(pos, rot() if callable(rot) else rot, grip, max_step)

    def _reorient(self, R, grip, name, steps=45):
        """제자리에서 자세만 바꾼다 — 이동과 큰 회전을 겹치면 OSC 가 헤맨다."""
        self.stage = name
        p, R_start = self._eef()
        for i in range(steps):
            yield self._action(p, _slerp(R_start, R, min(1.0, (i + 1) / (steps * 0.7))), grip)

    def _swing(self, goal, grip, name, lead=0.09, budget=300):
        """손잡이를 잡은 채 문을 goal 까지 끈다. 그리퍼도 문과 함께 돈다.

        ⚠️ 목표 각도를 **실제 문 각도보다 lead 만큼만 앞**에 둔다. 정해진 속도로 각도를
        올리면(open-loop) 목표가 문보다 앞서 달아나 손잡이에 옆으로 전단력이 걸리고,
        손가락 사이에서 손잡이가 빠진다 (개구 19mm -> 43mm 로 벌어짐 [실측 2026-09-05]).
        """
        self.stage = name
        env = self.env
        d = np.sign(goal - env.door_angle()) or 1.0
        for _ in range(budget):
            th = env.door_angle()
            if (goal - th) * d <= 0.02:
                break
            cmd = th + d * lead
            if (cmd - goal) * d > 0:
                cmd = goal
            yield self._action(env.handle_at(cmd), rot_z(cmd) @ R_HANDLE, grip)
        for _ in range(15):
            yield self._action(env.handle_at(goal), rot_z(goal) @ R_HANDLE, grip)

    def _hold(self, pos, R, grip, n, name):
        self.stage = name
        for _ in range(n):
            yield self._action(pos, R, grip)

    def _grip(self, pos, R, grip, name, budget=GRIP_BUDGET):
        """손가락이 멈출 때까지 기다린다.

        ⚠️ 고정 스텝수로 기다리면 안 된다. Panda 그리퍼는 명령이 10 스텝에 걸쳐 차오르고
        관절 damping 이 100 이라 실제로 물리기까지 30 스텝 넘게 걸린다 — 14 스텝만 기다렸더니
        아직 30 mm 벌어진 채로 문을 끌기 시작해 손잡이가 손가락 밖으로 빠졌다 [2026-09-05].
        """
        self.stage = name
        r = self.env.robots[0]
        idx = r._ref_gripper_joint_pos_indexes[r.arms[0]]
        prev, still = None, 0
        for i in range(budget):
            yield self._action(pos, R, grip)
            q = np.array(self.env.sim.data.qpos[idx])
            still = still + 1 if prev is not None and np.abs(q - prev).max() < GRIP_EPS else 0
            prev = q
            # 초반에는 명령이 아직 차오르는 중이라 "안 움직임"이 곧 "다 물림"이 아니다.
            if still >= 5 and i >= GRIP_MIN:
                break

    # ── 계획 ──────────────────────────────────────────────────────────────
    def _plan(self):
        env = self.env
        OPEN_ANGLE = env.open_target
        R0, Ro = R_HANDLE, rot_z(OPEN_ANGLE) @ R_HANDLE
        # 접근 방향의 반대로 물러나는 벡터 (문이 돌면 접근축도 같이 돈다)
        back0 = -R0[:, 2] * PRE
        backo = -Ro[:, 2] * PRE_OPEN
        h0 = env.handle_at(0.0)
        ho = env.handle_at(OPEN_ANGLE)

        # ── 문 열기 ──
        yield from self._move(h0 + back0, R0, -1, "approach_handle", budget=120)
        # 파지점은 정밀해야 한다 — 손잡이 두께가 17.8 mm 라 몇 mm 만 어긋나도 미끄러진다
        yield from self._move(lambda: env.handle_at(env.door_angle()), R0, -1,
                              "reach_handle", budget=90, ramp=10, tol=0.004, settle=5)
        p, _ = self._eef()
        yield from self._grip(p, R0, 1, "grasp_handle")
        yield from self._swing(OPEN_ANGLE, 1, "open_door")
        yield from self._grip(ho, Ro, -1, "release_handle")
        yield from self._move(ho + backo, Ro, -1, "back_off", budget=70, ramp=0, glide=25)

        # ── 열린 문 위로 넘어 큐브 쪽으로 ──
        over = ho + backo
        yield from self._move([over[0], over[1], OVER_Z], Ro, -1, "up_over_door",
                              budget=110, ramp=0, max_step=0.6, glide=40)
        yield from self._reorient(R_PICK, -1, "turn_down")
        cube = env.cube_pos()
        yield from self._move([cube[0], cube[1], OVER_Z], R_PICK, -1, "cross_to_cube",
                              budget=160, ramp=0, max_step=0.6, glide=70)

        # ── 큐브 집기 ──
        yield from self._move(lambda: env.cube_pos() + [0, 0, 0.13], R_PICK, -1,
                              "above_cube", budget=90, ramp=0)
        yield from self._move(lambda: env.cube_pos() + [0, 0, 0.002], R_PICK, -1,
                              "descend", budget=70, ramp=0, settle=6, max_step=0.4, tol=0.005)
        p, _ = self._eef()
        yield from self._grip(p, R_PICK, 1, "grasp_cube")
        yield from self._move(cube + [0, 0, 0.18], R_PICK, 1, "lift", budget=60, ramp=0)

        # ── 넣기 ── 개폐축(-y)이 그대로라 손목만 돌리면 수평 삽입 자세가 된다.
        # ⚠️ 이동은 전부 **위보기 자세**로 한다. 수평 자세로 크게 움직이면 OSC 가 팔꿈치를
        # 펴며 z=1.4 까지 달아난다 — 자세 전환은 목적지에서 제자리로 끝낸다 [2026-09-05].
        c, half = env.interior_center(), env.interior_half()
        z_in = c[2] - half[2] + 0.07
        stage_pt = np.array([c[0] - half[0] - 0.13, c[1], z_in])
        drop_pt = np.array([c[0] - half[0] + 0.07, c[1], z_in])
        yield from self._move([stage_pt[0], stage_pt[1], z_in + 0.15], R_PICK, 1,
                              "carry", budget=180, ramp=0, max_step=0.6, glide=80)
        yield from self._reorient(R0, 1, "turn_forward")
        yield from self._move(stage_pt, R0, 1, "lower", budget=90, ramp=0,
                              max_step=0.4, glide=25)
        yield from self._move(drop_pt, R0, 1, "insert", budget=80, ramp=0, settle=4,
                              max_step=0.4)
        yield from self._grip(drop_pt, R0, -1, "release_cube")
        yield from self._move(stage_pt, R0, -1, "retreat", budget=90, ramp=0,
                              max_step=0.4, glide=30)

        # ── 다시 문 위로 넘어 손잡이 쪽으로 ──
        # 문은 -83° 를 지나 관절 한계(-90°)까지 밀려 있을 수 있다. 계획을 세울 때의 값이
        # 아니라 **지금 문 각도**로 손잡이 위치·자세를 다시 잡는다.
        th = env.door_angle()
        Ro, ho = rot_z(th) @ R_HANDLE, env.handle_at(th)
        over = ho - Ro[:, 2] * PRE_OPEN
        yield from self._reorient(R_PICK, -1, "turn_up")
        yield from self._move([stage_pt[0], stage_pt[1], OVER_Z], R_PICK, -1,
                              "up_over_door_2", budget=110, ramp=0, max_step=0.6, glide=40)
        yield from self._move([over[0], over[1], OVER_Z], R_PICK, -1, "cross_to_door",
                              budget=160, ramp=0, max_step=0.6, glide=70)
        yield from self._reorient(Ro, -1, "turn_to_door")

        # ── 문 닫기 ──
        yield from self._move(over, Ro, -1, "approach_open_handle", budget=110, ramp=0,
                              glide=30)
        yield from self._move(lambda: env.handle_at(env.door_angle()), Ro, -1,
                              "reach_open_handle", budget=90, ramp=10, tol=0.004, settle=5)
        p, _ = self._eef()
        yield from self._grip(p, Ro, 1, "grasp_handle_2")
        yield from self._swing(0.0, 1, "close_door")
        yield from self._grip(h0, R0, -1, "release_handle_2")
        yield from self._move(h0 + back0, R0, -1, "back_off_2", budget=60, ramp=0)

        # ── 버튼 ── 닫은 그리퍼 끝으로 민다. 목표를 패널 안쪽으로 조금 넣어 접촉을 만든다.
        b = env.button_pos()
        yield from self._move(b + [-PRE, 0, 0], R0, 1, "approach_button", budget=120, ramp=0)
        yield from self._move(b + [0.02, 0, 0], R0, 1, "press", budget=60, ramp=0,
                              settle=10, max_step=0.3)

    # ── 외부 인터페이스 ───────────────────────────────────────────────────
    def act(self):
        try:
            return next(self._gen)
        except StopIteration:
            self.done = True
            self.stage = "done"
            p, R = self._eef()
            return self._action(p, R, 1)
