"""스크립트 전문가의 **움직이는 방식** — task 와 무관한 공통 부분.

[사용자 2026-10-04] "움직이는 방식은 scripted_expert.py 에 두고, task 특성에 맞춰 (물체 위치·각도 같은)
정보를 기반으로 궤적을 만드는 방식만 task 마다 다르게 한다."

움직이는 방식:
  1. 궤적을 **만들어 두고** 스텝마다 시간대로 따라간다. 실행 중에 다시 그리지 않는다 — 실제 위치·속도에서
     자꾸 다시 그리면 청크로 끊어 실행할 때 "실제 속도 -> 다음 계획" 되먹임으로 흔들렸다
     [측정 2026-10-03: square 청크 실행 42/50 · 포화 52% -> 이 방식 50/50].
  2. 물체와 닿는 상태가 바뀌는 **사건**(잡힘·놓임·정렬 등)에서만 남은 궤적을 새로 만든다 — 하위 클래스가
     `start(n)` · `update(n)` 에서 정한다.
  3. 행동 = "궤적의 다음 점 - 실제 eef" 에 이득을 곱한 delta (robosuite OSC_POSE delta 기준).

궤적 재료: `segments`(경유점을 멈추지 않고 지나는 5차 다항식 구간) · `cruise`(가속 -> 최고 속도 순항 ->
감속) · `hold`(지금 자세 유지 — 그리퍼를 여닫을 때).
수집기 인터페이스: `Expert(env, jitter=False)` · `act()` · `stage` · `done` · `aborted`.
"""

import numpy as np
from scipy.spatial.transform import Rotation


# ── 5차 다항식 · 시간 맞춤 (square_expert 연속 모드도 이걸 쓴다) ────────────────────
def quintic(x0, v0, xf, vf, T):
    """5차 다항식 계수 — 위치·속도는 양끝 조건대로, 가속도는 양끝 0. 반환 (x0, v0, c3, c4, c5).
    실물 Piper 보간과 같은 식(Cashier_policy-dp `robots/piper_robot.py:113` _quintic_coeffs)."""
    ds, dv = xf - x0 - v0 * T, vf - v0
    return (x0, v0, (20 * ds - 8 * dv * T) / (2 * T ** 3),
            (-30 * ds + 14 * dv * T) / (2 * T ** 4), (12 * ds - 6 * dv * T) / (2 * T ** 5))


def qeval(c, t):
    x0, v0, c3, c4, c5 = c
    return x0 + v0 * t + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5


def fit_T(x0, v0, xf, vf, v_max, t_min=3):
    """속도가 v_max 를 넘지 않는 가장 짧은 시간 T [step].

    정지->정지면 T = 1.875 x 거리 / v_max 지만, 이미 움직이는 중에 다시 계획하면 그만큼 길게
    잡을 이유가 없다 — 길게 잡으면 다시 계획할 때마다 감속해 이송이 느려졌다 [측정 2026-10-01].
    출발 속도가 이미 v_max 보다 빠르면 그 속도를 상한으로 본다.
    """
    lim = max(v_max, float(np.linalg.norm(v0))) * 1.02
    T = max(t_min, float(np.linalg.norm(xf - x0)) / v_max)
    for _ in range(40):
        _, _, c3, c4, c5 = quintic(x0, v0, xf, vf, T)
        t = np.linspace(0.0, T, 17)[:, None]
        if np.linalg.norm(v0 + 3 * c3 * t ** 2 + 4 * c4 * t ** 3 + 5 * c5 * t ** 4, axis=1).max() <= lim:
            return T
        T *= 1.1
    return T


# 회전은 float64(scipy)로 — robosuite mat2quat 은 float32 라 0.03° 아래 회전이 0 이 된다 [실측 2026-10-03]
def rvec(R):
    return Rotation.from_matrix(R).as_rotvec()


def rmat(v):
    return Rotation.from_rotvec(np.asarray(v, float)).as_matrix()


Z3 = np.zeros(3)


class ScriptedExpert:
    """궤적을 만들어 두고 따라가는 전문가의 공통 부분. 하위 클래스는 `start(n)` · `update(n)` 만 정한다.

    act() 는 env.step **직전에** 한 번 부른다: 처음이면 start, 아니면 update 로 사건을 보고 필요하면
    `self.set(...)` 으로 남은 궤적을 바꾼 뒤, 이번 스텝이 겨냥할 점으로 행동을 낸다.
    """

    def __init__(self, env, v_max=0.013, w_max=np.radians(5.0), t_min=3, kp=6.0, kr=2.0,
                 pos_scale=0.05, rot_scale=0.5, grip_moved=0.003, grip_still=3, grip_eps=2e-4,
                 grip_budget=40):
        self.env = env
        self.v_max, self.w_max, self.t_min = v_max, w_max, t_min
        # 행동 변환: robosuite OSC_POSE delta 는 행동 1 = 위치 0.05 m · 회전 0.5 rad (output_max)
        self.kp, self.kr, self.pos_scale, self.rot_scale = kp, kr, pos_scale, rot_scale
        self.grip_moved, self.grip_still, self.grip_eps = grip_moved, grip_still, grip_eps
        self.grip_budget = grip_budget
        self.n = 0
        self.log = []                  # (step, 단계) — 언제 궤적을 새로 만들었나
        self.traj = None
        self.phase = "init"
        self.done = False
        self.aborted = False

    # ── 하위 클래스가 정한다 ──
    def start(self, n):
        raise NotImplementedError

    def update(self, n):
        raise NotImplementedError

    # ── 따라가기 ──
    @property
    def stage(self):
        return self.phase

    @property
    def sim(self):
        return self.env.sim            # hard_reset 이면 리셋마다 sim 객체가 바뀐다 — 캐시하지 않는다

    def act(self):
        if self.traj is None:
            self.start(0)
        else:
            self.update(self.n)
        pos, R, grip = self.target(self.n)
        self.n += 1
        return self.action(pos, R, grip)

    def target(self, n, k=0):
        """step n+k 의 행동이 겨냥할 자세 = 궤적에서 시각 n+k+1 의 점. 끝나면 마지막 점을 유지."""
        return self.traj[min(max(n - self.t0 + k, 0), len(self.traj) - 1)]

    def set(self, traj, n, phase, wait=None):
        """남은 궤적을 바꾼다. wait = 궤적 끝의 대기 칸 수(그 앞까지가 움직임)."""
        self.traj, self.t0, self.phase, self.wait = traj, n, phase, wait
        self.log.append((n, phase))

    def motion_done(self, n):
        return n - self.t0 >= len(self.traj) - (self.wait or 0)

    def action(self, pos, R, grip):
        p, Rc = self.eef()
        a = np.zeros(self.env.action_dim)
        a[:3] = np.clip(self.kp * (pos - p) / self.pos_scale, -1.0, 1.0)
        a[3:6] = np.clip(self.kr * rvec(R @ Rc.T) / self.rot_scale, -1.0, 1.0)
        a[-1] = grip
        return a

    # ── 로봇 상태 (robosuite 한 팔) ──
    def eef(self):
        r = self.env.robots[0]
        s = r.eef_site_id[r.arms[0]]
        return (np.array(self.sim.data.site_xpos[s]), np.array(self.sim.data.site_xmat[s]).reshape(3, 3))

    def gripper_q(self):
        r = self.env.robots[0]
        return np.array(self.sim.data.qpos[r._ref_gripper_joint_pos_indexes[r.arms[0]]])

    # ── 궤적 재료 ──
    def segments(self, p0, R0, wps, grip, v0=None):
        """(p0, R0) 에서 속도 v0(없으면 정지)로 출발해 경유점들을 지나 마지막에서 멈추는 궤적.

        wps: [(pos, R, v_max | None)]. 반환: 스텝마다 (pos, R, grip) — 첫 원소가 출발 다음 스텝의 목표다.
        경유점 끝속도는 다음 경유점 쪽으로 min(0.5 v_max, 거리/t_min), 시간은 속도 상한을 넘지 않는
        가장 짧은 값(fit_T)을 스텝 단위로 올림."""
        out = []
        p, R, v = np.asarray(p0, float), R0, (Z3.copy() if v0 is None else np.asarray(v0, float))
        for i, (pt, Rt, vl) in enumerate(wps):
            pt, vl = np.asarray(pt, float), vl or self.v_max
            vf = Z3.copy()
            if i < len(wps) - 1:
                d = np.asarray(wps[i + 1][0], float) - pt
                nd = np.linalg.norm(d)
                if nd > 1e-6:
                    vf = d / nd * min(0.5 * vl, nd / self.t_min)
            r = rvec(Rt @ R.T)
            T = int(np.ceil(max(fit_T(p, v, pt, vf, vl, self.t_min), fit_T(Z3, Z3, r, Z3, self.w_max, self.t_min))))
            cp, cr = quintic(p, v, pt, vf, T), quintic(Z3, Z3, r, Z3, T)
            out += [(qeval(cp, k), rmat(qeval(cr, k)) @ R, float(grip)) for k in range(1, T + 1)]
            p, R, v = pt, Rt, vf
        return out

    def cruise(self, p0, R0, corner, end, Rt, grip, v_max=None, ramp=10, r_max=0.08):
        """p0 -> corner -> end 꺾인 경로를 **최고 속도로 달리는** 궤적 [사용자 2026-10-03].

        정지->정지 5차 구간은 평균 속도가 최고의 53% 라 먼 이동이 느리다(square 운반 39 cm 에 67 스텝 -> 49).
        ramp 스텝 동안 smoothstep 으로 가속 -> v_max 순항 -> ramp 스텝 감속(가속도도 이어진다).
        모서리는 2차 베지어로 깎는다 — 반경은 앞 구간의 절반 · 뒤 구간의 절반 · r_max 중 작은 값.
        회전은 전체 시간에 5차로 나눈다(최고 각속도 1.875 x 평균 <= w_max)."""
        v_max = v_max or self.v_max
        p0, c, e = (np.asarray(x, float) for x in (p0, corner, end))
        d1, d2 = c - p0, e - c
        l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
        r = min(0.5 * l1, 0.5 * l2, r_max)
        a, b = c - d1 / max(l1, 1e-9) * r, c + d2 / max(l2, 1e-9) * r
        u = np.linspace(0, 1, 200)[:, None]
        pts = np.vstack([p0 + (a - p0) * u, (1 - u) ** 2 * a + 2 * u * (1 - u) * c + u ** 2 * b, b + (e - b) * u])
        sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        L = sl[-1]
        rv = rvec(Rt @ R0.T)
        T = max(int(np.ceil(ramp + L / v_max)), 2 * ramp, int(np.ceil(1.875 * np.linalg.norm(rv) / self.w_max)))
        v = L / (T - ramp)

        def s_of(t):
            if t <= ramp:
                x = t / ramp
                return v * ramp * (x ** 3 - x ** 4 / 2)
            if t >= T - ramp:
                y = (T - t) / ramp
                return L - v * ramp * (y ** 3 - y ** 4 / 2)
            return v * ramp / 2 + v * (t - ramp)

        cr = quintic(Z3, Z3, rv, Z3, T)
        return [(np.array([np.interp(s_of(k), sl, pts[:, i]) for i in range(3)]),
                 rmat(qeval(cr, k)) @ R0, float(grip)) for k in range(1, T + 1)]

    def hold(self, n, phase, grip, steps=None):
        """**지금 실제 자세**를 유지하며 그리퍼만 여닫는다. 계획한 점(몇 mm 아래)을 계속 겨냥하면 손끝이
        눌린 채 걸려 손가락이 안 닫힌다 [측정 2026-10-03 square]. 끝나는 시점은 grip_done 으로 본다."""
        p, Rc = self.eef()
        steps = steps or self.grip_budget
        self.set([(p, Rc, float(grip))] * steps, n, phase, wait=steps)
        self.q0, self.still, self.prev = None, 0, None

    def grip_done(self):
        """손가락이 움직였다가 멈췄나. 명령 직후엔 아직 안 움직여서 멈춤만 보면 너무 일찍 끝난다."""
        q = self.gripper_q()
        if self.q0 is None:
            self.q0 = q
        self.still = self.still + 1 if self.prev is not None and np.abs(q - self.prev).max() < self.grip_eps else 0
        self.prev = q
        return np.abs(q - self.q0).max() > self.grip_moved and self.still >= self.grip_still
