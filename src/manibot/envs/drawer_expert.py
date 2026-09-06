"""DrawerTask 를 푸는 스크립트 정책 — 학습 데이터를 만드는 시연자.

  ① 왼팔 위층 열기 → ② 오른팔 블록 꺼내기 → ③ 왼팔 위층 닫기
  → ④ 왼팔 아래층 열기 → ⑤ 오른팔 블록 넣기 → ⑥ 왼팔 아래층 닫기 → ⑦ 복귀

**제어는 관절 절대위치**(`NS.controller_config(joint_space=True)`)다. 목표 손 위치·자세를
IK 로 풀어 관절값으로 지령한다. OSC 는 영공간이 리셋 자세로 당겨서, 기구학적으로 닿는
지점인데도 제어가 8~34 cm 못 갔다 [실측 2026-09-06].

**모든 이동은 데카르트 경로로 만든다.** 관절공간 직선으로 가면 손이 실제로 어느 공간을
지나는지 통제가 안 돼, IK 해가 맞아도 가는 길에 서랍장에 막히고 그 힘에 열어둔 서랍이
도로 닫혔다 [실측 2026-09-06, 반복].

**궤적은 한 번의 매끄러운 프로파일로 간다** — Catmull-Rom 스플라인 + smoothstep 시간
프로파일 + 자세는 SLERP. 예전에는 2 cm 구간마다 IK 를 풀고 매번 가속→감속→정지를
반복해 손이 눈에 띄게 덜덜거렸다 [사용자 지적 2026-09-06].

그 밖에 실측으로 확정한 것은 각 상수·함수의 주석에 있다.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import manibot.robots.nero.sim as NS
from manibot.envs.ik import ArmIK, rot_z

# 위층 손잡이는 앞에서 **수평으로** 문다 (사람이 서랍 여는 방식) [사용자 2026-09-06].
R_BAR_H = NS.frame((1, 0, 0), (0, 0, 1))
# 아래층은 **대각선**으로 앞아래에서 문다 [사용자 2026-09-06].
#  · 개폐축에 당기는 방향(x) 성분이 있어 뒤쪽 패드가 막대를 미는 형상 잠금이 된다
#  · 놓고 같은 축으로 물러나면 손가락이 막대에서 그냥 빠진다 (수평 파지는 갈고리처럼
#    당겨 서랍이 도로 열렸다)
#  · 그리퍼 몸통이 파지점 뒤위로 물러나 **위층 막대를 14 mm 여유로 피한다**
# 기울기 45° 는 접근점의 손목이 어깨에서 0.246 m 라 최소 도달 반경(0.281 m) 안으로
# 들어가 접근 자체가 실패했다. **30°** 여야 한다 [실측 2026-09-06].
_CA, _SA = np.cos(np.radians(30)), np.sin(np.radians(30))
DIAG = np.array([_CA, 0.0, -_SA])
R_BAR_D = NS.frame(tuple(DIAG), (_SA, 0.0, _CA))
R_TOP = NS.frame((0, 0, -1), (0, 1, 0))      # 블록: 위에서, 월드 y 로 문다

# 왼팔이 막대를 잡는 **월드 y 좌표**. 막대 중심에서의 오프셋으로 두면 안 된다 — 막대
# 형상을 바꾸면 중심이 옮겨가는데 오프셋은 그대로라 막대 밖을 집는다 [실측 2026-09-06].
# 도달 지도상 왼팔은 y=+0.19~0.20 구간에서 가장 넓다.
GRIP_Y = 0.19
# 블록을 든 채 대기하는 자리 (서랍장 밖·위). ⚠️ 블록 **하단**이 위층 서랍 앞판
# 윗면(0.896)보다 위여야 한다. z 0.92 로 두면 하단이 0.878 이라 18 mm 겹쳐, 닫히는
# 앞판이 블록을 쳐서 서랍이 1.98 → 12.5 cm 로 도로 열렸다 [실측 2026-09-06].
CARRY = np.array([-0.09, -0.12, 0.95])

SLOT = {"right": slice(0, 7), "left": slice(7, 14)}
GRIP_IDX = {"right": 14, "left": 15}


def _smooth(u):
    """smoothstep — 시작·끝에서 속도가 0 이라 가감속이 부드럽다."""
    return u * u * (3.0 - 2.0 * u)


class DrawerExpert:
    """`act()` 를 부를 때마다 다음 한 스텝의 지령(관절 절대위치 16 차원)을 준다."""

    def __init__(self, env, verbose=False):
        self.env = env
        self.r = env.robots[0]
        self.verbose = verbose
        self.jidx = {a: [env.sim.model.joint(f"robot0_{a}_joint{i}").qposadr[0]
                         for i in range(1, 8)] for a in self.r.arms}
        self.ik = {a: ArmIK(env, a) for a in self.r.arms}
        self.reset()

    # ── 외부 인터페이스 ───────────────────────────────────────────────────
    def reset(self):
        self.cmd = {a: np.array(self.env.sim.data.qpos[self.jidx[a]]) for a in self.r.arms}
        self.grip = {a: -1.0 for a in self.r.arms}
        self.home = {a: self.cmd[a].copy() for a in self.r.arms}
        self.stage = "start"
        self.done = False
        self._gen = self._plan()

    def act(self):
        """다음 지령. 계획이 끝나면 `done` 이 서고 마지막 지령을 그대로 유지한다."""
        try:
            next(self._gen)
        except StopIteration:
            self.done = True
            self.stage = "done"
        return self._action()

    def warmup(self, n=40):
        """**첫 동작은 반드시 실패한다 — 기록하지 않는 예행 동작이 필요하다.**

        프로세스 안에서 첫 파지-당기기만 0.08 cm 이고 2회차부터는 11.84 cm 로 **완전히
        같은 값**이 나온다(결정론적) [실측 2026-09-06]. reset 을 여러 번 하거나 제자리에서
        300 스텝 정착시키는 것으로는 안 되고, **팔이나 그리퍼가 실제로 한 번 움직여야**
        한다. 그리퍼를 여닫는 것으로 충분하다. "격리하면 되는데 전체 시퀀스에서는
        안 된다"가 반복된 원인이 전부 이것이었다.

        마지막에 `env.reset()` 하므로, 이 호출 다음이 곧 에피소드의 첫 프레임이다.
        """
        for g in (1.0, -1.0):
            self.grip["left"] = g
            for _ in range(n):
                self.env.step(self._action())
        self.env.reset()
        self.reset()

    # ── 저수준 ────────────────────────────────────────────────────────────
    def _action(self):
        a = np.zeros(self.env.action_dim)
        for x in self.r.arms:
            a[SLOT[x]] = self.cmd[x]
            a[GRIP_IDX[x]] = self.grip[x]
        return a

    def _hand(self, a):
        return np.array(self.env.sim.data.site_xpos[self.r.eef_site_id[a]])

    def _hand_R(self, a):
        return np.array(self.env.sim.data.site_xmat[self.r.eef_site_id[a]]).reshape(3, 3)

    def _opening(self, a):
        """그리퍼 개구 [mm]."""
        q = np.array(self.env.sim.data.qpos[self.r._ref_gripper_joint_pos_indexes[a]])
        return (q[0] - q[1]) * 1000

    def _say(self, msg):
        if self.verbose:
            print(f"    {msg}", flush=True)

    # ── 이동 원시 동작 ────────────────────────────────────────────────────
    def _hold(self, n, tag=None, wait_arm=None):
        """제자리 유지. `wait_arm` 을 주면 그 그리퍼가 멎는 즉시 빠져나온다."""
        if tag:
            self.stage = tag
        prev, still = None, 0
        for i in range(n):
            yield
            if wait_arm:
                gq = np.array(self.env.sim.data.qpos[
                    self.r._ref_gripper_joint_pos_indexes[wait_arm]])
                still = still + 1 if prev is not None and np.abs(gq - prev).max() < 2e-4 else 0
                prev = gq
                if still >= 5 and i >= n * 0.4:
                    break

    def _ramp(self, a, q1, n):
        """관절공간 보간 — 목표가 자세가 아니라 관절값일 때(복귀)만 쓴다."""
        q0 = self.cmd[a].copy()
        for i in range(1, n + 1):
            self.cmd[a] = q0 + (q1 - q0) * _smooth(i / n)
            yield
        self.cmd[a] = q1.copy()

    def _line(self, a, p1, R=None, tag=None, v=0.004, tol=0.02):
        """현재 손 위치 → p1 을 **데카르트 직선**으로. 자세는 R(없으면 현재)로 유지.

        `v` 는 곧 속도다 — 20 Hz 이므로 0.004 m/스텝 = 0.08 m/s.
        """
        if tag:
            self.stage = tag
        if R is None:
            R = self._hand_R(a)
        p0 = self._hand(a)
        p1 = np.asarray(p1, float)
        n = max(10, int(np.ceil(np.linalg.norm(p1 - p0) / v)))
        for i in range(1, n + 1):
            t = p0 + (p1 - p0) * _smooth(i / n)
            q, ep, _ = self.ik[a].solve(t, R, self.cmd[a])
            if ep > 0.012:
                self._say(f"⚠ {tag or ''} 직선 {i}/{n} IK 실패 {np.round(t, 3)}")
                return False
            self.cmd[a] = q
            yield
        err = np.linalg.norm(self._hand(a) - p1)
        if err > tol:
            self._say(f"⚠ {tag or ''} 직선 도착 오차 {err * 1000:.0f}mm")
        return err <= tol

    def _curve(self, a, pts, R_end=None, v=0.004, tag=None, tol=0.025):
        """웨이포인트들을 지나는 **하나의 부드러운 곡선**으로 이동한다.

        사람이 손을 뻗을 때처럼 위치와 자세가 **동시에** 바뀐다: 경로는 Catmull-Rom
        스플라인(중간점에서 꺾이지 않는다), 시간은 전체 호 길이에 smoothstep 한 번
        (중간에서 멈추지 않는다), 자세는 SLERP(제자리 회전이 따로 없다).
        """
        if tag:
            self.stage = tag
        P = np.vstack([self._hand(a)] + [np.asarray(p, float) for p in pts])
        C = np.vstack([P[0], P, P[-1]])              # 양 끝을 복제해 접선을 만든다
        dense = []
        for i in range(len(P) - 1):
            a0, a1, a2, a3 = C[i], C[i + 1], C[i + 2], C[i + 3]
            for s in np.linspace(0.0, 1.0, 24, endpoint=False):
                s2, s3 = s * s, s * s * s
                dense.append(0.5 * ((2 * a1) + (-a0 + a2) * s
                                    + (2 * a0 - 5 * a1 + 4 * a2 - a3) * s2
                                    + (-a0 + 3 * a1 - 3 * a2 + a3) * s3))
        dense.append(P[-1])
        D = np.array(dense)
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(D, axis=0), axis=1))])
        L = cum[-1]
        if L < 1e-6:
            return True
        R0 = self._hand_R(a)
        R1 = np.asarray(R_end) if R_end is not None else R0
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R0, R1])))
        n = max(14, int(np.ceil(L / v)))
        for i in range(1, n + 1):
            u = _smooth(i / n)
            pos = np.array([np.interp(u * L, cum, D[:, k]) for k in range(3)])
            q, ep, _ = self.ik[a].solve(pos, slerp([u])[0].as_matrix(), self.cmd[a])
            if ep > 0.015:
                self._say(f"⚠ {tag or ''} 곡선 {i}/{n} IK 실패 {np.round(pos, 3)}")
                return False
            self.cmd[a] = q
            yield
        err = np.linalg.norm(self._hand(a) - P[-1])
        if err > tol:
            self._say(f"⚠ {tag or ''} 곡선 도착 오차 {err * 1000:.0f}mm")
        return err <= tol

    def _travel(self, a, p1, R=None, tag=None, lift=0.10, v=0.004):
        """멀리 이동 — 출발점과 목표를 **하나의 호**로 잇는다 (자세도 같이 돈다).

        ⚠️ "안전 높이" 를 하나로 못 박으면 안 된다. z=1.00 은 NERO 도달 범위 밖이라
        (어깨 1.08 · 최대 도달 0.60 m) 자세 전환부터 IK 가 실패했다 [실측 2026-09-06].
        대신 출발·도착의 중점만 `lift` 만큼 띄운다.
        """
        mid = (self._hand(a) + np.asarray(p1, float)) / 2.0 + np.array([0, 0, lift])
        return (yield from self._curve(a, [mid, p1], R, v=v, tag=tag))

    def _reorient(self, a, R, tag=None, n=140):
        """제자리에서 손 자세만 R 로 바꾼다. 높은 곳에서만 부른다."""
        if tag:
            self.stage = tag
        p = self._hand(a)
        q0 = self.cmd[a]
        rs = np.random.RandomState(0)
        best = None
        for k in range(60):     # 해가 여럿이라 시드를 흔들어 **가장 가까운** 해를 고른다
            s0 = q0 if k == 0 else np.clip(q0 + rs.normal(0, 0.4, 7),
                                           self.ik[a].rng[:, 0] + 0.1,
                                           self.ik[a].rng[:, 1] - 0.1)
            q, ep, ea = self.ik[a].solve(p, R, s0)
            if ep < 0.004 and ea < 0.03:
                c = np.linalg.norm(q - q0)
                if best is None or c < best[0]:
                    best = (c, q)
        if best is None:
            self._say(f"⚠ {tag or ''} 자세 전환 IK 실패")
            return False
        yield from self._ramp(a, best[1], n)
        return np.linalg.norm(self._hand(a) - p) < 0.03

    def _set_pose(self, a, R, up, tag):
        """작업 자세를 만든다 — 위로 뜬 뒤 빈 공중에서 한 번만 돌린다."""
        yield from self._line(a, self._hand(a) + np.array([0, 0, up]), tag=tag)
        return (yield from self._reorient(a, R, tag))

    def _preclose(self, a, mm, nmax=140, tag=None):
        """그리퍼를 목표 개구까지만 미리 좁히고 **그 상태로 유지**한다.

        활짝 벌린 채(100 mm) 서랍 안으로 내려가면 손가락 **바깥면**(파지 중심에서 55 mm)이
        서랍 벽에 걸려 개구가 85 mm 에서 멎는다 [실측 2026-09-06]. 사람도 물체 폭에 맞춰
        손을 벌리고 넣는다. grip=0 은 부호가 0 이라 지령이 그대로 유지된다.
        ⚠️ 관성으로 목표보다 8~9 mm 더 닫힌다 — 부르는 쪽이 그만큼 크게 잡아야 한다.
        """
        if tag:
            self.stage = tag
        self.grip[a] = 1.0
        for _ in range(nmax):
            yield
            if self._opening(a) <= mm:
                break
        self.grip[a] = 0.0
        return self._opening(a)

    # ── 손잡이 다루기 ─────────────────────────────────────────────────────
    # **한 번 잡으면 그 서랍을 다 쓸 때까지 놓지 않는다** [사용자 2026-09-06].
    # 열고 놓았다가 닫으러 다시 접근하는 구조는 실패의 절반을 만들었다 — 물러나는 경로가
    # 열린 서랍 앞을 가로질러 방금 연 것을 도로 닫았고, 다시 잡을 때 좌표가 이미 틀어져
    # 있었다. 잡은 채로 두면 그 단계가 통째로 사라진다.

    def _grab_handle(self, which, tag, dx=0.0):
        """손잡이를 잡는다. 위층은 앞에서 수평으로, 아래층은 **대각선 앞아래로**."""
        H = self.env.handle_pos(which)
        H[1] = GRIP_Y
        if which == "upper":
            # 손잡이 앞 12 cm 로 **한 번에** 간다. 예전에는 "뒤 평면으로 → 높이 맞추고 →
            # 전진" 3단계였는데, 그 중간 평면(x -0.24)이 도달 경계라 IK 가 깨지고 그때
            # 팔이 흔들려 아래층 서랍을 쳤다 [사용자 지적 · 실측 2026-09-06].
            yield from self._line("left", H - np.array([0.12, 0, 0]), R_BAR_H,
                                  tag=f"{tag} front", tol=0.015)
            op = yield from self._preclose("left", 52, tag=f"{tag} narrow")
            yield from self._line("left", H + np.array([dx, 0, 0]), R_BAR_H,
                                  tag=f"{tag} enclose", v=0.002, tol=0.015)
        else:
            pre = H - DIAG * 0.12                    # 손잡이 뒤위 12 cm (대각선 축 위)
            ok = yield from self._curve("left", [pre], R_BAR_D, tag=f"{tag} approach",
                                        tol=0.015)
            if not ok:
                yield from self._line("left", pre, tag=f"{tag} approach2", v=0.003, tol=0.015)
            op = yield from self._preclose("left", 52, tag=f"{tag} narrow")
            yield from self._line("left", H + DIAG * dx, R_BAR_D, tag=f"{tag} enclose",
                                  v=0.002, tol=0.015)
        Rg = self._hand_R("left")
        self.grip["left"] = 1.0
        yield from self._hold(120, wait_arm="left", tag=f"{tag} grasp")
        op2 = self._opening("left")
        self._say(f"[{which}] 좁힘 {op:.0f} → 파지 {op2:.1f}mm (막대 22mm · 20~34 면 정상)")
        return Rg, 20.0 <= op2 <= 34.0

    def _slide_handle(self, which, delta, Rg, tag):
        """잡은 채로 x 로 delta 만큼 (음수=당겨 열기 / 양수=밀어 닫기).

        ⚠️ **천천히**. 0.04 m/s 면 접촉이 뚫려 막대가 패드를 빠져나갔다. 막대가 22 mm 로
        얇아 민감하다 [실측 2026-09-06].
        """
        p0 = self._hand("left").copy()
        yield from self._line("left", p0 + np.array([delta, 0, 0]), Rg, tag=tag,
                              tol=0.03, v=0.001)
        self._say(f"[{which}] 손 x {p0[0]:+.3f} → {self._hand('left')[0]:+.3f} "
                  f"(목표 {delta:+.3f}) · 서랍 {self.env.slide(which) * 100:.2f}cm")

    def _close_fully(self, which, Rg, tag, tries=3):
        """**끝까지** 닫는다. 한 번에 다 안 닫히면 남은 만큼 다시 민다 [사용자 지적]."""
        for k in range(tries):
            left = self.env.slide(which)
            if left < 0.008:
                return True
            yield from self._slide_handle(which, left + 0.010, Rg,
                                          f"{tag}{'' if k == 0 else f' {k + 1}'}")
        return self.env.slide(which) < 0.02

    def _release_handle(self, Rg, tag):
        """**그리퍼를 벌리고 손을 뺀다.** 벌린 뒤 물러나면 막대에서 그냥 빠진다.

        닫은 채로 빼면 손가락이 막대를 갈고리처럼 당겨 서랍이 도로 열린다
        [사용자 지적 2026-09-06].
        """
        self.grip["left"] = -1.0
        yield from self._hold(130, wait_arm="left", tag=f"{tag} open gripper")
        yield from self._line("left", self._hand("left") - np.array([0.12, 0, 0]), Rg,
                              tag=f"{tag} withdraw", tol=0.04)

    def _open_drawer(self, which, tries=3):
        """잡고 당기기. **잘 못 잡았거나 안 열리면 놓고 다시 잡는다** — 사람도 손잡이를
        놓치면 다시 잡으므로 시연 데이터로도 자연스럽다."""
        target = self.env.open_target_lower if which == "lower" else self.env.open_target
        Rg = None
        for k in range(tries):
            Rg, held = yield from self._grab_handle(which, f"open {which}",
                                                    dx=(0.0, -0.012, 0.012)[k])
            if held:
                yield from self._slide_handle(which, -target, Rg, f"pull {which}")
                if self.env.slide(which) > target * 0.7:
                    return Rg
            self._say(f"[{which}] 재시도 {k + 1}/{tries} "
                      f"(파지 {held} · 서랍 {self.env.slide(which) * 100:.2f}cm)")
            yield from self._release_handle(Rg, f"retry {which}")
        return Rg

    # ── 전체 계획 ─────────────────────────────────────────────────────────
    def _plan(self):
        env = self.env

        # ⓪ 왼팔 작업 자세. **위로 올리면서 뒤로 물러나며** 자세를 잡는다.
        # 준비 자세 높이(z 0.758)에서 제자리 회전하면, 손이 아직 막대 밖(y 0.34)이어도
        # **팔뚝이** 아래층 손잡이(z 0.71)를 스쳐 6.2 cm 열었다 [접촉 로그로 확인
        # 2026-09-06]. 물러남·상승·회전을 하나의 곡선으로 합친다.
        yield from self._curve("left", [self._hand("left") + np.array([-0.03, 0, 0.09])],
                               R_BAR_H, tag="left ready", tol=0.025)

        # ① 왼팔: 위층 열기 (잡은 채로 유지)
        Rg = yield from self._open_drawer("upper")

        # ② 오른팔: 블록 꺼내기 (왼팔은 손잡이를 계속 잡고 있다).
        # 오른팔 자세는 **여기서** 잡는다. ⓪ 에서 미리 잡아 두면 그 동작이 왼팔의 손잡이
        # 파지에 영향을 줘 당기기가 실패했다 [실측 2026-09-06].
        yield from self._set_pose("right", R_TOP, 0.12, "right ready")
        B = env.block_pos()
        # 접근 높이 0.10: 0.14(z 0.995)는 오른팔 도달 밖이고, 0.08(z 0.935)은 수평 이동 중
        # 손끝(0.897)이 블록 상단(0.890)을 7 mm 차로 스쳐 블록을 밀어냈다 [실측].
        yield from self._travel("right", B + np.array([0, 0, 0.10]), R_TOP, "above block")
        # 블록 바로 위로 **다시 정렬**한다 — 이동 도착 오차와 접근 중 블록이 밀린 양이
        # 겹치면 패드 여유(8 mm)를 넘겨 헛집는다. 정렬은 **절대 높이**로 잡는다(현재 손
        # 높이를 쓰면 도착 오차가 그대로 남아 하강이 모자란다) [실측 2026-09-06].
        B = env.block_pos()
        yield from self._line("right", np.array([B[0], B[1], B[2] + 0.10]), tag="align",
                              v=0.003, tol=0.012)
        B = env.block_pos()
        # 블록을 서랍 벽에서 떼어 두었으므로 **살짝만** 좁힌다 — 감싸고 내려가서 닫는
        # 자연스러운 순서가 산다. 관성으로 8~9 mm 더 닫히는 것까지 감안한 값이다.
        op = yield from self._preclose("right", 82, tag="narrow grip")
        self._say(f"내려가기 전 개구 {op:.1f}mm (서랍 벽에 안 걸리는 폭)")
        ok = yield from self._line("right", B + np.array([0, 0, env.grip_up]),
                                   tag="descend", v=0.003, tol=0.012)
        if not ok:
            B = env.block_pos()
            yield from self._line("right", B + np.array([0, 0, env.grip_up]),
                                  tag="descend2", v=0.002, tol=0.015)
        Rb = self._hand_R("right")
        self.grip["right"] = 1.0
        yield from self._hold(120, wait_arm="right", tag="grasp block")
        self._say(f"오른 그리퍼 개구 {self._opening('right'):.1f}mm "
                  f"(블록 40mm · 41~54 면 정상)")
        # 쥔 물체를 옮길 때는 **느리게**. 0.08 m/s 로 들면 블록이 패드 사이에서 미끄러졌다.
        yield from self._line("right", self._hand("right") + np.array([0, 0, 0.12]), Rb,
                              tag="lift out", tol=0.03, v=0.001)
        yield from self._line("right", CARRY, Rb, tag="carry out", tol=0.03, v=0.002)

        # ③ 왼팔: 위층 닫기 (아직 잡고 있으므로 그대로 민다)
        yield from self._close_fully("upper", Rg, "push upper")
        yield from self._release_handle(Rg, "upper")

        # ④ 왼팔: 아래층 열기 (잡은 채로 유지)
        Rg = yield from self._open_drawer("lower")

        # ⑤ 오른팔: 아래층에 넣기.
        # 놓는 자리를 **로봇 쪽으로 3 cm · 오른팔 쪽으로 1 cm** 옮긴다 [사용자 2026-09-06]:
        #  · -x 3cm: 그대로 내려가면 블록이 **위층 손잡이**(x -0.111~-0.089, z 0.839~0.861)
        #    에 걸려 z 0.887 에서 멈춘다.
        #  · -y 1cm: 서랍 중앙 쪽으로. -4 cm 로 두면 오른벽(-0.202)에서 42 mm 라 그리퍼가
        #    서랍면에 너무 가깝다 [사용자 지적].
        drop = env.slot_pos("lower") + np.array([-0.03, -0.01, 0])
        # 높은 곳에서 x·y 를 먼저 맞추고 **수직으로** 내려간다 — 비스듬히 가면 손잡이를 스친다.
        yield from self._curve("right", [np.array([drop[0], drop[1], CARRY[2]])], Rb,
                               tag="carry in", v=0.003, tol=0.03)
        drop = env.slot_pos("lower") + np.array([-0.03, -0.01, 0])
        yield from self._line("right", drop + np.array([0, 0, env.grip_up + 0.020]), Rb,
                              tag="insert", tol=0.02, v=0.002)
        self.grip["right"] = -1.0
        yield from self._hold(80, wait_arm="right", tag="release block")
        yield from self._line("right", self._hand("right") + np.array([0, 0, 0.14]), Rb,
                              tag="retreat right", tol=0.03, v=0.003)

        # ⑥ 왼팔: 아래층 닫기
        yield from self._close_fully("lower", Rg, "push lower")
        yield from self._release_handle(Rg, "lower")

        # ⑦ 양팔 초기자세 복귀
        for a in self.r.arms:
            yield from self._ramp(a, self.home[a], 170)
        yield from self._hold(70, tag="return home")
