"""팔 자세를 푸는 최소 IK — 환경이 "이 task 를 할 수 있는 초기 자세"를 잡는 데 쓴다.

**왜 필요한가**: OSC 만으로 목표 자세에 가면 영공간이 아무데나 떨어진다. 전자레인지 문
손잡이(수평 접근)로 가면 Panda 의 joint5 가 -166°(하한)에 붙어 버려서, 문을 따라 손목을
83° 돌려야 하는 다음 단계에서 못 움직인다 [실측 2026-09-05]. robosuite OSC 는 영공간에서
**리셋 자세**를 목표로 당기므로, 리셋 자세를 호 전체가 여유를 갖는 해로 잡아주면 해결된다.

야코비안 기반 damped least-squares 를 쓴다. scipy least_squares 로는 SO(3) 잔차가 잘 안
풀렸다(자세 오차 10~45° 잔류).
"""

import numpy as np
from robosuite.utils.transform_utils import mat2quat, quat2axisangle


def frame(approach, close, close_axis="x"):
    """접근축(eef z)·개폐축으로 목표 회전행렬을 만든다.

    ⚠️ **개폐축이 로봇마다 다르다** [실측 2026-09-05]:
        Panda : 손가락이 eef **x** 로 ±0.021 벌어지고 +z 로 뻗는다  -> close_axis="x"
        NERO  : 손가락이 eef **y** 로 ±0.025 벌어지고 +z 로 뻗는다  -> close_axis="y"
    이걸 틀리면 손목을 90° 틀어 명령하게 되고, 도달 측정이 통째로 실패로 나온다
    (NERO 첫 측정이 실제로 그랬다).
    """
    z = np.asarray(approach, float)
    z = z / np.linalg.norm(z)
    c = np.asarray(close, float)
    c = c - z * (c @ z)
    c = c / np.linalg.norm(c)
    if close_axis == "x":
        return np.stack([c, np.cross(z, c), z], axis=1)
    if close_axis == "y":
        return np.stack([np.cross(c, z), c, z], axis=1)
    raise ValueError(f"close_axis 는 'x' 또는 'y' — 받은 값 {close_axis!r}")


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def roll(R, phi):
    """접근축 둘레로 굴린다. 세로 막대 손잡이는 굴려도 잡히므로 여유 있는 해를 고를 수 있다."""
    return R @ rot_z(phi)


class ArmIK:
    """robosuite env 의 한 팔에 대한 DLS IK. sim 을 직접 건드리므로 리셋 전에만 쓸 것."""

    def __init__(self, env, arm=None):
        import mujoco
        self._mujoco = mujoco
        r = env.robots[0]
        arm = arm or r.arms[0]
        self.arm = arm
        self.m, self.d = env.sim.model._model, env.sim.data._data
        # 양팔 로봇에서는 **그 팔의 관절만** 푼다 — 전 관절을 넘기면 반대 팔이 같이 움직인다.
        joints = [n for n in r.robot_model.joints if f"_{arm}_joint" in n] or r.robot_model.joints
        self.joints = joints
        self.jidx = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in joints])
        self.dofs = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in joints])
        self.sid = r.eef_site_id[arm]
        self.rng = np.array([self.m.jnt_range[self.m.joint(n).id] for n in joints])
        self.mid = self.rng.mean(1)
        self.n = len(self.jidx)

    def _fk(self, q):
        self.d.qpos[self.jidx] = q
        self._mujoco.mj_kinematics(self.m, self.d)
        self._mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def margin(self, q):
        """각 관절이 한계까지 남긴 여유 중 최소값 [rad]."""
        return float(np.minimum(q - self.rng[:, 0], self.rng[:, 1] - q).min())

    def solve(self, pos, R, q0, iters=250, lam=0.05, w_null=0.03):
        lo, hi = self.rng[:, 0] + 0.02, self.rng[:, 1] - 0.02
        q = np.clip(np.asarray(q0, float), lo, hi)
        Jp, Jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            p, Rc = self._fk(q)
            e = np.concatenate([pos - p, quat2axisangle(mat2quat(R @ Rc.T))])
            if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
                break
            self._mujoco.mj_jacSite(self.m, self.d, Jp, Jr, self.sid)
            J = np.vstack([Jp[:, self.dofs], Jr[:, self.dofs]])
            A = np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), np.hstack([e[:, None], J]))
            null = (np.eye(self.n) - J.T @ A[:, 1:]) @ (self.mid - q)
            q = np.clip(q + np.clip(J.T @ A[:, 0], -0.2, 0.2) + w_null * null, lo, hi)
        p, Rc = self._fk(q)
        return (q, float(np.linalg.norm(p - pos)),
                float(np.linalg.norm(quat2axisangle(mat2quat(R @ Rc.T)))))

    def solve_path(self, poses, seeds=8, tol_pos=0.005, tol_ang=0.035, rng=None):
        """자세열 전체를 이어서 푼다. 관절 여유가 가장 큰 가지를 고른다.

        하나씩 따로 풀면 가지가 갈려 중간에 손목을 반바퀴 돌려야 하는 해가 나온다.
        앞 해를 다음 시드로 쓰는 것이 이어짐을 보장한다.
        """
        rng = rng or np.random.RandomState(0)
        best = None
        for k in range(seeds):
            q = self.mid.copy() if k == 0 else rng.uniform(self.rng[:, 0] + 0.3,
                                                           self.rng[:, 1] - 0.3)
            qs, worst = [], np.inf
            for pos, R in poses:
                q, ep, ea = self.solve(pos, R, q)
                if ep > tol_pos or ea > tol_ang:
                    qs = None
                    break
                qs.append(q.copy())
                worst = min(worst, self.margin(q))
            if qs is not None and (best is None or worst > best[0]):
                best = (worst, np.array(qs))
        return best
