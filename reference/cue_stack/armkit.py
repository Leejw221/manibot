"""시연 수집 공용 도구 — 회전 표현 · 궤적 · 팔 제어.

cue_stack(collect_stack2.py) 과 되집기 cube stack(collect_regrasp.py) 이 함께 쓴다.
task 가 달라도 로봇·컨트롤러 규약은 같기 때문이다 [실측 2026-08-31: RoboCasa PandaOmron 과
robosuite Panda 둘 다 OSC_POSE · output_max 0.05m/0.5rad · 입력은 로봇 베이스 프레임 · 20 Hz].
"""
import numpy as np

GRIP = "gripper0_right_grip_site"
POS_SCALE, ROT_SCALE = 0.05, 0.5          # OSC output_max [실측: 컨트롤러 설정]


def logmap(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (th / (2 * np.sin(th)))


def expmap(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def rot6d(R):
    """회전행렬 -> 6D 표현(앞 두 열). 축각(로그맵)은 크기가 pi 를 넘으면 부호가 통째로
    뒤집혀 같은 자세가 정반대 숫자로 튄다 — 실측에서 스텝간 6.28 rad(=2pi) 점프가 있었다
    [2026-08-30]. 6D 는 그런 불연속이 원리적으로 없어 로봇 학습에서 표준으로 쓴다."""
    return R[:, :2].T.reshape(6)


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def grasp_R(theta):
    """정면을 마주보는 자세. 손가락(로컬 x)=월드 x, 접근축(로컬 z)=월드 -y.
    손목 카메라는 로컬 +z 를 보므로[실측] 이 자세면 정면을 본다."""
    xg = np.array([1., 0., 0.]); zg = np.array([0., -1., 0.])
    return rot_z(theta) @ np.column_stack([xg, np.cross(zg, xg), zg])


def top_R(theta=0.0):
    """위에서 내려다보는 파지 자세. 접근축(로컬 z) = 월드 -z."""
    xg = np.array([1., 0., 0.]); zg = np.array([0., 0., -1.])
    return rot_z(theta) @ np.column_stack([xg, np.cross(zg, xg), zg])


def lerp_traj(p0, R0, p1, R1, step=0.008, rot_step=0.05):
    """두 자세 사이를 제어 스텝 단위로 촘촘히 잇는다. 한 스텝에 실제로 갈 수 있는 거리가
    약 1.25cm 라[실측] 그보다 짧게 잡아 명령이 포화하지 않게 한다."""
    dw = logmap(R1 @ R0.T)
    n = max(2, int(np.ceil(max(np.linalg.norm(p1 - p0) / step,
                               np.linalg.norm(dw) / rot_step))))
    return [(p0 + (p1 - p0) * i / n, expmap(dw * i / n) @ R0) for i in range(1, n + 1)]


class Arm:
    """궤적을 따라가며 **명령을 저역통과 필터**로 매끄럽게 만든다.

    명령은 목표 변위를 OSC output_max 로 나눈 정규화 값이고, 매 제어 스텝(20Hz)마다
    목표를 새로 잡는다. 웨이포인트마다 P 제어를 새로 시작하면 목표 근처 조기 종료 ->
    다음 목표에서 오차 급증이 반복되며 명령 부호가 뒤집힌다 [v7 실측: 문 구간 8.6~9.3%].
    """

    def __init__(self, env, rec=None, alpha=0.35):
        self.env, self.rec, self.alpha = env, rec, alpha
        bid = env.sim.model.body_name2id(env.robots[0].robot_model.root_body)
        self.Rb = env.sim.data.body_xmat[bid].reshape(3, 3).copy()
        self.prev = np.zeros(6)

    def pose(self):
        d = self.env.sim.data
        return d.get_site_xpos(GRIP).copy(), d.get_site_xmat(GRIP).reshape(3, 3).copy()

    def _step(self, pos, R_des, grip):
        p, R = self.pose()
        raw = np.concatenate([self.Rb.T @ (pos - p) / POS_SCALE,
                              self.Rb.T @ logmap(R_des @ R.T) / ROT_SCALE])
        cmd = self.alpha * np.clip(raw, -1, 1) + (1 - self.alpha) * self.prev
        self.prev = cmd
        a = np.zeros(self.env.action_dim)
        a[0:6] = cmd
        a[6] = grip
        self.env.step(a)
        if self.rec:
            self.rec(a)

    def follow(self, traj, grip, settle=0, tol_p=0.006, tol_r=0.10):
        """궤적을 따라간 뒤 **도착을 확인**한다. 궤적만 재생하고 넘어가면 못 따라간 오차가
        다음 동작으로 누적된다 [실측 2026-08-29: 대기 자세에서 13cm 뜬 채로 다음 문에
        접근 -> 손잡이 위 18cm 에서 허공을 쥐고 당김]. 정착 명령은 목표가 고정이라
        오차가 작아 명령도 작고, 저역통과와 함께라 떨림을 만들지 않는다."""
        for pos, R in traj:
            self._step(pos, R, grip)
        tgt_p, tgt_R = traj[-1]
        for _ in range(max(settle, 1)):
            p, R = self.pose()
            if (np.linalg.norm(tgt_p - p) < tol_p
                    and np.linalg.norm(logmap(tgt_R @ R.T)) < tol_r):
                break
            self._step(tgt_p, tgt_R, grip)
        p, _ = self.pose()
        return float(np.linalg.norm(tgt_p - p))

    def goto(self, pos, R_des, grip, settle=60, **kw):
        p, R = self.pose()
        return self.follow(lerp_traj(p, R, np.asarray(pos, float), R_des, **kw), grip, settle)

    def hold(self, grip, n):
        p, R = self.pose()
        for _ in range(n):
            self._step(p, R, grip)

    def grip_ramp(self, g0, g1, n):
        """그리퍼 명령을 n 스텝에 걸쳐 선형으로 바꾼다 (사용자 지시 2026-08-31: 천천히).
        계단으로 주면 손가락이 튕기듯 여닫혀 블록을 밀어낸다. 램프 중에는 팔이 멈춰 있지만
        **그리퍼 관절은 계속 움직이므로** 관측이 정지하지 않는다 — v7 의 흡수 상태 문제
        (같은 관측이 이어져 정책이 그 자리에 갇히는 것)를 이쪽은 피한다."""
        p, R = self.pose()
        for i in range(1, n + 1):
            self._step(p, R, g0 + (g1 - g0) * i / n)


def to_abs_action(eef_pos, eef_rot, grip_cmd):
    """저장용 action = **다음 시점의 절대 eef 자세** (위치3 + 6D회전6 + 그리퍼명령1 = 10).

    컨트롤러에 보낸 정규화 델타를 그대로 저장하면 [-1,1] 로 잘려서 6cm 가려던 것과 20cm
    가려던 것이 같은 1.0 이 된다 — 정보가 사라지고 on/off 신호가 된다 [v7 실측 포화 28.8%].
    절대 자세로 두면 잘라내기가 없어 궤적이 정확히 복원되고, 평가에서
    (예측자세 - 현재자세)/0.05 로 명령을 만들면 된다 (사용자 지시 2026-08-30).
    """
    nxt = lambda x: np.concatenate([x[1:], x[-1:]], 0)
    return np.concatenate([nxt(eef_pos), nxt(eef_rot),
                           np.asarray(grip_cmd).reshape(-1, 1)], 1).astype(np.float32)
