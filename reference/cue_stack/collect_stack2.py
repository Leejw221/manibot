"""색 표식 -> 거울상 쌓기, **2판**. 문 동작을 바꾸고 궤적을 매끄럽게 만든다.

1판(collect_stack.py)의 문제 두 가지를 고친다.
  떨림   호를 22개 웨이포인트로 쪼개 웨이포인트마다 P제어를 새로 시작했다. 목표 근처에서
         조기 종료 -> 다음 목표에서 오차 급증이 반복되며 명령 부호가 12스텝마다 뒤집혔다
         (실측: 문 구간 8.6~9.3%). 이제 **제어 스텝(20Hz) 단위로 촘촘한 궤적 하나**를 깔고
         목표 변위를 그대로 정규화해 넣는다(OSC output_max 0.05m 로 나눔) + 저역통과.
  손목    손잡이를 잡은 채 들여다봐서 손목 카메라가 문짝만 봤다(표식 색 강도 0.0).
         이제 손잡이를 놓고 **캐비닛 입구 앞**으로 가서 본다. 손목 카메라는 손 좌표계
         +z 를 보고(실측), grasp_R 의 z 가 -y(캐비닛 안)라 그 자세면 선반이 정면에 온다.

문 동작 순서 (사용자 설계):
  손잡이로 당겨 일부 개방 -> 놓고 패널 **안쪽 면**을 밀어 완전 개방 -> 손목으로 표식 확인
  -> 바깥 면을 **밀어** 완전 폐쇄.
  ("여는 것"은 당기는 방향이다 — 손잡이가 열릴수록 +y 로 나온다[실측]. 그래서 더 열려면
   패널 뒤로 손을 넣어 밀고, 닫을 때는 앞에서 민다.)
"""
import argparse, itertools, json, zlib
import numpy as np
import robocasa, robosuite
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from pathlib import Path
import scene
from armkit import (GRIP, POS_SCALE, ROT_SCALE, logmap, expmap,
                    rot6d, rot_z, grasp_R, top_R, lerp_traj, Arm, to_abs_action)

ROOT = Path(robocasa.__file__).resolve().parent.parent
DATASET = ROOT / "datasets/v1.0/pretrain/composite/RestockBowls/20250725/lerobot"
STANDBY_X = 6.617                         # 캐비닛 중앙(두 힌지 6.177/7.057 의 중점)

PHASES = ["r1_approach", "r1_pull", "r1_push_open", "r1_peek", "r1_push_close", "r1_home",
          "r2_approach", "r2_pull", "r2_push_open", "r2_peek", "r2_push_close", "r2_home",
          "pick_approach", "pick_grasp", "pick_lift", "place", "release", "home_end"]
STAGES = ["A-1_왼쪽_열기", "A-2_왼쪽_닫기", "B-1_오른쪽_열기", "B-2_오른쪽_닫기",
          "C_초록_쌓기", "D_파랑_쌓기", "E_종료"]
S_OPEN = {"left": 0, "right": 2}
S_CLOSE = {"left": 1, "right": 3}
S_STACK = {"green": 4, "blue": 5}
S_END = 6


class Door:
    """한 쪽 문의 기하 — 손잡이 호와 패널 앞/뒤 밀기 지점."""

    def __init__(self, env, side):
        sim, cab = env.sim, env.cab
        jn = [n for n in cab.door_joint_names if side in n][0]
        jid = sim.model.joint_name2id(jn)
        self.adr = sim.model.jnt_qposadr[jid]
        bid = sim.model.jnt_bodyid[jid]
        R = sim.data.body_xmat[bid].reshape(3, 3)
        self.center = (sim.data.body_xpos[bid] + R @ sim.model.jnt_pos[jid]).copy()
        self.h0 = sim.data.get_site_xpos(
            f"{cab.name}_{side}_door_handle_default_site").copy()
        self.sign = 1.0 if side == "right" else -1.0
        self.side = side

    def handle(self, t):
        v = self.h0[:2] - self.center[:2]
        c, s = np.cos(t), np.sin(t)
        return np.array([self.center[0] + c * v[0] - s * v[1],
                         self.center[1] + s * v[0] + c * v[1], self.h0[2]])

    def normal(self, t):
        """패널의 안쪽(캐비닛 쪽) 법선. 각도가 커지면 패널 위 점들은 -normal 방향으로 움직인다
        — 즉 **더 열려면 안쪽 면을 밀어야** 하고, 닫으려면 바깥 면을 밀어야 한다."""
        return rot_z(t) @ np.array([0., -1., 0.])

    def push_open(self, t):
        """더 여는 밀기: 그리퍼를 패널 **안쪽** 면에 대고 바깥 방향으로 민다.
        손잡이 site 는 패널 중심면보다 1.2cm 바깥에 있고 패널 두께 반이 1.4cm 이라
        안쪽 면은 손잡이에서 약 2.6cm 안쪽이다[실측 geom size]. 손끝이 거기 닿도록 1cm 더."""
        return self.handle(t) + self.normal(t) * 0.036

    def push_close(self, t, frac=0.55):
        """닫는 밀기: 바깥 면을 안쪽으로 민다.
        **자유단(손잡이) 이 아니라 힌지 쪽 frac 지점을 민다** — 오른쪽 문은 활짝 열리면
        손잡이가 로봇에서 48cm 바깥으로 나가 팔이 억지로 뻗다가 문을 되레 밀어버렸다
        [실측 2026-08-28: 닫힘 +1.462]. 힌지 쪽은 훨씬 가까워 자세가 안정적이다."""
        v = self.h0[:2] - self.center[:2]
        c, sn = np.cos(t), np.sin(t)
        p = np.array([self.center[0] + (c * v[0] - sn * v[1]) * frac,
                      self.center[1] + (sn * v[0] + c * v[1]) * frac, self.h0[2]])
        return p - self.normal(t) * 0.045

    def angle(self, env):
        return float(env.sim.data.qpos[self.adr])


def ease(t0, t1, ds=0.008, radius=0.45, mode="out"):
    """열 때는 끝으로 갈수록 **감속**, 닫을 때는 처음에 **천천히** 출발한다
    (사용자 제안 2026-08-30). 둘 다 '활짝 열린 각도' 근처에 시간을 몰아주므로,
    표식이 보이는 구간이 길어지면서 전체 길이는 거의 안 늘어난다.
      mode="out"  빠르게 출발 -> 느리게 도착   (여는 동작)
      mode="in"   느리게 출발 -> 빠르게 도착   (닫는 동작)
    이징 때문에 구간별 간격이 달라지므로 웨이포인트 수를 1.6배로 잡아 최대 간격이
    한 스텝에 갈 수 있는 거리(약 1.25cm)를 넘지 않게 한다."""
    n = max(4, int(np.ceil(abs(t1 - t0) * radius / ds * 1.6)))
    u = np.linspace(0.0, 1.0, n)
    u = np.sin(np.pi / 2 * u) if mode == "out" else 1 - np.cos(np.pi / 2 * u)
    return t0 + (t1 - t0) * u


def dense(f, t0, t1, ds=0.008, radius=0.45):
    """각도 구간을 호 길이 기준으로 촘촘히 나눈다.
    ds 를 도달 한계(1.25cm/스텝)보다 넉넉히 작게 잡아야 저역통과 필터로 지연된 팔이
    손잡이를 놓치지 않는다 [실측 2026-08-28: ds=0.010 에서 문이 0.99->0.458 만 닫힘]."""
    n = max(2, int(np.ceil(abs(t1 - t0) * radius / ds)))
    return np.linspace(t0, t1, n)


def search_round(env, arm, state, side, base, args, rng, home):
    """당겨서 일부 개방 -> 놓고 안쪽 면을 밀어 완전 개방 -> 손목으로 확인 -> 밀어서 폐쇄."""
    d = Door(env, side)
    off = rng.uniform(0.006, 0.011)
    # 밀어서 더 여는 동작은 기각했다 [실측 2026-08-28, 3회 시도]: 45cm 폭 문이 반쯤 열린
    # 상태에서 패널 뒤로 그리퍼를 넣으려면 좁은 틈을 통과해야 하는데, 매번 패널을 건드려
    # 오히려 닫혔다(0.45->0.36, 0.50->0.00). 여는 건 손잡이로 끝까지 당기고, 미는 것은
    # **닫기에만** 쓴다(닫기는 -0.002 로 완벽히 동작).
    t2 = d.sign * (args.open_angle + rng.uniform(-0.04, 0.06))

    state["phase"] = base + 0
    # 두 손잡이가 8cm 밖에 안 떨어져 있고 대기 자세가 그 사이(x=6.617)라, 바로 접근하면
    # 옆 문 손잡이를 스쳐 파지가 빗나간다 [실측 2026-08-29: 오른쪽 문 처리 후 왼쪽 문
    # 열기가 2회 연속 0.00]. 목표 손잡이 **정면**으로 먼저 나온 뒤 들어간다.
    arm.goto(d.h0 + [0, 0.32, 0], grasp_R(0), -1)
    arm.goto(d.h0 + [0, 0.16, 0], grasp_R(0), -1)
    # **접근하면서 그리퍼를 닫는다.** 예전엔 손잡이에 도착한 뒤 hold(1,12) 로 12스텝
    # 정지한 채 쥐었는데, 그 구간이 흡수 상태가 됐다 — rollout 에서 정책이 오른쪽
    # 손잡이를 잡은 채 1,570스텝 멈췄고, 그때 예측한 이동 요구가 0.4cm(정상 구간 5cm)
    # 였다 [실측 2026-08-30, 정책 출력 직접 확인]. 움직이면서 쥐면 정지 구간이 없다.
    # 손잡이까지 벌린 채 들어간 뒤, **정착 없이** 그립 명령만 바꾸고 곧바로 여는 궤적으로
    # 이어붙인다. 도착해서 멈춰 쥐면(hold) 그 구간이 흡수 상태가 되고 — rollout 에서 정책이
    # 손잡이를 쥔 채 1,570스텝 멈췄다 — 정착 루프로 잘게 다가가도 마찬가지다(마지막 14스텝
    # 0.004 cm/스텝). 움직이는 도중에 쥐면 정지 구간이 원리적으로 없다 [2026-08-31].
    arm.goto(d.h0 + [0, off, 0], grasp_R(0), -1, step=0.005)
    for _ in range(6):                       # 그립 명령만 닫힘으로 — 자세 목표는 그대로
        arm._step(d.h0 + [0, off, 0], grasp_R(0), 1)

    state["phase"] = base + 1                     # 감속하며 연다 (손잡이를 놓지 않는다)
    arm.follow([(d.handle(t) + [0, off, 0], grasp_R(t)) for t in ease(0, t2, mode="out")], 1)
    a_pull = a_open = d.angle(env)

    state["phase"] = base + 4                     # 같은 흐름으로 바로 되돌려 닫는다
    # 놓았다 다시 잡는 구간을 없앴다 — 재파지가 실패 요인이었고(지난 판에서 여러 번 버림)
    # 오른쪽 문에서는 놓고 기다리는 시간이 89스텝으로 길었다 (사용자 지적 2026-08-30).
    for pos_, R_ in [(d.handle(t) + [0, off, 0], grasp_R(t))
                     for t in ease(a_open, -d.sign * 0.14, mode="in")]:
        arm._step(pos_, R_, 1)
        if abs(d.angle(env)) < 0.02:
            break
    # 완전히 0도로 닫을 필요는 없다 (사용자 지시 2026-08-30). 실측상 **0.45 rad(26도)까지
    # 열려 있어도 표식은 전혀 안 보이므로** 은닉은 유지된다. 그래서 마무리 정지를 줄인다.
    arm.hold(1, 3)
    a_close = d.angle(env)
    arm.hold(-1, 5)

    back = 0.0

    return dict(pull=a_pull, open=a_open, close=a_close, back=back)


def blk_pos(env, name):
    a = env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(name + "_joint")]
    return env.sim.data.qpos[a:a + 3].copy()


def collect_one(env, args, rng, cue_side, cue_color, first_side):
    st = LU.get_episode_states(DATASET, args.base_episode)
    xml = scene.build_xml(LU.get_episode_model_xml(DATASET, args.base_episode),
                          cue_side, cue_color)
    reset_to(env, dict(states=scene.pad_state(st[0], 123, 121), model=xml,
                       ep_meta=json.dumps(LU.get_episode_meta(DATASET, args.base_episode))))
    sim = env.sim
    for n in ("obj1_joint0", "obj2_joint0"):
        a = sim.model.jnt_qposadr[sim.model.joint_name2id(n)]
        sim.data.qpos[a:a + 3] = [0, 0, -5]
    for n, xy in (("blk_green", scene.GREEN_XY), ("blk_blue", scene.BLUE_XY)):
        a = sim.model.jnt_qposadr[sim.model.joint_name2id(n + "_joint")]
        j = rng.uniform(-args.blk_jitter, args.blk_jitter, 2)
        sim.data.qpos[a:a + 7] = [xy[0] + j[0], xy[1] + j[1],
                                  scene.COUNTER_Z + scene.HALF, 1, 0, 0, 0]
    sim.forward()
    init = {n: blk_pos(env, n) for n in ("blk_green", "blk_blue")}

    keys = ("img_global", "img_wrist", "img_bench", "eef_pos", "eef_rot",
            "gripper", "action", "phase")
    buf = {k: [] for k in keys}
    state = {"phase": 0}

    def rec(a):
        for k, cam in (("img_global", "globalview"), ("img_wrist", "robot0_eye_in_hand"),
                       ("img_bench", "benchview")):
            buf[k].append(sim.render(camera_name=cam, width=args.hw, height=args.hw)[::-1].copy())
        p, R = sim.data.get_site_xpos(GRIP).copy(), sim.data.get_site_xmat(GRIP).reshape(3, 3)
        buf["eef_pos"].append(p)
        buf["eef_rot"].append(rot6d(R))
        buf["gripper"].append(sim.data.qpos[
            env.robots[0]._ref_gripper_joint_pos_indexes["right"]].copy())
        buf["action"].append(a[:7].copy())
        buf["phase"].append(state["phase"])

    arm = Arm(env, alpha=args.alpha)
    p0, R0 = arm.pose()
    arm.goto(p0 + rng.uniform(-args.jitter, args.jitter, 3), R0, -1)   # 기록 전 흔들기
    arm.rec = rec
    home = arm.pose()

    res, rounds = {}, [first_side]
    res["r1"] = search_round(env, arm, state, first_side, 0, args, rng, home)
    if cue_side != first_side:
        other = "left" if first_side == "right" else "right"
        rounds.append(other)
        res["r2"] = search_round(env, arm, state, other, 6, args, rng, home)

    tgt = "blk_" + cue_color
    bp = blk_pos(env, tgt)
    top_z = scene.COUNTER_Z + 2 * scene.HALF
    red_top = scene.COUNTER_Z + 2 * scene.BASE_HALF
    state["phase"] = 12
    # 문 작업을 마친 자세에서 바로 블록으로 가면 팔이 특정 배치에 갇혀 먼 쪽 블록(파랑,
    # x=6.51)에 못 닿는다 [실측 2026-08-29: right/blue/firstL 이 13회 연속 쌓기 15.1cm
    # 로 동일 실패 — 난수를 바꿔도 같은 값이라 무작위 변동이 아니다].
    # 초기 자세를 한 번 거쳐 배치를 리셋한다. 어차피 여기서 그리퍼가 아래로 전환되므로
    # 라운드 사이의 어색한 전환(사용자 지적)과는 무관하다.
    # 초기 자세 경유를 뺐다 — 대기·놓기 구간이 있던 시절 먼 블록에 못 닿아 넣었는데,
    # 그 구간들을 다 없앤 지금은 문에서 곧바로 블록으로 간다. 경유가 남아 있으면 도달이
    # 안 돼 정착 루프 60스텝을 태우며 제자리에 머문다 [실측 2026-08-30: pick접근 331스텝
    # 중 앞부분 이동량 0.03cm/스텝].
    arm.goto([bp[0], bp[1], bp[2] + 0.30], top_R(), -1)
    arm.goto([bp[0], bp[1], bp[2] + 0.13], top_R(), -1)
    state["phase"] = 13
    arm.goto([bp[0], bp[1], bp[2] + 0.004], top_R(), -1, step=0.005)
    arm.hold(1, 14)
    state["phase"] = 14
    arm.goto([bp[0], bp[1], top_z + 0.14], top_R(), 1)
    res["lift_z"] = float(blk_pos(env, tgt)[2])
    state["phase"] = 15
    drop_z = red_top + scene.HALF + 0.015
    arm.goto([scene.RED_XY[0], scene.RED_XY[1], top_z + 0.14], top_R(), 1)
    arm.goto([scene.RED_XY[0], scene.RED_XY[1], drop_z], top_R(), 1, step=0.005)
    state["phase"] = 16
    arm.hold(-1, 14)
    state["phase"] = 17
    arm.goto([scene.RED_XY[0], scene.RED_XY[1], top_z + 0.16], top_R(), -1)
    arm.goto(home[0], home[1], -1)

    fp = blk_pos(env, tgt)
    other = "blk_green" if cue_color == "blue" else "blk_blue"
    res["stack_dxy"] = float(np.linalg.norm(fp[:2] - np.array(scene.RED_XY)))
    res["stack_dz"] = float(fp[2] - (red_top + scene.HALF))
    res["other_moved"] = float(np.linalg.norm(blk_pos(env, other)[:2] - init[other][:2]))
    out = {k: np.array(v) for k, v in buf.items()}
    # **action 을 "다음 시점의 절대 eef 자세"로 저장한다** (사용자 지시 2026-08-30).
    # 컨트롤러에 보낸 정규화 델타를 그대로 저장하면 [-1,1] 로 잘려서 6cm 가려던 것과
    # 20cm 가려던 것이 같은 1.0 이 된다 — 정보가 사라지고 on/off 신호가 된다
    # (v7 실측 포화 28.8%). 절대 자세로 두면 잘라내기가 없어 궤적이 정확히 복원되고,
    # 평가에서 (예측자세 - 현재자세)/0.05 로 명령을 만들면 된다.
    #   action = 다음 eef 위치3 + 다음 eef 회전6(6D) + 그리퍼 명령1 = 10차원
    pos = out["eef_pos"]
    nxt = lambda x: np.concatenate([x[1:], x[-1:]], 0)
    out["action"] = np.concatenate([nxt(pos), nxt(out["eef_rot"]),
                                    out["action"][:, 6:7]], 1).astype(np.float32)
    out["stage"] = stage_from_phase(out["phase"], rounds, cue_color)
    return out, res


def stage_from_phase(phase, rounds, cue_color):
    """열기(접근·당김·밀어열기·확인)=X-1, 닫기(밀어닫기·복귀)=X-2."""
    s = np.full_like(phase, S_END)
    for k, side in enumerate(rounds):
        b = 6 * k
        s[(phase >= b) & (phase <= b + 3)] = S_OPEN[side]
        s[(phase == b + 4) | (phase == b + 5)] = S_CLOSE[side]
    s[(phase >= 12) & (phase <= 16)] = S_STACK[cue_color]
    s[phase == 17] = S_END
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-condition", type=int, default=10)
    ap.add_argument("--only", default=None, help="예: right,green,left  (표식위치,색,먼저열문)")
    ap.add_argument("--base-episode", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.035)
    ap.add_argument("--blk-jitter", type=float, default=0.010)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pull-angle", type=float, default=0.55, help="손잡이로 여는 각")
    ap.add_argument("--open-angle", type=float, default=1.05,
                    help="여는 각. 표식은 1.0 rad 이상에서만 보인다[실측]")
    ap.add_argument("--obs-x", type=float, default=6.90)
    ap.add_argument("--obs-y", type=float, default=-3.85)
    ap.add_argument("--obs-z", type=float, default=1.42)
    ap.add_argument("--peek-sweep", type=float, default=0.08,
                    help="관찰 구간에서 앞뒤로 훑는 폭 (m). 정지 구간을 없애기 위한 것")
    ap.add_argument("--peek-reps", type=int, default=2, help="훑기 왕복 횟수")
    ap.add_argument("--alpha", type=float, default=0.35, help="명령 저역통과 계수")
    ap.add_argument("--hw", type=int, default=128)
    ap.add_argument("--max-attempts", type=int, default=40,
                    help="한 칸에서 목표 개수를 채우기 위한 최대 시도 수")
    ap.add_argument("--video-dir", default=None,
                    help="주면 저장된 에피소드마다 obs 영상(글로벌|손목) 을 남긴다")
    ap.add_argument("--out", default="/home/moai/claude/stack_data_v2")
    args = ap.parse_args()

    m = LU.get_env_metadata(DATASET); kw = m["env_kwargs"]; kw["env_name"] = m["env_name"]
    kw.update(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
              camera_names=["robot0_eye_in_hand"], camera_heights=args.hw, camera_widths=args.hw)
    env = robosuite.make(**kw)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.only:
        cs, cc, fs = args.only.split(",")
        cells = [(cs, cc, fs)]
    else:   # 8칸 격자: 표식위치 2 x 색 2 x 먼저 여는 문 2
        cells = list(itertools.product(("right", "left"), ("green", "blue"), ("right", "left")))
    # 시연 데이터에는 실패가 섞이면 안 된다 — 정책이 그걸 배운다. **성공한 것만 저장**하고
    # 칸마다 목표 개수를 채울 때까지 다시 뽑는다 (사용자 지시 2026-08-29).
    n_ok, n_try = 0, 0
    for cs, cc, fs in cells:
        got, attempt = 0, 0
        while got < args.per_condition and attempt < args.max_attempts:
            key = f"{cs}|{cc}|{fs}|{attempt}|{args.seed}".encode()
            rng = np.random.default_rng(zlib.crc32(key))
            buf, res = collect_one(env, args, rng, cs, cc, fs)
            attempt += 1; n_try += 1
            r = [res["r1"]] + ([res["r2"]] if "r2" in res else [])
            # 판정은 시연의 질에 직결되는 것만: 문이 충분히 열렸는가·완전히 닫혔는가·
            # 맞는 위치에 쌓았는가. 대기 자세 도달 오차(back)는 무관해서 뺐다.
            # 닫힘 기준 0.12 -> 0.30 rad: 0.45 rad 까지도 표식이 안 보인다[실측]. 여유를
            # 두면 억지로 밀어붙이지 않아 팔이 무리하지 않는다.
            good = (all(abs(x["open"]) > 0.9 and abs(x["close"]) < 0.30 for x in r)
                    and res["stack_dxy"] < 0.03 and abs(res["stack_dz"]) < 0.015)
            det = " | ".join(f"열림{x['open']:+.2f} 닫힘{x['close']:+.3f} 복귀{x['back']*100:.0f}cm"
                             for x in r)
            if good:
                tag = f"{cs}_{cc}_first{fs[0].upper()}_{got:02d}"
                np.savez_compressed(out / f"ep_{tag}.npz", cue_side=cs, cue_color=cc,
                                    first_side=fs, attempt=attempt, **buf)
                if args.video_dir:
                    # **정책이 실제로 보는 두 카메라**로 저장한다 (글로벌 | 손목).
                    # 작업대 뷰는 사람이 볼 때만 쓰고 여기엔 안 넣는다.
                    import imageio, os
                    os.makedirs(args.video_dir, exist_ok=True)
                    imageio.mimsave(f"{args.video_dir}/{tag}.mp4",
                                    np.concatenate([buf["img_global"], buf["img_wrist"]],
                                                   axis=2)[::2], fps=20)
                got += 1; n_ok += 1
                print(f"[{tag}] {len(buf['phase']):4d}프레임  {det} | 쌓기 "
                      f"{res['stack_dxy']*100:4.1f}cm  저장 ({got}/{args.per_condition})",
                      flush=True)
            else:
                print(f"  버림 {cs}/{cc}/first{fs[0].upper()} 시도{attempt}: {det} | 쌓기 "
                      f"{res['stack_dxy']*100:4.1f}cm", flush=True)
        if got < args.per_condition:
            print(f"!! {cs}/{cc}/first{fs} {got}/{args.per_condition} 만 채움 "
                  f"({attempt}회 시도)", flush=True)
    print(f"=== 저장 {n_ok}/{len(cells)*args.per_condition} · 총 시도 {n_try} "
          f"(성공률 {n_ok/max(n_try,1)*100:.0f}%) ===")


if __name__ == "__main__":
    main()
