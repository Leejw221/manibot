"""배포 데이터 수집 — 정책을 굴리다 **사람이 트리거하면 스크립트가 교정**한다.

APO Algorithm 1 의 `Deployment(pi_theta, D_h)` 에 해당한다.  base policy 를 배포해
실패로 가는 상황을 사람이 보고 `i` 로 개입을 켠다.  교정 행동은 전문가(`sim.expert`)가
현재 상태에서 이어서 낸다 — VR 텔레오퍼가 없어도 되고 시드를 붙일 수 있다.

**라벨은 APO 원문 규약을 쓴다** [원문 직접, APO Alg.1 L15-16 · §3.1]:
    c = 2   사람(여기선 스크립트) 이 교정한 프레임
    c = 1   정책이 실행한 프레임
    c = 0   각 개입 시작 **직전 K 프레임** — 실패로 이어진 행동.  재라벨로 만든다
K 는 논문 부록 B 의 10 을 기본값으로 둔다 ("the last 10 actions before human intervention").

키 (`teleoperators/keyboard_trigger.py`, LeRobot 규약 + i):
    i = 개입 토글 · →/n = 다음 에피소드 · ←/r = 다시 · s = 저장 · ESC/q = 중단

사용:
    python -m manibot.scripts.collect_intervention task=square_scripted \\
        checkpoint_path=outputs/.../checkpoints/step_0000050000 n_episodes=10
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from manibot.policies.factory import make_policy
from manibot.rollout import make_predict_fn
from manibot.utils.checkpoints import load_ema_weights, load_model_weights
from manibot.utils.dataset_utils import create_dataset_stats
from manibot.utils.logger import setup_logging
from manibot.utils.task_utils import derive_task_meta, make_eval_env, resolve

logger = logging.getLogger(__name__)

C_PREINTV, C_POLICY, C_INTV = 0, 1, 2


# cv2.waitKey 가 주는 코드 -> 우리 키 이름. Wayland 에서 영상 창이 곧 조작면이다.
#
# ⚠ **`& 0xFF` 를 씌우면 안 된다.** Linux 에서 cv2 는 방향키를 X11 keysym 으로 주는데
# (왼쪽 65361) 하위 바이트만 취하면 65361 & 255 = 81 = 'Q' 가 되어 **왼쪽 화살표가
# 종료 명령이 된다** (2026-09-08 실측: 왼쪽을 누르니 창이 꺼졌다). 위/오른쪽도 각각
# 'R'(다시)/'S'(저장) 로 둔갑한다. 그래서 전체 코드를 그대로 보고 표로 푼다.
# 빌드마다 방향키 코드가 달라서 세 계열을 모두 넣는다. 우리 문자 명령(i/n/r/s/q =
# 105/110/114/115/113)은 81~84 와 겹치지 않으므로 그 구간은 방향키로 읽어도 안전하다.
_CV_KEYS = {
    27: "esc", 13: "enter", 32: "space",
    65361: "left", 65362: "up", 65363: "right", 65364: "down",          # X11 keysym
    2424832: "left", 2490368: "up", 2555904: "right", 2621440: "down",  # 일부 Windows/Qt 빌드
    81: "left", 82: "up", 83: "right", 84: "down",                      # 하위바이트만 오는 빌드
}


def _show(raw, cams, res, stage, ep, step, intervening):
    """정책이 보는 카메라를 크게 띄우고 상태를 겹쳐 그린다. 반환: 눌린 키 이름 또는 None."""
    tiles = [raw.sim.render(width=res, height=res, camera_name=c)[::-1] for c in cams]
    im = np.concatenate(tiles, axis=1)[:, :, ::-1].copy()      # RGB -> BGR
    on = intervening
    cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 160) if on else (40, 40, 40), -1)
    cv2.putText(im, f"ep{ep} step{step}  {'INTERVENING (i=off)' if on else 'policy (i=on)'}"
                f"  [{stage}]", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imshow("manibot deployment", im)
    k = cv2.waitKey(1)
    if k < 0:
        return None
    name = _CV_KEYS.get(k)
    if name is not None:
        return name
    return chr(k) if 32 < k < 127 else None


def _raw(env):
    """lerobot 래퍼를 벗겨 robosuite env 를 꺼낸다 — 전문가와 기록이 이걸 쓴다."""
    e = env
    for _ in range(4):
        if hasattr(e, "robots"):
            return e
        e = getattr(e, "env", None)
        if e is None:
            break
    raise RuntimeError("robosuite env 를 못 찾았다")


def relabel_preintv(modes, k):
    """각 개입 시작 직전의 **정책 프레임** k 개를 c=0 으로 바꾼다 (APO §3.1 의 relabel).

    개입 시작 = 이전이 개입이 아니고 지금이 개입인 지점. 정책 프레임(c=1)만 바꾼다 —
    앞선 개입 구간(c=2)까지 덮으면 "실패로 이어진 행동" 이 아닌 것이 섞인다.
    """
    m = np.asarray(modes, dtype=np.int64).copy()
    onsets = np.where((m == C_INTV) & (np.roll(m, 1) != C_INTV))[0]
    for o in onsets[onsets > 0]:
        w = m[max(0, o - k):o]
        w[w == C_POLICY] = C_PREINTV
    return m


@hydra.main(config_path="../configs", config_name="collect_intervention", version_base="1.3")
def collect(cfg: DictConfig):
    from manibot.teleoperators.keyboard_trigger import HELP, KeyboardTrigger

    setup_logging(save_dir=cfg.log_dir, debug=cfg.debug)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset_meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, dataset_meta)
    policy, pre, post = make_policy(cfg, dataset_meta, stats)
    policy = policy.to(cfg.device)
    ckpt = cfg.checkpoint_path
    load_model_weights(policy, ckpt, cfg.device)
    used_ema = cfg.get("use_ema", True) and load_ema_weights(policy, ckpt, cfg.device)
    policy.eval()
    logger.info(f"체크포인트 {ckpt} (EMA {'적용' if used_ema else '없음'})")

    env = make_eval_env(cfg.task)
    raw = _raw(env)
    Expert = resolve(cfg.task.sim.expert)
    low_dim = list(cfg.task.sim.state_from)
    cams = list(OmegaConf.to_container(cfg.task.sim.cameras, resolve=True))
    predict_fn = make_predict_fn(policy, cfg, cfg.device, preprocessor=pre, postprocessor=post)
    obs_h, act_h = cfg.policy.obs_horizon, cfg.policy.action_horizon

    trig = KeyboardTrigger()
    out = cfg.get("out_path") or cfg.task.hdf5_path.replace(".hdf5", "_intervention.hdf5")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    logger.info(f"저장: {out}\n키: {HELP}")

    # ⭐ 비동기 추론 — **지금 청크를 재생하는 동안 다음 청크를 계산한다.**
    # [실측 2026-09-08] 추론 223.7 ms · env.step x8 29 ms · 20Hz 재생 400 ms.
    # 동기로 두면 400 ms 재생 + 224 ms 정지가 번갈아 나와 화면이 끊긴다(사용자 지적).
    # 추론이 재생 시간보다 짧으므로 겹치면 완전히 가려진다.
    # ⚠ 대가: 다음 청크가 **한 청크 전(0.4초) 관측**으로 계산된다 = 배포 지연이 생긴다.
    #    실물 배포에는 원래 있는 지연이라 개입 데이터로는 오히려 현실적이지만,
    #    eval.py(동기)와는 조건이 달라진다 — 성공률을 그쪽과 직접 비교하지 말 것.
    pool = ThreadPoolExecutor(1) if cfg.async_infer else None

    from collections import deque
    kept = 0
    with h5py.File(out, "w") as f:
        grp = f.create_group("data")
        while kept < cfg.n_episodes and not trig.events["stop_recording"]:
            obs = env.reset()
            trig.reset_episode()
            hist = deque([obs] * obs_h, maxlen=obs_h)
            t_next = time.perf_counter()
            expert, future = None, None
            traj = {k: [] for k in low_dim + [f"{c}_image" for c in cams]}
            acts, modes = [], []
            success = False

            while len(acts) < cfg.max_steps:       # 청크가 아니라 **환경 스텝** 으로 센다
                if trig.events["exit_early"] or trig.events["stop_recording"]:
                    break
                if trig.intervening:
                    # 개입 켠 순간 **현재 상태에서** 다시 계획한다 (전문가가 재진입한다)
                    if expert is None:
                        # ⚠ 개입 교정에는 **다양성을 넣지 않는다**. 시연은 다양해야 정책이
                        # 넓게 배우지만, 교정은 확실해야 한다 — 실패한 교정이 c=2 로 들어가면
                        # "실패로 이어진 행동" 라벨이 오염된다.
                        expert = Expert(raw, jitter=False)
                    future = None                     # 개입 중에 만들어 둔 정책 청크는 버린다
                    chunk, mode = [expert.act()], C_INTV
                else:
                    expert = None                     # 정책으로 돌아오면 남은 계획을 버린다
                    if pool is None:
                        chunk = predict_fn(list(hist))
                    else:
                        if future is None:            # 첫 청크만 기다린다
                            future = pool.submit(predict_fn, list(hist))
                        chunk = future.result()
                        future = pool.submit(predict_fn, list(hist))   # 재생하며 다음 것을 계산
                    chunk, mode = chunk[:act_h], C_POLICY

                for a in chunk:
                    if cfg.view:
                        trig.feed(_show(raw, cams, cfg.view_res,
                                        expert.stage if expert else "-", kept, len(acts),
                                        trig.intervening))
                    if cfg.control_fps > 0:            # 사람이 보고 반응할 시간을 준다
                        t_next = max(t_next, time.perf_counter()) + 1.0 / cfg.control_fps
                    ro = raw._get_observations()
                    for k in low_dim:
                        traj[k].append(np.asarray(ro[k], dtype=np.float32).ravel())
                    for c in cams:
                        traj[f"{c}_image"].append(ro[f"{c}_image"][::-1])
                    acts.append(np.asarray(a, dtype=np.float32))
                    modes.append(mode)
                    obs, _, _, _ = env.step(np.asarray(a))
                    hist.append(obs)
                    if raw._check_success():
                        success = True
                    if cfg.control_fps > 0:
                        time.sleep(max(0.0, t_next - time.perf_counter()))
                    if success or len(acts) >= cfg.max_steps \
                            or trig.intervening != (mode == C_INTV):
                        break                          # 토글이 바뀌면 남은 청크를 버린다
                if success:
                    break

            m = relabel_preintv(modes, cfg.preintv_k)
            n_i, n_p = int((m == C_INTV).sum()), int((m == C_PREINTV).sum())
            if trig.events["rerecord_episode"]:
                logger.info(f"  다시 찍기 — 버림 ({len(acts)} 프레임)")
                continue
            d = grp.create_group(f"demo_{kept}")
            d.create_dataset("actions", data=np.asarray(acts, dtype=np.float32))
            d.create_dataset("action_mode", data=m)
            o = d.create_group("obs")
            for k, v in traj.items():
                arr = np.asarray(v)
                o.create_dataset(k, data=arr, dtype=arr.dtype)
            d.attrs["num_samples"] = len(acts)
            d.attrs["success"] = success
            kept += 1
            logger.info(f"  demo_{kept-1}: {len(acts):4d} 프레임 · 개입 {n_i} · pre-intv {n_p} · "
                        f"{'성공' if success else '실패'} · 누적 {kept}/{cfg.n_episodes}")
        grp.attrs["total"] = int(sum(grp[k].attrs["num_samples"] for k in grp))
        grp.attrs["env_args"] = OmegaConf.to_yaml(cfg.task.sim)
        grp.attrs["preintv_k"] = int(cfg.preintv_k)
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)
    trig.stop()
    if cfg.view:
        cv2.destroyAllWindows()
    logger.info(f"수집 완료: {kept} 에피소드 -> {out}")


def main():
    collect()


if __name__ == "__main__":
    main()
