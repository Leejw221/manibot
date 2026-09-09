"""배포 데이터 수집 — 정책을 굴리다 **사람이 트리거하면 스크립트가 교정**한다.

APO Algorithm 1 의 `Deployment(pi_theta, D_h)` 에 해당한다.  base policy 를 배포해
실패로 가는 상황을 사람이 보고 `i` 로 개입을 켠다.  교정 행동은 전문가(`sim.expert`)가
현재 상태에서 이어서 낸다 — VR 텔레오퍼가 없어도 되고 시드를 붙일 수 있다.

**저장 형식과 라벨은 실물(`manipulation_pipeline/flare/scripts/rollout_intervention.py`)과
같게 맞춘다** — 시뮬과 실물이 갈라지면 여기서 만든 것을 실물에 못 얹는다.
    LeRobotDataset (parquet + mp4) + 프레임마다 `action_mode`
    action_mode  0 = rollout(정책)  ·  1 = intervention(교정)
APO 의 c=0(개입 직전 K 프레임)은 **학습 직전에** `utils/intervention_labels.relabel_preintv`
로 만든다. 수집에 넣으면 K 를 바꿀 때마다 다시 수집해야 하고 실물 데이터와 형식이 갈린다.

키 (`teleoperators/keyboard_trigger.py`, LeRobot 규약 + i):
    i = 개입 토글 · →/n = 다음 에피소드 · ←/r = 다시 · s = 저장 · ESC/q = 중단

사용:
    python -m manibot.scripts.collect_intervention task=square_ph50 \\
        checkpoint_path=outputs/.../checkpoints/step_0000050000 n_episodes=50 \\
        repo_id=Leejungwook/square-deploy-round1

    # 학습용 zarr 로 (실물 데이터와 같은 경로).  ⚠ **--native 를 빼면 안 된다** —
    # 기본값이 240x320 으로 키우는데 robosuite 는 84x84 로 렌더하고 정책은 76x76 으로 크롭한다
    python src/manibot/scripts/convert.py --local-dir <root> --native -o data/<이름>_zarr
"""

import json
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from manibot.policies.factory import make_policy
from manibot.rollout import make_predict_fn
from manibot.rollout.merger import make_merger
from manibot.utils.checkpoints import load_ema_weights, load_model_weights
from manibot.utils.dataset_utils import create_dataset_stats
from manibot.utils.logger import setup_logging
from manibot.utils.task_utils import derive_task_meta, make_eval_env, resolve

logger = logging.getLogger(__name__)

from manibot.utils.intervention_labels import LABEL_INTV, LABEL_ROLLOUT


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
    # robosuite 카메라 이름 -> 우리 관측 이름. LeRobot 특징 이름이 이 매핑을 그대로 쓴다
    cams = OmegaConf.to_container(cfg.task.sim.cameras, resolve=True)
    predict_fn = make_predict_fn(policy, cfg, cfg.device, preprocessor=pre, postprocessor=post)
    obs_h, act_h = cfg.policy.obs_horizon, cfg.policy.action_horizon

    trig = KeyboardTrigger()

    # 실물과 같은 형식으로 쓴다 — LeRobotDataset(parquet + mp4).  학습은 `scripts/convert.py`
    # 로 zarr 화해서 읽는다(실물 데이터가 타는 경로와 같다).  형상은 첫 관측에서 읽는다.
    ro0 = raw._get_observations()
    img_shape = list(np.asarray(ro0[f"{next(iter(cams))}_image"]).shape)
    state_dim = int(sum(np.asarray(ro0[k]).size for k in low_dim))
    features = {
        **{name: {"dtype": "video", "shape": img_shape,
                  "names": ["height", "width", "channel"]} for name in cams.values()},
        "observation.state": {"dtype": "float32", "shape": [state_dim],
                              "names": [f"s{i}" for i in range(state_dim)]},
        "action": {"dtype": "float32", "shape": [int(raw.action_dim)],
                   "names": [f"a{i}" for i in range(int(raw.action_dim))]},
        # 실물과 같은 이름·형상.  0 = rollout · 1 = intervention
        "action_mode": {"dtype": "int64", "shape": (1,), "names": None},
    }
    # ⚠ **중간에 죽으면 finalize 전의 parquet 은 footer 가 없어 통째로 못 읽는다.**
    # [실측 2026-09-09] 42 에피소드에서 EGL 드라이버가 세그폴트로 죽어 전부 잃었다
    # (libnvidia-eglcore — hard_reset 이 리셋마다 렌더러를 새로 만든다).
    # ① 이미 있으면 이어받고 ② save_every 마다 finalize 해서 손실을 그만큼으로 묶는다.
    # 실물(`flare/scripts/rollout_intervention.py`)도 같은 자리에서 resume 을 쓴다.
    root = cfg.get("root")

    def _open():
        if root and Path(root).exists():
            logger.info(f"이어받기: {root}")
            return LeRobotDataset.resume(repo_id=cfg.repo_id, root=root)
        return LeRobotDataset.create(repo_id=cfg.repo_id, fps=int(cfg.task.fps), root=root,
                                     features=features, robot_type=str(cfg.task.sim.robots),
                                     use_videos=True)

    ds = _open()
    succ_path = Path(ds.root) / "episode_success.json"
    ep_success = json.loads(succ_path.read_text()) if succ_path.exists() else []
    kept0 = len(ep_success)
    logger.info(f"저장: {ds.root} (repo_id={cfg.repo_id})\n키: {HELP}")

    # ⭐ 비동기 추론 — **지금 청크를 재생하는 동안 다음 청크를 계산한다.**
    # 실물에는 추론 시간만큼의 지연이 원래 있으므로 시뮬도 같은 구조로 둔다.
    #
    # ⚠ 지연은 **없애지 않고 흡수한다**: 청크를 관측 시각에 앵커해 `merger.get_action(step)`
    # 로 꺼내므로, 늦게 온 청크는 앞부분이 버려질 뿐 실행되는 행동은 언제나 "지금 시각에
    # 대한 예측"이다. [실측 2026-09-09] 앵커 없이 chunk[0] 을 지금 실행하던 예전 판은
    # 20 에피소드 중 0 성공, 이 구조로 바꾸니 10 성공(동기 옛 구조는 6).
    pool = ThreadPoolExecutor(1) if cfg.async_infer else None
    merger = make_merger(cfg.merger, te_coeff=cfg.te_coeff)
    # 우리 정책(LeRobot 계열이 아닌 쪽)은 **관측 창의 첫 프레임**에 앵커한 전체 예측 구간을
    # 돌려준다 (`utils/dataset_utils.py` 의 action_indices=range(pred_horizon)).
    # LeRobot 계열은 generate_actions 가 이미 "지금"부터 잘라 준다 -> 0.
    anchor_offset = 0 if hasattr(policy, "predict_action_chunk") else obs_h - 1

    from collections import deque
    kept = 0
    while kept0 + kept < cfg.n_episodes and not trig.events["stop_recording"]:
        obs = env.reset()
        trig.reset_episode()
        hist = deque([obs] * obs_h, maxlen=obs_h)
        t_next = time.perf_counter()
        expert, pending, last_action, step = None, None, None, 0
        merger.clear()
        frames = []
        success = False

        # ⭐ **실물(`manipulation_pipeline`)의 배포 루프와 같은 구조다.**
        # 청크를 절대 스텝에 앵커해 `merger.get_action(step)` 으로 꺼낸다 — 그래야
        # "지금 시각에 대한 예측"이 지금 실행된다. 예전 판은 t-8 관측으로 만든 chunk[0]
        # 을 t 에 실행해서 **모든 행동이 8스텝 늦게 적용**됐고 성공률이 0/20 이었다.
        while step < cfg.max_steps:
            if trig.events["exit_early"] or trig.events["stop_recording"]:
                break

            # ① 끝난 추론을 앵커에 맞춰 제출한다. 늦게 끝났으면 청크 앞부분이 버려질 뿐이다
            if pending is not None and pending[1].done():
                t_obs, fut = pending
                merger.submit(t_obs - anchor_offset, np.asarray(fut.result()))
                pending = None
            # ② 요청이 비어 있으면 즉시 다음 것을 던진다 (continuous inference)
            if pending is None and pool is not None and not trig.intervening:
                pending = (step, pool.submit(predict_fn, list(hist)))

            if trig.intervening:
                # 개입 켠 순간 **현재 상태에서** 다시 계획한다 (전문가가 재진입한다)
                if expert is None:
                    # ⚠ 개입 교정에는 **다양성을 넣지 않는다**. 시연은 다양해야 정책이
                    # 넓게 배우지만, 교정은 확실해야 한다 — 실패한 교정이 c=2 로 들어가면
                    # "실패로 이어진 행동" 라벨이 오염된다.
                    expert = Expert(raw, jitter=False)
                merger.clear()                     # 정책이 만들어 둔 예측은 버린다
                last_action = None
                action, mode = expert.act(), LABEL_INTV
            else:
                if expert is not None:             # 정책으로 돌아왔다
                    expert = None
                    merger.clear()
                    last_action = None
                action = merger.get_action(step)
                if action is None and pending is not None:
                    # 콜드 스타트 — 첫 청크는 기다린다 (실물은 자세를 유지하며 기다린다)
                    t_obs, fut = pending
                    merger.submit(t_obs - anchor_offset, np.asarray(fut.result()))
                    pending = None
                    action = merger.get_action(step)
                if action is None:
                    action = last_action           # STALL — 마지막 행동을 유지한다
                    if action is None:
                        continue
                last_action = action
                mode = LABEL_ROLLOUT

            if cfg.view:
                trig.feed(_show(raw, cams, cfg.view_res,
                                expert.stage if expert else "-", kept, step,
                                trig.intervening))
            if cfg.control_fps > 0:                # 사람이 보고 반응할 시간을 준다
                t_next = max(t_next, time.perf_counter()) + 1.0 / cfg.control_fps
            ro = raw._get_observations()
            frames.append({
                **{name: ro[f"{c}_image"][::-1] for c, name in cams.items()},
                "observation.state": np.concatenate(
                    [np.asarray(ro[k], dtype=np.float32).ravel() for k in low_dim]),
                "action": np.asarray(action, dtype=np.float32),
                "action_mode": np.array([mode], dtype=np.int64),
                "task": cfg.single_task,
            })
            obs, _, _, _ = env.step(np.asarray(action))
            hist.append(obs)
            step += 1
            if raw._check_success():
                success = True
            if cfg.control_fps > 0:
                time.sleep(max(0.0, t_next - time.perf_counter()))
            if success:
                break

        if trig.events["rerecord_episode"] or not frames:
            logger.info(f"  다시 찍기 — 버림 ({len(frames)} 프레임)")
            continue
        n_i = sum(1 for f in frames if int(f["action_mode"][0]) == LABEL_INTV)
        for fr in frames:
            ds.add_frame(fr)
        ds.save_episode()
        ep_success.append(bool(success))
        succ_path.write_text(json.dumps(ep_success))
        kept += 1
        logger.info(f"  ep{kept0+kept-1}: {len(frames):4d} 프레임 · 개입 {n_i} · "
                    f"{'성공' if success else '실패'} · 누적 {kept0+kept}/{cfg.n_episodes}")
        if cfg.save_every > 0 and kept % cfg.save_every == 0:
            ds.finalize()                  # 여기까지는 죽어도 남는다
            ds = _open()

# ⚠ finalize() 를 빠뜨리면 **잘린 parquet 이 남는다** — save_episode 가 백그라운드로
# 쓰기 때문이다 (2026-09-08 실측: 5,651,594 -> 5,483,546 바이트로 잘렸다).
    ds.finalize()
    # 성공 여부는 시뮬에만 있는 정보라 LeRobot 스키마를 건드리지 않고 옆에 둔다.
    succ_path.write_text(json.dumps(ep_success))
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)
    trig.stop()
    if cfg.view:
        cv2.destroyAllWindows()
    logger.info(f"수집 완료: {kept} 에피소드 · 성공 {sum(ep_success)}/{len(ep_success)} "
                f"-> {ds.root}")


def main():
    collect()


if __name__ == "__main__":
    main()
