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

import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from manibot.policies.factory import make_policy
from manibot.rollout import make_predict_fn
from manibot.rollout.merger import make_merger
from manibot.utils.checkpoints import load_ema_weights, load_model_weights
from manibot.utils.dataset_utils import create_dataset_stats
from manibot.utils.viewer import SimViewer
from manibot.utils.logger import setup_logging
from manibot.utils.seeding import seed_sim_env
from manibot.utils.task_utils import derive_task_meta, make_eval_env, resolve

logger = logging.getLogger(__name__)

from manibot.utils.intervention_labels import LABEL_INTV, LABEL_ROLLOUT
# ⚠ 저장 형식은 eval 과 **같은 모듈**을 쓴다 — 따로 구현하면 또 갈라진다
from manibot.utils import deploy_dataset as dd


def _demo_frame_count(cfg):
    """base policy 가 학습한 시연 데이터의 총 프레임 수 — 라운드 목표의 기준이다.

    `mani_sim/scripts/collect.py:_demo_frame_count` 와 같은 개념이고, 실물
    `rollout_intervention.py` 의 계산과도 같다. 학습에 실제로 쓰인 zarr 를 먼저 보고,
    없으면 원본 hdf5 로 떨어진다.
    """
    root = cfg.task.get("dataset_root")
    if root and os.path.exists(root):
        import zarr
        return int(zarr.open(root, mode="r")["data"]["action"].shape[0])
    import h5py
    with h5py.File(cfg.task.hdf5_path, "r") as f:
        return int(sum(f["data"][k].attrs["num_samples"] for k in f["data"]))


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

    # 로그는 데이터 폴더 **옆에** 둔다 (`<root>_collect.log`). 기본 output_dir 을 쓰면 데이터와
    # 무관한 세션 폴더가 생기고, 폴더 **안**에 두면 LeRobotDataset.create 가 "이미 있다"로 막힌다.
    _root = cfg.get("root")
    if _root:
        _r = Path(_root)
        setup_logging(save_dir=str(_r.parent), log_file=f"{_r.name}_collect.log", debug=cfg.debug)
    else:
        setup_logging(save_dir=cfg.log_dir, log_file="collect.log", debug=cfg.debug)
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
    seed_sim_env(raw, cfg.seed)
    Expert = resolve(cfg.task.sim.expert)
    low_dim = list(cfg.task.sim.state_from)
    # robosuite 카메라 이름 -> 우리 관측 이름. LeRobot 특징 이름이 이 매핑을 그대로 쓴다
    cams = OmegaConf.to_container(cfg.task.sim.cameras, resolve=True)
    predict_fn = make_predict_fn(policy, cfg, cfg.device, preprocessor=pre, postprocessor=post)
    obs_h, act_h = cfg.policy.obs_horizon, cfg.policy.action_horizon

    trig = KeyboardTrigger()
    # Wayland 에서는 전역 키 캡처가 안 돼 **이 창이 곧 조작면**이다 (utils/viewer 참조)
    viewer = SimViewer(cams, res=cfg.view_res, fps=cfg.control_fps,
                       title="manibot deployment") if cfg.view else None

    # 실물과 같은 형식 — LeRobotDataset. 학습은 `scripts/convert.py --native` 로 zarr 화한다.
    # ⚠ finalize 전의 parquet 은 footer 가 없어 못 읽는다. 렌더러·인코더가 죽는 일이 있으므로
    # ① 있으면 이어받고 ② save_every 마다 finalize 해서 손실을 그만큼으로 묶는다.
    features = dd.make_features(raw._get_observations(), cams, low_dim, raw.action_dim,
                                cfg.use_videos)
    root = cfg.get("root")

    def _open():
        return dd.open_or_resume(cfg.repo_id, root, cfg.task.fps, features,
                                 cfg.task.sim.robots, cfg.use_videos)

    ds = _open()

    # 라운드 크기는 에피소드가 아니라 **프레임**으로 정한다 — SIRIUS/APO 두 조건에 같은
    # 데이터량을 주려는 규약이다 (`mani_sim/scripts/collect.py:341`). 목표를 넘는 분량은
    # 에피소드를 자르지 않으니 자연히 생긴다. 합친 데이터셋의 demo 비율 = 1/(1+ratio).
    round_threshold = None
    if cfg.get("round_size_ratio"):
        demo_frames = _demo_frame_count(cfg)
        round_threshold = int(demo_frames * cfg.round_size_ratio)
        logger.info(f"[라운드 목표] {round_threshold:,} 프레임 "
                    f"({cfg.round_size_ratio}x {demo_frames:,} demo 프레임 -> "
                    f"합친 데이터셋 demo 비율 {1.0 / (1.0 + cfg.round_size_ratio):.0%})")
    ep_success = dd.read_success(ds.root)
    kept0 = len(ep_success)
    round_frames = int(getattr(ds, "num_frames", 0) or 0)     # 이어받은 분량부터 센다
    intv_frames = 0
    logger.info(f"저장: {ds.root} (repo_id={cfg.repo_id})\n키: {HELP}")

    # 비동기 추론 — 재생하는 동안 다음 청크를 계산한다. 실물에는 추론 지연이 원래 있으므로
    # 시뮬도 같은 구조로 둔다. 지연은 없애지 않고 **흡수**한다: 청크를 시각에 앵커해 merger
    # 에서 꺼내므로 늦게 온 청크는 앞부분이 버려질 뿐, 실행되는 행동은 언제나 지금의 예측이다.
    pool = ThreadPoolExecutor(1) if cfg.async_infer else None
    merger = make_merger(cfg.merger, te_coeff=cfg.te_coeff)

    from collections import deque
    kept = 0
    while not trig.events["stop_recording"]:
        if round_threshold is not None and round_frames >= round_threshold:
            logger.info(f"[라운드 종료] 목표 프레임 달성 "
                        f"({round_frames:,} >= {round_threshold:,})")
            break
        if kept0 + kept >= cfg.n_episodes:      # 안전 상한
            logger.info(f"[중단] 에피소드 상한 {cfg.n_episodes} 도달")
            break
        obs = env.reset()
        trig.reset_episode()
        hist = deque([obs] * obs_h, maxlen=obs_h)
        if viewer is not None:
            viewer.reset_clock()
        expert, pending, last_action, step = None, None, None, 0
        merger.clear()
        frames = []
        success = False

        # 실물(`manipulation_pipeline`)의 배포 루프와 같은 구조다.
        while step < cfg.max_steps:
            if trig.events["exit_early"] or trig.events["stop_recording"]:
                break

            # ① 끝난 추론을 앵커에 맞춰 제출한다. 늦게 끝났으면 청크 앞부분이 버려질 뿐이다
            if pending is not None and pending[1].done():
                t_obs, fut = pending
                merger.submit(t_obs, np.asarray(fut.result()))
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
                    merger.submit(t_obs, np.asarray(fut.result()))
                    pending = None
                    action = merger.get_action(step)
                if action is None:
                    action = last_action           # STALL — 마지막 행동을 유지한다
                    if action is None:
                        continue
                last_action = action
                mode = LABEL_ROLLOUT

            if viewer is not None:
                on = trig.intervening
                trig.feed(viewer.show(
                    raw, f"ep{kept0+kept} step{step}  "
                         f"{'INTERVENING (i=off)' if on else 'policy (i=on)'}  "
                         f"[{expert.stage if expert else '-'}]", highlight=on))
            frames.append(dd.make_frame(raw._get_observations(), cams, low_dim,
                                        action, mode, cfg.single_task))
            obs, _, _, _ = env.step(np.asarray(action))
            hist.append(obs)
            step += 1
            if raw._check_success():
                success = True
            if viewer is not None:
                viewer.pace()
            if success:
                break

        if trig.events["rerecord_episode"] or not frames:
            logger.info(f"  다시 찍기 — 버림 ({len(frames)} 프레임)")
            continue
        n_i = sum(1 for f in frames if int(f["action_mode"][0]) == LABEL_INTV)
        for fr in frames:
            ds.add_frame(fr)
        # parallel_encoding=True 는 카메라마다 ProcessPoolExecutor 를 띄우는데
        # (`lerobot/datasets/dataset_writer.py:344`), CUDA·MuJoCo·추론 스레드가 살아있는
        # 부모에서 fork 하는 구조라 워커가 죽는 일이 있다. False 면 풀 없이 차례로 인코딩한다.
        ds.save_episode(parallel_encoding=bool(cfg.parallel_encoding))
        ep_success.append(bool(success))
        kept += 1
        round_frames += len(frames)
        intv_frames += n_i
        prog = (f"{round_frames:,}/{round_threshold:,} 프레임"
                if round_threshold else f"{kept0+kept}/{cfg.n_episodes} 에피소드")
        logger.info(f"  ep{kept0+kept-1}: {len(frames):4d} 프레임 · 개입 {n_i} · "
                    f"{'성공' if success else '실패'} · 누적 {prog} "
                    f"(개입 {intv_frames/max(round_frames,1):.0%})")
        if cfg.save_every > 0 and kept % cfg.save_every == 0:
            ds.finalize()                  # 여기까지는 죽어도 남는다
            # 성공 목록도 여기서만 쓴다 — 매번 쓰면 못 읽는 에피소드까지 세어 어긋난다
            dd.write_success(ds.root, ep_success)
            ds = _open()

    # finalize() 를 빠뜨리면 잘린 parquet 이 남는다 — save_episode 가 백그라운드로 쓴다.
    ds.finalize()
    dd.write_success(ds.root, ep_success)
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)
    trig.stop()
    if viewer is not None:
        viewer.close()
    logger.info(f"수집 완료: {kept} 에피소드 · 성공 {sum(ep_success)}/{len(ep_success)} "
                f"-> {ds.root}")


def main():
    collect()


if __name__ == "__main__":
    main()
