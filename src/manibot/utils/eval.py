"""Rollout evaluation in simulation.

One episode is: reset, then repeatedly ask the policy for an action chunk and
execute the first `action_horizon` actions of it. Success is whatever the
environment says (`env.is_success()`), which is the measure this project
judges training by — offline loss is not it.

The policy enters through `predict_fn` (see rollout.policy_server), so this
module knows no policy interface, the same boundary the inference threads use.
Videos are assembled from the observation images already in the rollout, so no
second render path exists to drift from the first.
"""

import logging
from collections import deque
from pathlib import Path

import numpy as np
import torch

from manibot.utils.utils import write_video

logger = logging.getLogger(__name__)


def _frames_to_video(frames, path: Path, fps: float) -> None:
    """frames: list of (C, H, W) float [0,1] — the same tensors the policy sees."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.stack([
        (np.asarray(f).transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8) for f in frames
    ])
    write_video(str(path), stacked, fps)


def _unwrap(env):
    """lerobot 래퍼를 벗겨 robosuite env 를 꺼낸다 — 렌더가 이걸 쓴다."""
    e = env
    for _ in range(4):
        if hasattr(e, "sim"):
            return e
        e = getattr(e, "env", None)
        if e is None:
            break
    raise RuntimeError("robosuite env 를 못 찾았다")


def _progress(env):
    """task 가 단계별 진행을 들고 있으면 그 사본. 없으면 None.

    성공률만으로는 실패를 못 읽는다 — "손잡이도 못 잡았다"와 "다 하고 복귀에서 실패했다"가
    같은 0 으로 보인다. 여러 단계를 순서대로 통과해야 하는 task 는 어디서 멈췄는지가
    다음에 무엇을 고칠지를 정한다. 그런 dict 가 없는 task 는 그냥 None 이다.
    """
    inner = getattr(env, "env", None)
    p = getattr(inner, "progress", None)
    if not isinstance(p, dict):
        return None
    # numpy bool 이 섞여 들어온다(판정을 numpy 비교로 하는 단계가 있다). 그대로 두면
    # 결과를 json 으로 쓸 때 "Object of type bool is not JSON serializable" 로 터진다.
    return {k: bool(v) for k, v in p.items()}


def rollout_episode(env, predict_fn, obs_horizon, action_horizon, max_steps, video_key=None,
                    merger_name="temporal_ensemble", te_coeff=0.01, anchor_offset=0,
                    async_infer=True, recorder=None, viewer=None, ep_label=""):
    """Run one episode. Returns (success, sum_reward, max_reward, steps, frames, progress).

    ⭐ **배포(`scripts/collect_intervention.py`)·실물(`manipulation_pipeline`)과 같은 구조다.**
    청크를 관측 시각에 앵커해 `merger.get_action(step)` 로 꺼낸다 — 늦게 온 청크는 앞부분이
    버려질 뿐, 실행되는 행동은 언제나 "지금 시각에 대한 예측"이다.  예전 판은 청크를
    그대로 8스텝 재생해서, 비동기 배포에서는 모든 행동이 한 청크씩 늦게 적용됐다
    [실측 2026-09-09, 같은 체크포인트: 청크 재생 6/20 vs merger 28/50].

    `recorder(action)` 을 주면 매 스텝 `env.step` **직전에** 부른다 — 배포 데이터를
    같이 모으는 경로다(호출자가 env 를 붙잡고 원본 관측을 꺼낸다).
    `viewer` 를 주면 같은 자리에서 창을 갱신한다 (`utils/viewer.SimViewer`).
    """
    from concurrent.futures import ThreadPoolExecutor

    from manibot.rollout.merger import make_merger

    obs = env.reset()
    history = deque([obs] * obs_horizon, maxlen=obs_horizon)
    frames = [obs[video_key]] if video_key else None

    if viewer is not None:
        viewer.reset_clock()
    merger = make_merger(merger_name, te_coeff=te_coeff)
    pool = ThreadPoolExecutor(1) if async_infer else None
    pending = None
    last_action = None
    rewards = []
    success = False
    steps = 0
    try:
        while steps < max_steps and not success:
            if pool is None:
                # 동기: 이번 스텝의 예측이 없을 때만 새로 뽑는다 (추론 횟수는 예전과 같다)
                if merger.get_action(steps) is None:
                    merger.submit(steps - anchor_offset,
                                  np.asarray(predict_fn(list(history))))
            else:
                if pending is not None and pending[1].done():
                    merger.submit(pending[0] - anchor_offset,
                                  np.asarray(pending[1].result()))
                    pending = None
                if pending is None:            # 제출하는 즉시 다음 요청 (continuous inference)
                    pending = (steps, pool.submit(predict_fn, list(history)))

            action = merger.get_action(steps)
            if action is None and pending is not None:
                # 콜드 스타트 — 첫 청크는 기다린다
                merger.submit(pending[0] - anchor_offset, np.asarray(pending[1].result()))
                pending = None
                action = merger.get_action(steps)
            if action is None:
                action = last_action           # STALL — 마지막 행동을 유지한다
                if action is None:
                    raise RuntimeError("첫 청크를 못 받았다 — predict_fn 을 확인할 것")
            last_action = action

            if viewer is not None:
                viewer.show(_unwrap(env), f"{ep_label}step{steps}")
                viewer.pace()
            if recorder is not None:
                recorder(action)
            obs, reward, _, _ = env.step(np.asarray(action))
            history.append(obs)
            if frames is not None:
                frames.append(obs[video_key])
            rewards.append(float(reward))
            steps += 1
            # robosuite 는 성공 상태에서도 done 을 안 세우는 태스크가 있어 is_success 를 본다.
            if env.is_success()["task"]:
                success = True
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    return (success, float(np.sum(rewards)), float(np.max(rewards) if rewards else 0.0),
            steps, frames, _progress(env))


def eval_policy(
    env,
    predict_fn,
    n_episodes: int,
    obs_horizon: int,
    action_horizon: int,
    max_steps: int,
    fps: float = 20.0,
    videos_dir: Path | None = None,
    max_episodes_rendered: int = 0,
    video_key: str | None = None,
    merger_name: str = "temporal_ensemble",
    te_coeff: float = 0.01,
    anchor_offset: int = 0,
    async_infer: bool = True,
    collector=None,
    viewer=None,
) -> dict:
    """Roll out `n_episodes` and aggregate.

    Returns
        aggregated   pc_success · avg_sum_reward · avg_max_reward
        per_episode  성공 여부·보상·스텝 수
        video_paths  max_episodes_rendered 개까지
    """
    per_episode, video_paths = [], []
    for ep in range(n_episodes):
        record = videos_dir is not None and video_key is not None and ep < max_episodes_rendered
        if collector is not None:
            collector.start(ep)
        success, sum_r, max_r, steps, frames, prog = rollout_episode(
            env, predict_fn, obs_horizon, action_horizon, max_steps,
            video_key=video_key if record else None,
            merger_name=merger_name, te_coeff=te_coeff, anchor_offset=anchor_offset,
            async_infer=async_infer,
            recorder=collector.record if collector is not None else None,
            viewer=viewer, ep_label=f"ep{ep} ",
        )
        if collector is not None:
            collector.finish(ep, success)
        per_episode.append({"episode": ep, "success": success, "sum_reward": sum_r,
                            "max_reward": max_r, "steps": steps,
                            **({"progress": prog} if prog else {})})
        if record and frames:
            path = Path(videos_dir) / f"eval_episode_{ep}.mp4"
            _frames_to_video(frames, path, fps)
            video_paths.append(str(path))
        done = ep + 1
        n_ok = sum(e["success"] for e in per_episode)
        flags = "" if not prog else " " + "".join(
            (k[0].upper() if v else "-") for k, v in prog.items())
        logger.info(f"eval ep {ep}: success={success} steps={steps}{flags} | "
                    f"누적 {n_ok}/{done} ({n_ok/done:.1%})")

    stages = {}
    if per_episode and "progress" in per_episode[0]:
        stages = {k: 100.0 * float(np.mean([e["progress"][k] for e in per_episode]))
                  for k in per_episode[0]["progress"]}
        logger.info("단계별 통과율: "
                    + " · ".join(f"{k} {v:.0f}%" for k, v in stages.items()))

    return {
        "aggregated": {
            "pc_success": 100.0 * float(np.mean([e["success"] for e in per_episode])),
            "avg_sum_reward": float(np.mean([e["sum_reward"] for e in per_episode])),
            "avg_max_reward": float(np.mean([e["max_reward"] for e in per_episode])),
        },
        "per_episode": per_episode,
        "stage_pass_rate": stages,
        "video_paths": video_paths,
    }
