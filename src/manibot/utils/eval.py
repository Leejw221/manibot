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


def rollout_episode(env, predict_fn, obs_horizon, action_horizon, max_steps, video_key=None):
    """Run one episode. Returns (success, sum_reward, max_reward, steps, frames, progress)."""
    obs = env.reset()
    history = deque([obs] * obs_horizon, maxlen=obs_horizon)
    frames = [obs[video_key]] if video_key else None

    rewards = []
    success = False
    steps = 0
    while steps < max_steps and not success:
        chunk = predict_fn(list(history))          # (pred_horizon, action_dim)
        for action in chunk[:action_horizon]:
            obs, reward, _, _ = env.step(np.asarray(action))
            history.append(obs)
            if frames is not None:
                frames.append(obs[video_key])
            rewards.append(float(reward))
            steps += 1
            # robosuite 는 성공 상태에서도 done 을 안 세우는 태스크가 있어 is_success 를 본다.
            if env.is_success()["task"]:
                success = True
                break
            if steps >= max_steps:
                break

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
        success, sum_r, max_r, steps, frames, prog = rollout_episode(
            env, predict_fn, obs_horizon, action_horizon, max_steps,
            video_key=video_key if record else None,
        )
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
