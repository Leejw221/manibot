"""rollout 평가 계약 검증.

학습 루프(scripts/train.py 의 validate_online)와 독립 평가(scripts/eval.py)가
같은 eval_policy 를 쓰므로, 여기서 계약이 지켜지면 두 경로의 숫자가 같은 코드에서 나온다.

정책 없이 predict_fn 만으로 돈다 — 그 경계가 지켜지는지도 같이 확인된다.
MUJOCO_GL=egl 로 실행한다. robosuite 가 없으면 건너뛴다.
"""
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from manibot.utils.task_utils import make_eval_env

CFG_DIR = str(Path(__file__).resolve().parents[1] / "src" / "manibot" / "configs")

try:
    import robosuite  # noqa: F401
except ImportError:
    print("  - robosuite 없음 — [sim] 미설치라 건너뛴다")
    raise SystemExit(0)

from manibot.utils.eval import eval_policy

with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
    cfg = compose(config_name="default_policy",
                  overrides=["task=robomimic_square", "policy=diffusion"])

PRED_HORIZON, ACTION_DIM = 8, 7
calls = {"n": 0}


def predict_fn(obs_history):
    """무작위 행동. 정책이 아니라 predict_fn 경계만으로 도는지 본다."""
    calls["n"] += 1
    assert len(obs_history) == cfg.policy.obs_horizon
    assert cfg.task.state_key in obs_history[-1], "관측 이름 규약이 깨졌다"
    return np.random.uniform(-1, 1, (PRED_HORIZON, ACTION_DIM)).astype(np.float32)


env = make_eval_env(cfg.task)
try:
    videos = Path("/tmp/manibot_test_eval_videos")
    info = eval_policy(
        env, predict_fn, n_episodes=2,
        obs_horizon=cfg.policy.obs_horizon, action_horizon=4, max_steps=12,
        fps=cfg.task.fps, videos_dir=videos, max_episodes_rendered=1,
        video_key=cfg.task.image_keys[0],
    )
finally:
    env.close()

agg = info["aggregated"]
assert set(agg) == {"pc_success", "avg_sum_reward", "avg_max_reward"}, agg
assert len(info["per_episode"]) == 2, info["per_episode"]
assert all(e["steps"] <= 12 for e in info["per_episode"]), info["per_episode"]
assert calls["n"] >= 2, "predict_fn 이 호출되지 않았다"
print(f"  ✓ 2 에피소드 rollout · steps {[e['steps'] for e in info['per_episode']]}")
print(f"  ✓ 집계 {agg}")
assert len(info["video_paths"]) == 1 and Path(info["video_paths"][0]).exists(), info["video_paths"]
print(f"  ✓ 비디오 {Path(info['video_paths'][0]).name} ({Path(info['video_paths'][0]).stat().st_size} bytes)")
print("  ✓ 정책 없이 predict_fn 만으로 평가가 돈다")
