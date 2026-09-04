"""task 설정 스키마와 관측 이름 통일 검증.

실물이든 시뮬이든 학습·평가가 보는 관측 이름은 한 벌(lerobot 형식)이어야 한다.
시뮬 env 가 설정에 적힌 image_keys + state_key 를 정확히 그대로 내보내는지 확인한다 —
여기가 어긋나면 학습과 평가가 다른 키를 보게 되고 에러 없이 성능으로만 드러난다.

MUJOCO_GL=egl 로 실행한다. robosuite 가 없으면 시뮬 부분은 건너뛴다.
"""
import numpy as np
from pathlib import Path
from hydra import compose, initialize_config_dir

from manibot.utils.task_utils import is_sim_task, make_eval_env

CFG_DIR = str((Path(__file__).resolve().parents[1] / "src" / "manibot" / "configs"))
SIM_TASKS = ["robomimic_square", "robomimic_lift", "robomimic_transport", "door_cabinet"]
REAL_TASKS = ["piper_cube_stack", "piper_bimanual_cube_stack"]


def load(task):
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        return compose(config_name="default_policy", overrides=[f"task={task}", "policy=diffusion"])


for t in SIM_TASKS + REAL_TASKS:
    c = load(t).task
    assert c.state_key == "observation.state", f"{t}: state_key 가 규약과 다르다"
    assert c.action_key == "action", f"{t}: action_key 가 규약과 다르다"
    for k in c.image_keys:
        assert k.startswith("observation.images."), f"{t}: {k} 가 규약과 다르다"
    assert is_sim_task(c) == (t in SIM_TASKS), f"{t}: sim 블록 유무가 기대와 다르다"
    print(f"  ✓ {t:26s} sim={'있음' if is_sim_task(c) else 'null'} "
          f"image={len(c.image_keys)} state_dim={c.state_dim} action_dim={c.action_dim}")

for t in REAL_TASKS:
    try:
        make_eval_env(load(t).task)
        raise AssertionError(f"{t}: 실물 task 인데 시뮬 env 가 만들어졌다")
    except ValueError:
        pass
print("  ✓ 실물 task 는 시뮬 env 요청을 거절한다")

try:
    import robosuite  # noqa: F401
except ImportError:
    print("  - robosuite 없음 — 시뮬 env 검증은 건너뛴다")
    raise SystemExit(0)

for t in SIM_TASKS:
    cfg = load(t)
    env = make_eval_env(cfg.task)
    obs = env.reset()
    expected = set(cfg.task.image_keys) | {cfg.task.state_key}
    assert set(obs) == expected, f"{t}: 관측 키가 설정과 다르다 — {set(obs) ^ expected}"
    assert np.asarray(obs[cfg.task.state_key]).shape == (cfg.task.state_dim,), \
        f"{t}: state_dim 불일치"
    obs2, _, _, _ = env.step(np.zeros(env.action_dimension))
    assert set(obs2) == expected, f"{t}: step 과 reset 의 키가 다르다"
    print(f"  ✓ {t:26s} env 관측이 설정의 이름과 정확히 일치")
