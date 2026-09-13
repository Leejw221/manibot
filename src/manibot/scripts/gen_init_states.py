"""평가용 고정 초기상태를 만들어 저장한다 (1회).

왜 필요한가: 시드를 심어도 초기 상태가 프로세스마다 다르다. 실측으로 같은 seed·같은
체크포인트가 51.0%/54.0% 를 냈고 100개 중 35개만 일치했다 [측정 2026-09-13].
원인은 하나가 아니다 — PYTHONHASHSEED 미고정으로 dict 순회 순서가 바뀌고, 로봇 초기
관절각에 gaussian 0.02 잡음이 있다. 상태를 통째로 저장해 되돌리면 전부 사라진다.

    python -m manibot.scripts.gen_init_states task=square_ph50 +n=200 \
        +out=data/eval_init_states_square_200.npz
"""

import hydra
import numpy as np

from manibot.utils.seeding import seed_sim_env
from manibot.utils.task_utils import is_sim_task, make_eval_env


@hydra.main(version_base="1.3", config_path="../configs", config_name="default_policy")
def main(cfg):
    assert is_sim_task(cfg.task), "시뮬 task 에서만 만든다"
    n = int(cfg.get("n", 200))
    out = cfg.get("out") or f"data/eval_init_states_{cfg.task.name}_{n}.npz"
    env = make_eval_env(cfg.task)
    raw = env
    for _ in range(5):
        if hasattr(raw, "robots"):
            break
        raw = getattr(raw, "env", raw)
    seed_sim_env(raw, cfg.seed)

    states = []
    for i in range(n):
        env.reset()
        states.append(np.asarray(env.get_state()["states"], dtype=np.float64))
    S = np.stack(states)
    np.savez(out, states=S, task=cfg.task.name, seed=cfg.seed)
    uniq = len({s.tobytes() for s in S})
    print(f"저장: {out}   {S.shape}  서로 다른 상태 {uniq}/{n}")


if __name__ == "__main__":
    main()
