"""시뮬 환경의 난수 고정.

⚠ `np.random.seed()` 만으로는 robosuite 의 물체 배치가 **통제되지 않는다** — 배치 샘플러가
자기 `np.random.Generator` 를 들고 있어서, 같은 시드로 두 번 돌려도 다른 에피소드가 나온다.
그래서 A/B 를 짝지어 비교할 수 없고 성공률 낱값도 흔들린다. 합성 샘플러의 하위까지 내려가며
rng 를 다시 심는다.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def seed_sim_env(env, seed: int) -> int:
    """전역 np.random 과 배치 샘플러의 rng 를 함께 고정한다. 심은 샘플러 수를 돌려준다."""
    np.random.seed(seed)
    n, stack = 0, [getattr(env, "placement_initializer", None)]
    while stack:
        s = stack.pop()
        if s is None:
            continue
        if hasattr(s, "rng"):
            s.rng = np.random.default_rng(seed * 1000 + n)
            n += 1
        stack.extend(getattr(s, "samplers", {}).values())
    if n:
        logger.info(f"배치 샘플러 {n}개 시드 고정 (seed={seed})")
    return n
