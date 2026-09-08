"""개입 데이터의 프레임 라벨 — **실물(`manipulation_pipeline`)과 같은 규약**을 쓴다.

수집은 두 값만 쓴다 (`flare/scripts/rollout_intervention.py` 와 동일):

    0  rollout        정책이 실행한 프레임
    1  intervention   사람(여기선 스크립트)이 교정한 프레임

APO 는 여기에 셋째 값을 요구한다 [원문 직접, APO Alg.1 L15-16 · §3.1]:

    2  pre-intervention   각 개입 시작 **직전 K 프레임** — 실패로 이어진 행동

이건 **수집이 아니라 학습 직전에 만든다.** 수집 파일에 넣어 버리면 K 를 바꿀 때마다
다시 수집해야 하고, 실물 데이터(0/1 만 있다)와 형식이 갈린다. 같은 함수를 실물 데이터에도
그대로 쓸 수 있게 분리해 둔다. K 의 기본값 10 은 APO 부록 B ("the last 10 actions
before human intervention").
"""

import numpy as np

LABEL_ROLLOUT = 0
LABEL_INTV = 1
LABEL_PREINTV = 2


def relabel_preintv(action_mode, k: int = 10):
    """개입 시작 직전의 **정책 프레임** k 개를 LABEL_PREINTV 로 바꾼 배열을 돌려준다.

    개입 시작 = 이전이 개입이 아니고 지금이 개입인 지점. 정책 프레임만 바꾼다 —
    앞선 개입 구간까지 덮으면 "실패로 이어진 행동" 이 아닌 것이 섞인다.
    에피소드 경계를 넘지 않으려면 에피소드마다 따로 부른다.
    """
    m = np.asarray(action_mode, dtype=np.int64).copy()
    if m.size == 0:
        return m
    onsets = np.where((m == LABEL_INTV) & (np.roll(m, 1) != LABEL_INTV))[0]
    for o in onsets[onsets > 0]:
        w = m[max(0, o - k):o]
        w[w == LABEL_ROLLOUT] = LABEL_PREINTV
    return m
