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


# ── recovery confidence ────────────────────────────────────────────────────
# 개입 경계를 기준으로 **연속적인** 선호 점수를 만든다 (2026-09-11 확정).
#
#   c(t) = -gamma^(tau-1-t)   t < tau   개입 직전 rollout — 실패로 이어진 행동
#        = +gamma^(t-tau)     t >= tau  개입 — 실패를 극복한 행동
#        = 0                            개입이 없는 에피소드
#
# 즉 마지막 rollout 프레임이 -1, 첫 개입 프레임이 +1 이고 양쪽으로 감쇠한다.
# 이산 3분류(rollout/intv/pre-intv)를 대체한다 — 경계에서 급변하지 않는다.
#
# ⚠ **샘플러 분류와 손실 판정은 반드시 이 한 기준(sign(S))을 공유해야 한다.**
#   둘을 따로 정의하면 샘플러가 목표한 배치 구성이 손실에서 재현되지 않는다
#   (실측 선례: pre-intv 구간 길이와 판정 윈도 길이가 어긋나 목표 25% 가 실제 18.3% 였다).


def recovery_confidence(action_mode, gamma: float = 0.95):
    """프레임별 c(t) 를 돌려준다. action_mode: (T,) — 0=rollout, 1=intervention.

    개입 경계가 여러 개면 **가장 가까운 경계**를 기준으로 삼는다. 경계 사이 중간
    지점에서 두 경계의 영향이 겹치는데, 합치면 부호가 상쇄돼 뜻이 흐려진다.
    """
    m = np.asarray(action_mode, dtype=np.int64).ravel()
    if m.size == 0:
        return np.zeros(0, dtype=np.float64)
    onsets = np.where((m == LABEL_INTV) & (np.roll(m, 1) != LABEL_INTV))[0]
    if m[0] == LABEL_INTV:
        onsets = np.unique(np.concatenate([[0], onsets]))
    if onsets.size == 0:
        return np.zeros(m.size, dtype=np.float64)

    d = np.arange(m.size)[:, None] - onsets[None, :]
    near = d[np.arange(m.size), np.abs(d).argmin(1)]
    # 지수를 **정수로 유지**한다. float 로 두면 경계를 걸친 청크에서 양·음이 마지막
    # 비트까지 상쇄되지 않아 S 가 정확히 0 이 안 된다 (실측: 39개 중 26개가 U 로 샜다).
    return np.where(near >= 0, gamma ** near, -(gamma ** (-near - 1)))


def chunk_scores(action_mode, horizon: int, gamma: float = 0.95, stride: int = 1):
    """슬라이딩 청크마다 S = sum_t c(t). 학습이 윈도를 뽑는 방식과 같게 stride=1 이 기본.

    반환: (starts, S) — starts 는 각 청크의 시작 프레임 인덱스.
    sign(S) 가 선호(desirable/undesirable), abs(S) 가 가중이다.
    """
    c = recovery_confidence(action_mode, gamma)
    n = c.size - horizon + 1
    if n <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    starts = np.arange(0, n, stride)
    csum = np.concatenate([[0.0], np.cumsum(c)])
    return starts, csum[starts + horizon] - csum[starts]


# 부호 판정은 **여기 한 곳에서만** 한다 — 샘플러와 손실이 같은 함수를 부른다.
PREF_TOL = 1e-9


def preference(S, tol: float = PREF_TOL):
    """청크 점수 S -> (sign, weight).

    sign: +1 desirable · -1 undesirable · 0 판정 보류(가중 0)
    weight: |S|.  tol 미만은 선호가 없는 것으로 본다 (개입 경계를 정확히 걸친 청크).
    """
    S = np.asarray(S, dtype=np.float64)
    sign = np.where(np.abs(S) < tol, 0, np.sign(S)).astype(np.int64)
    return sign, np.abs(S)
