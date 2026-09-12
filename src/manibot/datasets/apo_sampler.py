"""APO 의 balanced sampling — 풀 셋에서 **고정 개수씩** 꺼내 배치를 만든다.

APO 원문 [원문 직접 2026-09-12]:
    "we employ balanced sampling to ensure that each batch contains
     50% expert actions, 25% human intervention actions, and 25% failure actions"
공개 코드의 `correct_ratio=0.5 interaction_ratio=0.25` 와 일치한다.

**확률이 아니라 개수로 맞추는 이유**: 우리는 `w_i = l_i/sum(l_j)` 와 `z_0` 를 **배치 안에서**
계산한다. WeightedRandomSampler 처럼 기댓값으로만 맞추면 배치마다 구성이 흔들리고 그 통계도
같이 흔들린다. 매 배치가 정확히 32/16/16 이어야 한다.

**풀 분류는 `intervention_labels.preference()` 한 곳에서만 한다.** 샘플러와 손실이 기준을
따로 정의하면 목표한 배치 구성이 손실에서 재현되지 않는다 — 선례가 있다(목표 25%가 실제 18.3%).

1 에폭 = **가장 먼저 마르는 풀이 소진되는 시점**(APO 는 interaction 기준). 우리 데이터에선
incorrect 가 가장 작다: correct 10,064 / intervention 4,304 / incorrect 2,711 -> 169 배치.
"""

import numpy as np
from torch.utils.data import Sampler

__all__ = ["BalancedBatchSampler", "split_pools"]


def split_pools(sign, has_intervention, allowed=None):
    """(sign, has_intervention) -> (correct, intervention, incorrect) 인덱스 배열 셋.

    sign: (N,) `preference()` 가 낸 부호. has_intervention: (N,) 그 청크가 속한
    에피소드에 개입이 있었는지. 개입이 없는 데이터(시연·개입없음 롤아웃)는 S=0 이라
    부호로는 안 갈리므로 이 플래그로 correct 에 넣는다.
    """
    sign = np.asarray(sign)
    has = np.asarray(has_intervention, dtype=bool)
    # EpisodeAwareSampler 가 drop_n_last_frames 로 에피소드 끝을 잘라내므로, 그 목록에
    # 없는 인덱스를 뽑으면 윈도가 에피소드 밖으로 나간다. 받은 목록 안에서만 나눈다.
    ok = np.ones(len(sign), dtype=bool)
    if allowed is not None:
        ok[:] = False; ok[np.asarray(allowed)] = True
    correct = np.where(ok & ~has)[0]
    inter = np.where(ok & has & (sign > 0))[0]
    incorr = np.where(ok & has & (sign < 0))[0]
    return correct, inter, incorr          # sign==0 인 보류 청크는 어디에도 안 들어간다


class BalancedBatchSampler(Sampler):
    """batch_sampler 로 쓴다: DataLoader(dataset, batch_sampler=this).

    pools: 인덱스 배열들의 리스트. ratios 와 길이가 같아야 한다.
    exhaust: 에폭 종료를 정하는 풀의 위치(기본 -1 = 마지막 = incorrect).
    """

    def __init__(self, pools, batch_size: int, ratios=(0.5, 0.25, 0.25),
                 exhaust: int = -1, seed: int = 0, drop_last: bool = True):
        assert len(pools) == len(ratios), "풀과 비율의 개수가 다르다"
        self.pools = [np.asarray(p) for p in pools]
        self.per = [int(round(batch_size * r)) for r in ratios]
        # 반올림 오차를 가장 큰 풀에 흡수시킨다
        self.per[int(np.argmax(ratios))] += batch_size - sum(self.per)
        assert all(n > 0 for n in self.per), f"배치가 작아 비율을 못 맞춘다: {self.per}"
        for p, n in zip(self.pools, self.per):
            assert len(p) >= n, f"풀({len(p)})이 배치 몫({n})보다 작다"
        self.exhaust = exhaust % len(self.pools)
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __len__(self):
        return len(self.pools[self.exhaust]) // self.per[self.exhaust]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = [rng.permutation(p) for p in self.pools]
        ptr = [0] * len(self.pools)
        for _ in range(len(self)):
            batch = []
            for k, (o, n) in enumerate(zip(order, self.per)):
                if ptr[k] + n > len(o):            # 마른 풀은 다시 섞어 이어쓴다
                    order[k] = rng.permutation(self.pools[k]); ptr[k] = 0; o = order[k]
                batch.extend(o[ptr[k]:ptr[k] + n].tolist()); ptr[k] += n
            rng.shuffle(batch)                     # 배치 안 순서까지 섞는다
            yield batch
        self.epoch += 1
