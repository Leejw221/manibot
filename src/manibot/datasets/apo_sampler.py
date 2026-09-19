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


def split_pools(sign, has_intervention, allowed=None, is_demo=None):
    """(sign, has_intervention) -> (correct, intervention, incorrect) 인덱스 배열 셋.

    is_demo 를 주면 correct 를 **(시연, 정책 롤아웃)** 으로 쪼개 넷을 돌려준다.
    왜 쪼개나: correct 안의 시연 비율이 라운드마다 떨어진다(66% -> 55%). 배치 몫을
    correct 통째로 주면 **라운드가 갈수록 시연 노출이 줄고**, 실측에서 기존 시연 적합
    손실이 sf(잃은 성공)와 같이 움직였다 [2026-09-15: 기존 e_th +11% -> sf 17,
    +90% -> sf 39]. 시연에 고정 할당량을 주면 그 축이 라운드와 무관해진다.

    sign: (N,) `preference()` 가 낸 부호. has_intervention: (N,) 그 청크가 속한
    에피소드에 개입이 있었는지.

    **correct 판정은 프레임 단위다** [APO 공개 코드 대조 2026-09-20,
    dataset/balance_apo_dataset.py]. 원문은 `is_human` 으로 프레임마다 가른다:
        is_human == 2  -> correct        (정책이 낸 행동. 개입이 안 필요했던 구간)
        is_human == 1  -> intervention   (사람 교정)
        첫 개입 직전 K -> incorrect      (correct 에서 꺼내 옮긴다)
    그리고 `is_first_human` 이 is_human==2 에서 리셋되므로, **개입이 여러 번이면 각각
    자기 K 프레임 창만 incorrect 가 되고 나머지 정책 구간은 correct 로 남는다.**

    우리는 전에 `~has` (= 에피소드에 개입이 아예 없음) 로 판정했는데, 그러면 개입이
    한 번이라도 있는 에피소드의 **정책 행동이 통째로 버려진다**. 성공률 22.5% 짜리
    base 로는 거의 모든 배포 에피소드에 개입이 들어가므로, 실측에서 correct 풀 4,012 중
    배포 정책 행동이 **166 개(4.1%)** 뿐이었다 — 나머지 852 개가 버려졌다
    [측정 2026-09-20]. 그래서 배치 64 개 중 정책 행동이 1.3 개였고, "계속 이렇게 해라"
    를 가르치는 표본이 사실상 없었다.

    sign == 0 인 청크가 곧 원문의 is_human==2 다 — 개입 경계에서 멀어 recovery
    confidence 가 0 으로 감쇠한 정책 구간이다. 우리 데이터에서 `has & sign==0` 852 개가
    **전부 action_mode==0(정책) 프레임**임을 확인했다(개입 프레임은 섞이지 않았다).
    """
    sign = np.asarray(sign)
    has = np.asarray(has_intervention, dtype=bool)
    # EpisodeAwareSampler 가 drop_n_last_frames 로 에피소드 끝을 잘라내므로, 그 목록에
    # 없는 인덱스를 뽑으면 윈도가 에피소드 밖으로 나간다. 받은 목록 안에서만 나눈다.
    ok = np.ones(len(sign), dtype=bool)
    if allowed is not None:
        ok[:] = False; ok[np.asarray(allowed)] = True
    inter = np.where(ok & has & (sign > 0))[0]
    incorr = np.where(ok & has & (sign < 0))[0]
    # correct = 개입 없는 에피소드(시연) + **개입 에피소드의 개입-먼 정책 구간**
    is_corr = (~has) | (sign == 0)
    if is_demo is None:
        correct = np.where(ok & is_corr)[0]
        return correct, inter, incorr
    d = np.asarray(is_demo, dtype=bool)
    demo = np.where(ok & is_corr & d)[0]
    rollout = np.where(ok & is_corr & ~d)[0]
    return demo, rollout, inter, incorr


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
        # 비율 0 을 허용한다 — U 를 빼는 실험(SIRIUS 의 P*(preintv)=0)에 필요하다.
        assert any(n > 0 for n in self.per), f"모든 몫이 0 이다: {self.per}"
        for r, n in zip(ratios, self.per):
            assert r == 0 or n > 0, f"배치가 작아 비율을 못 맞춘다: {self.per}"
        for p, n in zip(self.pools, self.per):
            assert n == 0 or len(p) >= n, f"풀({len(p)})이 배치 몫({n})보다 작다"
        # 몫이 0 인 풀은 에폭 기준이 될 수 없다
        self.exhaust = exhaust % len(self.pools)
        if self.per[self.exhaust] == 0:
            self.exhaust = max(range(len(self.per)), key=lambda k: self.per[k])
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
                if n == 0:
                    continue
                if ptr[k] + n > len(o):            # 마른 풀은 다시 섞어 이어쓴다
                    order[k] = rng.permutation(self.pools[k]); ptr[k] = 0; o = order[k]
                batch.extend(o[ptr[k]:ptr[k] + n].tolist()); ptr[k] += n
            rng.shuffle(batch)                     # 배치 안 순서까지 섞는다
            yield batch
        self.epoch += 1
