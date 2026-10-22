"""공유 백본 DP + STG 헤드 학습 데이터 — 같은 프레임을 BC 역할과 STG 역할로 나눠 쓴다.

희민 self-improving-gym 의 JointStgDataset·JointBatchSampler(projects/square_assembly/.../joint_stg_dataset.py,
2026-10-21 확인)와 같은 구성이다:
- 인덱스 i < N 은 BC 역할(디퓨전 손실), i >= N 은 같은 프레임 i-N 의 STG 역할(CE 손실).
- 배치 = BC 풀 bc_batch 개(epoch 마다 섞어 겹침 없이, drop_last) + STG 풀 stg_batch 개(복원 추출).
- epoch 길이는 BC 풀 기준.
- STG 라벨 = 에피소드 안 남은 스텝(길이 - 1 - t), num_bins - 1 에서 자른다.
- STG 풀에서 val_fraction 만큼의 에피소드를 뺀다(STG 정확도를 떼어 둔 에피소드로 재기 위해). BC 풀과는 무관하다.

라운드 학습(R1~)에서 더하는 것 (2026-10-22) — 노션 계획 페이지 결정 6·8·16 과 10-22 가중 정의.
결정 경위·사용자 발언은 scripts/train_vip_stg.py docstring 의 "라운드 학습 기록":
- round_pools: 시연 전부 · 개입 에피소드는 첫 개입부터 · 개입 없는 에피소드 전부. BC·STG 풀이 같다.
- chunk_rates: 직전 라운드 STG 의 d 로 샘플 행동 창의 진척률(양 끝 / 할인). BC 배치 가중(policies/vip_stg.py)이 쓴다.
"""

import numpy as np
import torch

from manibot.utils.intervention_labels import LABEL_INTV, LABEL_ROLLOUT


def steps_to_go(episode_data_index):
    """(N,) 프레임별 남은 스텝 = 에피소드 끝 프레임 - 지금 프레임."""
    starts = np.asarray(episode_data_index["from"])
    ends = np.asarray(episode_data_index["to"])
    out = np.empty(int(ends[-1]), dtype=np.int64)
    for s, e in zip(starts, ends):
        out[s:e] = np.arange(e - s - 1, -1, -1)
    return out


def stg_val_episodes(n_episodes, val_fraction, seed):
    """STG 풀에서 뺄 에피소드 번호 — RandomState(seed) 순열의 앞쪽 round(n * val_fraction) 개.
    원본 episode_split_by_name 은 데모 이름을 사전순으로 정렬한 뒤 같은 순열을 쓰므로, 같은 seed 여도 뽑히는 번호는 다르다."""
    if not val_fraction:
        return set()
    n_val = max(1, int(round(n_episodes * val_fraction)))
    return set(np.random.RandomState(seed).permutation(n_episodes)[:n_val].tolist())


def stg_pool(episode_data_index, val_episodes):
    """STG 역할로 쓸 프레임 = val 에피소드를 뺀 모든 프레임(시연 데이터는 전부 성공 구간이다)."""
    idx = []
    for ep, (s, e) in enumerate(zip(episode_data_index["from"], episode_data_index["to"])):
        if ep not in val_episodes:
            idx.extend(range(int(s), int(e)))
    return np.asarray(idx, dtype=np.int64)


def round_pools(episode_data_index, action_mode, n_demo):
    """라운드 데이터의 (keep, floor) — 둘 다 (N,) bool.
    keep  = 학습에 쓰는 프레임: 시연(앞 n_demo 에피소드) 전부 · 개입 에피소드는 첫 개입 프레임부터 끝까지 ·
            개입 없는 에피소드 전부. 개입 에피소드의 개입 전 robot 행동은 BC·STG 모두 쓰지 않는다(결정 6).
    floor = 시연·개입 샘플 — BC 배치 몫의 하한 1/B 를 받는다. 개입 없는 robot 샘플은 하한이 없다."""
    mode = np.asarray(action_mode).ravel()
    bad = set(np.unique(mode).tolist()) - {LABEL_ROLLOUT, LABEL_INTV}
    if bad:
        raise ValueError(f"action_mode 에 robot·개입 밖의 라벨 {sorted(bad)} — 풀 규칙이 없다")
    keep, floor = np.zeros(len(mode), bool), np.zeros(len(mode), bool)
    for ep, (s, e) in enumerate(zip(episode_data_index["from"], episode_data_index["to"])):
        s, e = int(s), int(e)
        intv = mode[s:e] == LABEL_INTV
        if ep < n_demo:
            if intv.any():
                raise ValueError(f"시연 에피소드 {ep} 에 개입 라벨 — n_demo_episodes 확인")
            keep[s:e] = floor[s:e] = True
        elif intv.any():
            on = s + int(np.argmax(intv))
            # 개입 뒤 robot 이 다시 조작하면 robot·개입이 섞인 청크가 생긴다 — 규칙을 정한 적 없으니 멈춘다(결정 6)
            if not intv[on - s:].all():
                raise ValueError(f"에피소드 {ep}: 개입 뒤 robot 복귀 — 섞인 청크 규칙이 없다")
            keep[on:e] = floor[on:e] = True
        else:
            keep[s:e] = True
    return keep, floor


def chunk_rates(d, episode_data_index, delta, method, gamma):
    """(N,) 샘플 t 의 STG 진척률. d: (N,) 프레임별 남은 스텝 추정(score_stg) · delta: 행동 창 오프셋(t-1..t+14).
    창의 행동마다 r_k = d(o_j) - d(o_{j+1}) (j = t+delta[k], 그 행동 전후 관측). 창이 에피소드 밖이면 경계 프레임의 d 를
    쓴다 — 행동 패딩과 같은 규칙이라 끝 구간은 진척 0 칸이 섞인다(결정 4, 분모 고정).
      two  = sum_k r_k / H = (d_{t-1} - d_{t+15}) / H     양 끝 (결정 4)
      disc = sum_k gamma^k r_k / sum_k gamma^k              칸마다의 진척을 할인 평균 (10-22)
    STG 가 정확하고 정상 속도면 둘 다 1 이다(한 스텝에 남은 스텝 1 감소)."""
    delta = np.asarray(delta)
    if not np.array_equal(delta, np.arange(delta[0], delta[0] + len(delta))):
        raise ValueError(f"행동 창이 연속이 아니다: {delta.tolist()}")
    if method == "two":
        g = np.ones(len(delta))
    elif method == "disc":
        g = float(gamma) ** np.arange(len(delta))
    else:
        raise ValueError(f"rate={method!r} — two | disc")
    d = np.asarray(d, dtype=np.float64)
    offs = np.arange(delta[0], delta[-1] + 2)                  # 창 첫 행동 전 관측 .. 마지막 행동 뒤 관측 (H+1)
    out = np.empty(len(d))
    for s, e in zip(episode_data_index["from"], episode_data_index["to"]):
        s, e = int(s), int(e)
        w = d[np.clip(np.arange(s, e)[:, None] + offs, s, e - 1)]
        out[s:e] = ((w[:, :-1] - w[:, 1:]) * g).sum(1) / g.sum()
    return out


class JointStgDataset(torch.utils.data.Dataset):
    def __init__(self, base, stg_idx, labels, num_bins, bc_rate=None, bc_floor=None):
        """bc_rate·bc_floor: (N,) 라운드 학습의 BC 샘플 진척률과 하한 대상 — 주면 BC 역할 항목에 실어 보낸다."""
        self.base = base
        self.n = len(base)
        self._stg_set = set(np.asarray(stg_idx).tolist())
        self.labels = np.minimum(np.asarray(labels), num_bins - 1)
        if len(self.labels) != self.n:
            raise ValueError(f"STG 라벨 {len(self.labels)}개 != 프레임 {self.n}개")
        self.bc_rate, self.bc_floor = bc_rate, bc_floor
        if bc_rate is not None and not (len(bc_rate) == len(bc_floor) == self.n and np.isfinite(bc_rate).all()):
            raise ValueError("bc_rate·bc_floor 는 프레임 수만큼의 유한한 값이어야 한다")

    @property
    def num_frames(self):
        return self.base.num_frames

    @property
    def num_episodes(self):
        return self.base.num_episodes

    def __len__(self):
        return 2 * self.n

    def __getitem__(self, index):
        frame, stg_role = index % self.n, index >= self.n
        item = self.base[frame]
        item["bc_mask"] = torch.tensor(not stg_role)
        item["stg_mask"] = torch.tensor(bool(stg_role and frame in self._stg_set))
        item["stg_label"] = torch.tensor(int(self.labels[frame]))
        if self.bc_rate is not None:
            item["bc_rate"] = torch.tensor(float(self.bc_rate[frame]), dtype=torch.float32)
            item["bc_floor"] = torch.tensor(bool(self.bc_floor[frame]))
        return item


class JointBatchSampler:
    def __init__(self, bc_idx, stg_idx, n_frames, bc_batch, stg_batch, seed):
        self.bc_idx, self.stg_idx = np.asarray(bc_idx), np.asarray(stg_idx)
        self.n, self.bc_batch, self.stg_batch = n_frames, bc_batch, stg_batch
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.bc_idx) // self.bc_batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        bc = rng.permutation(self.bc_idx)
        for b in range(len(self)):
            stg = rng.choice(self.stg_idx, self.stg_batch) + self.n
            yield [int(i) for i in bc[b * self.bc_batch:(b + 1) * self.bc_batch]] + [int(i) for i in stg]
