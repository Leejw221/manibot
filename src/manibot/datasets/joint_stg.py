"""공유 백본 DP + STG 헤드 학습 데이터 — 같은 프레임을 BC 역할과 STG 역할로 나눠 쓴다.

희민 self-improving-gym 의 JointStgDataset·JointBatchSampler(projects/square_assembly/.../joint_stg_dataset.py,
2026-10-21 확인)와 같은 구성이다:
- 인덱스 i < N 은 BC 역할(디퓨전 손실), i >= N 은 같은 프레임 i-N 의 STG 역할(CE 손실).
- 배치 = BC 풀 bc_batch 개(epoch 마다 섞어 겹침 없이, drop_last) + STG 풀 stg_batch 개(복원 추출).
- epoch 길이는 BC 풀 기준.
- STG 라벨 = 에피소드 안 남은 스텝(길이 - 1 - t), num_bins - 1 에서 자른다.
- STG 풀에서 val_fraction 만큼의 에피소드를 뺀다(STG 정확도를 떼어 둔 에피소드로 재기 위해). BC 풀과는 무관하다.
"""

import numpy as np
import torch


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


class JointStgDataset(torch.utils.data.Dataset):
    def __init__(self, base, stg_idx, labels, num_bins):
        self.base = base
        self.n = len(base)
        self._stg_set = set(np.asarray(stg_idx).tolist())
        self.labels = np.minimum(np.asarray(labels), num_bins - 1)
        if len(self.labels) != self.n:
            raise ValueError(f"STG 라벨 {len(self.labels)}개 != 프레임 {self.n}개")

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
