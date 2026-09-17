"""Fixed-probe diagnostics for fine-tuning.

학습 loss 는 매 배치 샘플·t·noise 가 달라서 "새 데이터를 배우는 동안 기존 적합을 잃는가"
를 곡선으로 못 본다. 여기서는 기존/신규 에피소드에서 뽑은 **고정 배치**를 매번 같은
t·noise·crop 으로 재고, 출발 가중치로부터의 거리를 같이 남긴다.
"""

import random

import numpy as np
import torch
from torch.utils.data import default_collate

from manibot.utils.dataset_utils import EpisodeAwareSampler


def _save_rng(device):
    return (random.getstate(), np.random.get_state(),
            torch.get_rng_state(), torch.cuda.get_rng_state(device))


def _restore_rng(states, device):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    torch.cuda.set_rng_state(states[3], device)


class FitProbe:
    def __init__(self, cfg, dataset, network, preprocessor, device, amp_dtype, use_amp):
        p = cfg.probe
        self.seed = int(p.seed)
        self.preprocessor = preprocessor
        self.device = device
        self.amp_dtype = amp_dtype
        self.use_amp = use_amp

        # 학습 샘플러와 같은 유효 청크만 쓴다 — 에피소드 끝의 잘린 청크가 섞이면 학습과 다른 분포를 잰다.
        dn = cfg.policy.get(
            "drop_n_last_frames",
            cfg.policy.pred_horizon - cfg.policy.action_horizon - cfg.policy.obs_horizon + 1)
        allowed = np.array(list(EpisodeAwareSampler(
            dataset.episode_data_index, episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=dn, shuffle=False)))
        to = np.asarray(dataset.episode_data_index["to"])
        episode = np.asarray(dataset.episodes)[np.searchsorted(to, allowed, side="right")]

        if p.n_old_episodes is None:
            groups = {"all": allowed}
        else:
            groups = {"old": allowed[episode < p.n_old_episodes],
                      "new": allowed[episode >= p.n_old_episodes]}
        rng = np.random.default_rng(self.seed)
        states = _save_rng(device)  # 샘플 로딩이 전역 RNG 를 쓸 수 있다 — 학습 시작 상태를 지킨다
        try:
            self.batches = {
                name: [default_collate([dataset[int(i)] for i in
                                        rng.choice(idx, p.batch_size, replace=False)])
                       for _ in range(p.n_batches)]
                for name, idx in groups.items() if len(idx) >= p.batch_size}
        finally:
            _restore_rng(states, device)

        self.init_params = [q.detach().clone() for q in network.parameters()]
        self.init_norm = torch.sqrt(sum((q.float() ** 2).sum() for q in self.init_params))
        self.first = None

    @torch.no_grad()
    def measure(self, network, ema):
        # 학습 RNG 를 되돌린다 — probe 를 켜는 것만으로 학습 궤적이 바뀌면 on/off 비교가 무너진다.
        states = _save_rng(self.device)
        was_training = network.training
        network.train()  # 학습 손실과 같은 경로(랜덤 크롭)를 고정 시드로 잰다
        out = {}
        try:
            out.update(self._losses(network, "raw"))
            out["dist_rel_raw"] = self._dist(network)
            if ema is not None:
                ema.store(network.parameters())
                ema.copy_to(network.parameters())
                try:
                    out.update(self._losses(network, "ema"))
                    out["dist_rel_ema"] = self._dist(network)
                finally:
                    ema.restore(network.parameters())
        finally:
            _restore_rng(states, self.device)
            network.train(was_training)

        if self.first is None:
            self.first = dict(out)
        for k, v in list(out.items()):
            if k.startswith("L_") and self.first[k] > 0:
                out[f"{k}_rel"] = v / self.first[k] - 1.0
        return out

    def _losses(self, network, tag):
        out = {}
        for name, batches in self.batches.items():
            total = 0.0
            for j, cpu_batch in enumerate(batches):
                torch.manual_seed(self.seed * 1000 + j)
                batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                         for k, v in cpu_batch.items()}
                batch = self.preprocessor(batch)
                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    loss, _ = network.forward(batch)
                total += float(loss)
            out[f"L_{name}_{tag}"] = total / len(batches)
        return out

    def _dist(self, network):
        sq = sum(((q.float() - q0.float()) ** 2).sum()
                 for q, q0 in zip(network.parameters(), self.init_params))
        return float(torch.sqrt(sq) / self.init_norm)
