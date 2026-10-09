"""샘플마다 고정 가중을 주는 weighted BC — 가중은 클래스가 아니라 직전 라운드 정책의 재현 오차에서 온다.

가중 w_i 는 scripts/build_sample_weights.py 가 미리 만든다 (score_samples 의 L1 -> 평균 대비 비 ->
clip -> 평균 1). 손실은 SiriusLoss(normalize=True) 와 같다: 시점별 noise MSE (B, H) 에 샘플 가중을
16칸 같은 값으로 곱하고 sum(w*l)/sum(w). 클래스 라벨(self.lab)은 PI 샘플 제외(train.py)와 로그에만 쓴다.

판정 칸 [사용자 결정 2026-10-09]: PI 판정과 L1 채점은 샘플 t 의 행동 창(프레임 t-1..t+14) 중
프레임 t 부터이고 패딩이 아닌 칸만 쓴다 (judge_slots). 손실은 16칸 전부 — SiriusLoss 와 비교 조건을 맞춘다.
"""

import numpy as np
import torch

from manibot.losses.sirius import LABELS
from manibot.policies.diffusion_ops import prepare_cond, unet_out

__all__ = ["SampleWeightLoss", "judge_slots"]


def judge_slots(dataset, ep):
    """((N, H) bool 판정 칸, 규약 이름).  판정 칸 = 행동 창에서 프레임 t 부터(delta >= 0)이고 패딩이 아닌 칸.
    t-1 칸은 추론 때 생성만 되고 실행되지 않고(실행은 t 부터), 패딩 칸은 에피소드 밖을 경계 프레임으로 채운
    복사라서 뺀다. 규약 이름은 점수·가중 파일에 남겨 build·train 이 같은 칸으로 셌는지 대조한다."""
    from lerobot.utils.constants import ACTION
    delta = np.asarray(dataset.delta_indices[ACTION])
    pad = np.stack([dataset._get_query_indices(i, int(ep[i]))[1][f"{ACTION}_is_pad"].numpy()
                    for i in range(len(dataset))])
    return (delta >= 0) & ~pad, f"t..t+{int(delta.max())},no_pad"


class SampleWeightLoss:
    def __init__(self, weights_path, frame_label, act_idx, episode_index, n_demo, k_drop, judge, judge_name):
        """weights_path: build_sample_weights 의 npz (w, drop).  frame_label · episode_index: (F,).
        act_idx: (N, H).  n_demo · k_drop: 학습 설정 — 가중 파일을 만든 값과 같아야 한다.
        judge · judge_name: judge_slots 의 반환값 — PI 판정 칸 (train.py 가 drop 을 다시 셀 때 쓴다)."""
        d = np.load(weights_path)
        self.lab = np.asarray(frame_label, dtype=np.int64)[np.asarray(act_idx, dtype=np.int64)]
        N = len(self.lab)
        if len(d["w"]) != N:
            raise ValueError(f"가중 파일 길이 {len(d['w'])} != 데이터셋 샘플 {N}: {weights_path}")
        # 길이·drop 이 우연히 같아도 다른 데이터셋·설정으로 만든 파일이면 거른다
        if not np.array_equal(d["episode_index"], np.asarray(episode_index)[:N]):
            raise ValueError(f"가중 파일의 에피소드 구성이 데이터셋과 다르다: {weights_path}")
        made = (int(d["n_demo_episodes"]), int(d["drop_preintv_min"]))
        if made != (int(n_demo), int(k_drop)):
            raise ValueError(f"가중 파일의 (n_demo_episodes, drop_preintv_min)={made} != 학습 설정 "
                             f"{(int(n_demo), int(k_drop))}: {weights_path}")
        # 규약이 없는 파일 = 2026-10-09 이전의 16칸 판정 — drop 이 우연히 같아도 L1 의 유효 칸이 다르다
        made_judge = str(d["judge"]) if "judge" in d.files else "(없음: 16칸 판정)"
        if made_judge != judge_name:
            raise ValueError(f"가중 파일의 판정 규약 {made_judge} != 학습 {judge_name}: {weights_path}")
        self.judge = np.asarray(judge, dtype=bool)
        self.w = d["w"].astype(np.float32)
        self.drop = d["drop"].astype(bool)
        # 손실이 sum(w*l)/sum(w) 라 NaN·inf·음수 하나가 batch 손실 전체를 망가뜨린다
        if not (np.isfinite(self.w).all() and (self.w >= 0).all()):
            raise ValueError(f"가중 파일의 w 에 NaN·inf·음수가 있다: {weights_path}")

    def __call__(self, policy, batch):
        m = policy.diffusion
        T = m.noise_scheduler.config.num_train_timesteps
        cond, x0 = prepare_cond(policy, batch)
        idx = batch["dataset_index"].cpu().numpy()
        w = torch.as_tensor(self.w[idx], device=x0.device, dtype=x0.dtype)[:, None].expand(-1, x0.shape[1])

        t = torch.randint(0, T, (x0.shape[0],), device=x0.device, dtype=torch.long)
        eps = torch.randn_like(x0)
        e_the, _, _ = unet_out(m, cond, x0, t, eps)
        l_t = ((eps - e_the) ** 2).mean(dim=2)                  # (B, H) 시점별
        loss = (w * l_t).sum() / w.sum().clamp_min(1e-8)

        out = {"w_mean": float(w.mean()), "l_the_mean": float(l_t.mean().detach())}
        lab = torch.as_tensor(self.lab[idx], device=x0.device)
        for c, v in LABELS.items():
            sel = lab == v
            if sel.any():
                out[f"l_the/{c}"] = float(l_t[sel].median().detach())
                out[f"n/{c}"] = int(sel.sum())
        return loss, out
