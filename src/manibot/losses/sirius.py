"""SIRIUS 의 weighted BC — 프레임 라벨 · 시점별 가중.  공개 코드를 그대로 옮긴다.

⚠ **논문과 공개 코드가 다르다. 코드를 따른다.**
  논문 §IV-D : "We set P*(preintv)=0, essentially nullifying ..." · P*(demo)=P(demo)
  코드        : weight_preintv = 0.002 · w_demos = 1 (P*/P 가 아니라 상수)
  [코드 확인 2026-09-28, ~/Downloads/sirius/robomimic/utils/dataset.py:951 _sirius_reweight]

코드 그대로:
    라벨    프레임마다 {-1 demo, 0 rollout, 1 intv, -10 preintv}. preintv = 각 개입 시작
            직전 15 프레임, 앞선 개입을 만나면 멈춘다 (dataset.py:1246 _get_intv_labels)
            -> 우리 relabel_preintv(k=15) 와 프레임 단위로 같다 (r1·r2 불일치 0, 2026-09-28)
    비율    샘플의 **첫 시점 라벨**로 센다 (action_mode_selection=0, dataset.py:892)
    가중    w_demos=1 · w_intvs=0.5/r_intv · w_rollouts=(1-0.5-r_demo-0.002)/r_rollout
            · w_pre_intvs=0.002/r_preintv
    적용    가중 배열이 (N, seq_length) — **시점마다 자기 라벨의 가중** (dataset.py:1047,
            same_weight_for_seq=False). 손실은 action_loss (B,T) *= w ; mean() (bc.py:664)
공식 설정 [exps/sirius/sirius.json]: normalize=false · use_weighted_sampler=false
    -> 정규화 없음 · 자연 분포 샘플링 (finetune.balanced=null).

DP 로 옮기며 바뀐 것은 샘플별 손실뿐이다: -log pi(a_t|s) 대신 시점 t 의 노이즈 예측 MSE
(행동 차원 평균). 가중을 뺀 손실은 LeRobot DP 의 기본 손실(t 하나, 전 원소 MSE 평균)과 같다.
"""

import numpy as np
import torch

from manibot.policies.diffusion_ops import prepare_cond, unet_out

__all__ = ["SiriusLoss", "LABELS"]

LABELS = {"demo": -1, "robot": 0, "intv": 1, "preintv": -10}


class SiriusLoss:
    def __init__(self, frame_label, act_idx, p_star_intv=0.5, p_star_preintv=0.002):
        """frame_label: (F,) 프레임 라벨.  act_idx: (N, H) 샘플 i 의 액션 창 프레임 인덱스."""
        fl = np.asarray(frame_label, dtype=np.int64)
        act_idx = np.asarray(act_idx, dtype=np.int64)
        first = fl[act_idx[:, 0]]
        N = len(first)
        self.n = {c: int((first == v).sum()) for c, v in LABELS.items()}
        self.P = {c: n / N for c, n in self.n.items()}
        self.w_cls = {
            "demo": 1.0,
            "intv": p_star_intv / self.P["intv"] if self.n["intv"] else 0.0,
            "robot": ((1.0 - p_star_intv - self.P["demo"] - p_star_preintv) / self.P["robot"]
                      if self.n["robot"] else 0.0),
            "preintv": p_star_preintv / self.P["preintv"] if self.n["preintv"] else 0.0,
        }
        lut = {LABELS[c]: w for c, w in self.w_cls.items()}
        self.lab = fl[act_idx]                                   # (N, H)
        self.w = np.vectorize(lut.get)(self.lab).astype(np.float32)

    def __call__(self, policy, batch):
        m = policy.diffusion
        T = m.noise_scheduler.config.num_train_timesteps
        cond, x0 = prepare_cond(policy, batch)
        idx = batch["dataset_index"].cpu().numpy()
        w = torch.as_tensor(self.w[idx], device=x0.device, dtype=x0.dtype)   # (B, H)

        t = torch.randint(0, T, (x0.shape[0],), device=x0.device, dtype=torch.long)
        eps = torch.randn_like(x0)
        e_the, _, _ = unet_out(m, cond, x0, t, eps)
        l_t = ((eps - e_the) ** 2).mean(dim=2)                  # (B, H) 시점별
        loss = (w * l_t).mean()

        out = {"w_mean": float(w.mean()), "l_the_mean": float(l_t.mean().detach())}
        lab = torch.as_tensor(self.lab[idx], device=x0.device)
        for c, v in LABELS.items():
            sel = lab == v
            if sel.any():
                out[f"l_the/{c}"] = float(l_t[sel].median().detach())
                out[f"n/{c}"] = int(sel.sum())
        return loss, out
