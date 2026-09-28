"""SIRIUS 의 weighted BC — 클래스 단위 고정 가중.

⚠ **논문과 공개 코드가 다르다. 코드를 따른다.**
  논문 §IV-D : "We set P*(preintv)=0, essentially nullifying ..." · P*(demo)=P(demo)
  코드        : weight_preintv = 0.002 · w_demos = 1 (P*/P 가 아니라 상수)
  [코드 확인 2026-09-28, ~/Downloads/sirius/robomimic/utils/dataset.py:951 _sirius_reweight]

코드 그대로:
    weight_intv    = 0.5
    weight_preintv = 0.002
    w_demos     = 1
    w_intvs     = weight_intv / ratio_intv
    w_rollouts  = (1 - weight_intv - ratio_demos - weight_preintv) / ratio_rollouts
    w_pre_intvs = weight_preintv / ratio_pre_intv
    -> weight_dict = {-1: demos, 0: rollouts, 1: intvs, -10: pre_intvs}
    -> action_loss *= weights ; action_loss.mean()      (bc.py:286-289)
공식 설정 [exps/sirius/sirius.json]: normalize=false · use_weighted_sampler=false
    -> 정규화 없음 · **자연 분포 샘플링**. 우리 쪽은 finetune.balanced=null 과 함께 쓴다.
    preintv_relabeling: fixed · fixed_preintv_length=15  -> 우리 k_pre=15 와 같다.

우리 구현과의 차이 (기록용):
  · 가중이 **배치마다 달라지지 않는다** — wbc 의 적응 가중(1-exp(-64w))과 갈리는 지점.
  · preintv 판정은 우리 라벨 규칙(sign(S)<0, k_pre=15)을 쓴다. SIRIUS 는 길이 ell 구간 전체다.
    chunk_has_zero 로 제외하던 것은 **여기선 안 한다** — 원문에 그런 단계가 없다.
"""

import numpy as np
import torch

from manibot.policies.diffusion_ops import prepare_cond, unet_out

__all__ = ["SiriusLoss"]


class SiriusLoss:
    def __init__(self, chunk_S, has_intv, is_demo, p_star_intv=0.5,
                 p_star_preintv=0.002, n_t=1):
        S = np.asarray(chunk_S, dtype=np.float64)
        has = np.asarray(has_intv, dtype=bool)
        dem = np.asarray(is_demo, dtype=bool)
        sg = np.sign(S)
        cls = np.where(has & (sg > 0), "intv",
              np.where(has & (sg < 0), "preintv",
              np.where(dem, "demo", "robot")))
        n = {c: int((cls == c).sum()) for c in ("demo", "intv", "robot", "preintv")}
        N = sum(n.values())
        P = {c: v / N for c, v in n.items()}
        # 코드 그대로 — demo 는 상수 1, preintv 는 목표비 0.002 를 실제비로 나눈다
        Ps = {"intv": float(p_star_intv), "preintv": float(p_star_preintv),
              "demo": P["demo"]}
        Ps["robot"] = 1.0 - Ps["intv"] - P["demo"] - Ps["preintv"]
        self.w_cls = {
            "demo": 1.0,
            "intv": Ps["intv"] / P["intv"] if P["intv"] > 0 else 0.0,
            "robot": Ps["robot"] / P["robot"] if P["robot"] > 0 else 0.0,
            "preintv": Ps["preintv"] / P["preintv"] if P["preintv"] > 0 else 0.0,
        }
        self.w = np.array([self.w_cls[c] for c in cls], dtype=np.float32)
        self.cls = cls
        self.n_t = int(n_t)
        self.P, self.Ps, self.n = P, Ps, n

    def __call__(self, policy, batch):
        m = policy.diffusion
        T = m.noise_scheduler.config.num_train_timesteps
        cond, x0 = prepare_cond(policy, batch)
        B = x0.shape[0]
        idx = batch["dataset_index"].cpu().numpy()
        w = torch.as_tensor(self.w[idx], device=x0.device, dtype=x0.dtype)

        se = 0.0
        for _ in range(self.n_t):
            t = torch.randint(0, T, (B,), device=x0.device, dtype=torch.long)
            eps = torch.randn_like(x0)
            e_the, _, _ = unet_out(m, cond, x0, t, eps)
            se = se + ((eps - e_the) ** 2).mean(dim=(1, 2))
        l_the = se / self.n_t
        loss = (w * l_the).mean()

        out = {"w_mean": float(w.mean()), "l_the_mean": float(l_the.mean().detach())}
        for c in ("demo", "intv", "robot", "preintv"):
            sel = torch.as_tensor(self.cls[idx] == c, device=x0.device)
            if sel.any():
                out[f"l_the/{c}"] = float(l_the[sel].median().detach())
                out[f"n/{c}"] = int(sel.sum())
        return loss, out
