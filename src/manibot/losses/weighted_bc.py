"""가중 BC — SIRIUS 의 구조에 APO 의 적응 가중을 넣은 손실.

    loss = mean_i [ lambda_i * l_theta,i ]          l_theta = ||eps - eps_theta||^2

    lambda_i = 1 - exp(-beta_D * w_i)    desirable      (APO 의 lambda_D 그대로)
             = lam_u_fixed               undesirable    (상수)
             = 0                         보류(sign==0)
    w_i = l~_i / sum_j l~_j ,  l~ = |eps - eps_theta|_1 / m[t]   (APO §3.2 + 우리 t 정규화)

**APO 와 무엇이 다른가 — `sign` 을 곱하지 않는다.** APO/KTO 는 undesirable 에 sign=-1 을 걸어
`l_theta` 를 **키우는** 방향으로 민다. 우리 연속 공간에서는 그 밀기가 발산한다:
`d(log pi)/d(eps_theta) = 2(eps - eps_theta)` 라 **오차에 비례해 기울기가 커지는 양의 되먹임**이고,
이산 토큰(APO 원문)의 `d(log pi)/d(logits) = e_y - pi`(노름 <= sqrt(2))에는 없는 성질이다.
실측 [2026-09-21]: U 의 오차가 ref 의 155배까지 가고 eps 공간 이동량이 intv 의 **350배**였다.
그 결과 demo 가 step 100 에 이미 7.4배 나빠졌다 — 기울기 크기가 `sqrt(l_theta)` 에 비례하므로
**제일 잘 맞는 샘플이 제일 약한 방어력**을 갖기 때문이다(demo 0.042 vs U 6.9, 164배).

여기서는 undesirable 을 **밀지 않고 가중만 낮춘다**(SIRIUS 의 `P*(preintv)=0` 자리).
전부 양수 가중이라 발산 항이 없다.

**lam_u_fixed 를 왜 0.002 로 두나**: `lam_D` 평균이 0.35 이고, U 가 ref 의 48배까지 나빠진
상태에서 U 의 기울기 크기를 **base 수준 demo 의 방어력과 같게** 만드는 값이
`0.35 * sqrt(0.0020) / sqrt(48) = 0.0023` 이다 [측정 2026-09-21]. 즉 "밀기가 지키기를
못 넘는다" 는 상한의 상수판이다. ⚠ 이 값은 **우리 데이터에서 잰 것**이라 라운드가 바뀌면
다시 재야 한다(고정 상수로 쓰는 한계).
"""

import numpy as np
import torch

from manibot.policies.diffusion_ops import prepare_cond, unet_out
from manibot.utils.intervention_labels import preference

from .weighting import apo_weights

__all__ = ["WeightedBCLoss"]


class WeightedBCLoss:
    def __init__(self, m_t, chunk_S, chunk_has=None, beta_d=64.0, n_t=8,
                 lam_u_fixed=0.002, expert_mag=1.0, use_mag=False,
                 chunk_is_demo=None):
        self.m_t = m_t
        self.chunk_S = chunk_S
        self.chunk_has = chunk_has
        self.beta_d = beta_d
        self.n_t = int(n_t)
        self.lam_u_fixed = float(lam_u_fixed)
        self.expert_mag = expert_mag
        self.use_mag = use_mag
        self.chunk_is_demo = chunk_is_demo

    def __call__(self, policy, batch):
        m = policy.diffusion
        T = m.noise_scheduler.config.num_train_timesteps
        cond, x0 = prepare_cond(policy, batch)
        B = x0.shape[0]

        idx = batch["dataset_index"].cpu().numpy()
        has = None if self.chunk_has is None else self.chunk_has[idx]
        sign_np, mag_np = preference(self.chunk_S[idx], has, expert_mag=self.expert_mag)
        if not self.use_mag:
            mag_np = np.ones_like(mag_np)
        sign = torch.as_tensor(sign_np, device=x0.device)
        mag = torch.as_tensor(mag_np, device=x0.device, dtype=x0.dtype)

        # t 는 균등 — reward 가 없으므로 importance sampling 을 걸 자리가 없다.
        se_the, l1_acc = 0.0, 0.0
        for _ in range(self.n_t):
            t = torch.randint(0, T, (B,), device=x0.device, dtype=torch.long)
            eps = torch.randn_like(x0)
            e_the, _, _ = unet_out(m, cond, x0, t, eps)
            se_the = se_the + ((eps - e_the) ** 2).mean(dim=(1, 2))
            l1_acc = l1_acc + (eps - e_the).abs().mean(dim=(1, 2)).detach() \
                / self.m_t.to(x0.device)[t]
        l_the = se_the / self.n_t                       # **기울기가 사는 유일한 항**
        l_tilde = l1_acc / self.n_t

        # lam_D 만 쓴다 — apo_weights 는 sign<0 에 lam_U 를 주므로 그 자리를 상수로 덮는다.
        lam, wdiag = apo_weights(l_tilde, None, self.m_t, sign, mag,
                                 self.beta_d, self.beta_d)
        lam = torch.where(sign < 0, torch.full_like(lam, self.lam_u_fixed), lam)
        lam = torch.where(sign == 0, torch.zeros_like(lam), lam)
        loss = (lam.detach() * l_the).mean()

        with torch.no_grad():
            out = {"lam_mean": lam.mean().item(), "l_the_mean": l_the.mean().item(),
                   "label_frac": (sign != 0).float().mean().item(),
                   **{k: v for k, v in wdiag.items() if k != "lam_mean"}}
            # 그룹은 **승격 전 S** 로 가른다 (preference 가 S=0 을 +1 로 올리므로)
            has_np = np.zeros(B, dtype=bool) if has is None else np.asarray(has)
            dem = (np.asarray(self.chunk_is_demo)[idx] if self.chunk_is_demo is not None
                   else np.ones(B, dtype=bool))
            S_np = np.asarray(self.chunk_S[idx], dtype=np.float64)
            _t = 1e-6
            groups = {"intv": has_np & (S_np > _t), "U": has_np & (S_np < -_t),
                      "policy": has_np & (np.abs(S_np) <= _t),
                      "demo": (~has_np) & dem, "rollout": (~has_np) & (~dem)}
            for g, msk in groups.items():
                if not msk.any():
                    continue
                sel = torch.as_tensor(msk, device=x0.device)
                out[f"l_the/{g}"] = l_the[sel].median().item()
                out[f"lam/{g}"] = lam[sel].median().item()
        return loss, out
