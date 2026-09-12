"""APO (Action Preference Optimization) 를 Diffusion Policy 로 옮긴 손실.

원문 [원문 직접 2026-09-12]
    r_theta(o,a) = log[pi_theta(a|o) / pi_ref(a|o)]
    v = lambda_D * sigma(r - z0)      desirable
        lambda_U * sigma(z0 - r)      undesirable
    z0 = KL(pi_theta || pi_ref)

**diffusion 으로 옮기며 달라지는 것** — 왜 이 형태인지는 노션 「reward 설계」 참고.
  · log pi(a|o) 를 못 잰다(경로 적분). 고정분산 가우시안에서 두 정책의 로그비는
    **가중된 노이즈 예측 오차의 차**가 된다 — Diffusion-DPO 가 도착한 것과 같은 형태.
  · 그 앞에 **T** 를 곱해 "한 스텝"을 "청크 전체 합"의 불편추정으로 만든다.
    Diffusion-KTO 공개 구현은 이 T 도 t 계수도 안 곱한다 [코드 확인 2026-09-12].
  · 프레임 축은 **평균**(sum 아님) — Diffusion-KTO 공식 구현과 같다.

**선호 판정은 preference() 한 곳에서만** 한다. 샘플러와 손실이 기준을 따로 정의하면
샘플러가 목표한 배치 구성이 손실에서 재현되지 않는다.

⚠ **크롭**: 우리 정책은 안에서 랜덤 크롭을 한다(crop_shape=(76,76)). 두 정책이 다른 크롭을
보면 r 이 정책 차이가 아니라 "어디를 잘랐나" 를 섞어 잰다 — 실측으로 그 잡음이 정책 신호의
**0.81배**였다. RNG 상태를 되감아 같은 크롭을 쓰게 하고, 시작 시 카나리아로 검사한다.
"""

import torch

from manibot.policies.diffusion_ops import prepare_cond, unet_out
from manibot.utils.intervention_labels import preference

from .weighting import apo_weights

__all__ = ["APOLoss"]


class APOLoss:
    def __init__(self, ref, m_t, chunk_S, beta=30.0, beta_d=8.0, beta_u=8.0,
                 z0_clamp=(-5.0, 5.0), bc_weight=0.0, ref_mode="live"):
        self.ref = ref
        self.m_t = m_t
        # 샘플러가 푸는 것과 **같은 배열**. 기준이 둘이면 배치 구성이 손실에서 재현되지 않는다.
        self.chunk_S = chunk_S
        self.beta, self.beta_d, self.beta_u = beta, beta_d, beta_u
        self.z0_lo, self.z0_hi = z0_clamp
        self.bc_weight = bc_weight
        self.ref_mode = ref_mode
        self._canary_done = False
        # 크롭을 pi_theta 와 같은 **랜덤 모드**로. 인코더에서 self.training 으로 갈리는 곳은
        # 크롭 한 줄뿐이다(BatchNorm 은 GroupNorm 대체, Dropout 없음) — 확인 2026-09-12.
        self.ref.train(True)
        self.ref.requires_grad_(False)

    # ── 크롭 재현 카나리아 ────────────────────────────────────────────────
    def _canary(self, batch):
        """RNG 를 되감으면 cond 가 bit-identical 한가. 아니면 크롭 외에 RNG 를 쓰는 것이
        생긴 것이고, 그러면 r 에 조용히 잡음이 섞인다. **학습 전에 멈춘다.**"""
        st = torch.get_rng_state()
        with torch.no_grad():
            c1, _ = prepare_cond(self.ref, batch)
            torch.set_rng_state(st)
            c2, _ = prepare_cond(self.ref, batch)
        assert torch.equal(c1, c2), (
            "RNG 복원으로 cond 가 재현되지 않는다 — prepare_cond 안에 크롭 말고 RNG 를 "
            "쓰는 것이 생겼다. 두 정책이 다른 입력을 보게 되므로 학습을 멈춘다.")
        self._canary_done = True

    # ── 손실 ──────────────────────────────────────────────────────────────
    def __call__(self, policy, batch):
        from lerobot.utils.constants import ACTION

        if not self._canary_done:
            self._canary(batch)

        m = policy.diffusion
        T = m.noise_scheduler.config.num_train_timesteps

        # ① 두 정책이 **같은 크롭**을 보게 RNG 를 되감는다
        st = torch.get_rng_state()
        cond_t, x0 = prepare_cond(policy, batch)
        torch.set_rng_state(st)
        with torch.no_grad():
            cond_r, _ = prepare_cond(self.ref, batch)

        # ② 같은 t·eps 를 공유한다 (다르면 r 이 노이즈 차이를 잰다)
        B = x0.shape[0]
        t = torch.randint(0, T, (B,), device=x0.device, dtype=torch.long)
        eps = torch.randn_like(x0)

        eps_the, _, _ = unet_out(m, cond_t, x0, t, eps)
        with torch.no_grad():
            eps_ref, _, _ = unet_out(self.ref.diffusion, cond_r, x0, t, eps)
            # ③ z0 는 **엇갈린 짝**으로. 우리 데이터는 D/U 로 일부러 고른 것이라
            #    맞는 짝의 평균을 쓰면 기준점이 라벨에 오염된다 (KTO 원문).
            x0_mis = x0.roll(1, 0)
            mis_the, _, _ = unet_out(m, cond_t, x0_mis, t, eps)
            mis_ref, _, _ = unet_out(self.ref.diffusion, cond_r, x0_mis, t, eps)

        def reward(e_ref, e_the, target):
            se_ref = ((target - e_ref) ** 2).mean(dim=(1, 2))
            se_the = ((target - e_the) ** 2).mean(dim=(1, 2))
            return self.beta * T * (se_ref - se_the)

        r = reward(eps_ref, eps_the, eps)
        z0 = reward(mis_ref, mis_the, eps).mean().clamp_min(0.0).clamp(self.z0_lo, self.z0_hi).detach()

        # ④ 선호 — preference() 한 곳에서만 판정한다
        idx = batch["dataset_index"].cpu().numpy()
        sign_np, mag_np = preference(self.chunk_S[idx])
        sign = torch.as_tensor(sign_np, device=x0.device)
        mag = torch.as_tensor(mag_np, device=x0.device, dtype=x0.dtype)

        l1 = (eps - eps_the).abs().mean(dim=(1, 2))
        lam, wdiag = apo_weights(l1, t, self.m_t, sign, mag, self.beta_d, self.beta_u)

        u = torch.sigmoid(torch.where(sign > 0, r - z0, z0 - r))
        loss = -(lam * u).mean()

        out = {"r_mean": r.mean().item(), "r_std": r.std().item(), "z0": z0.item(),
               "u_mean": u.mean().item(),
               "sat_rate": (((u < 0.05) | (u > 0.95)).float().mean().item()),
               **wdiag}

        if self.bc_weight:
            l_bc = ((eps - eps_the) ** 2).mean()
            loss = loss + self.bc_weight * l_bc
            out["l_bc"] = l_bc.item()
        out["apo_frac"] = 1.0 if not self.bc_weight else abs(
            (-(lam * u).mean()).item()) / (abs(loss.item()) + 1e-12)
        return loss, out
