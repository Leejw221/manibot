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
    def __init__(self, ref, m_t, chunk_S, chunk_has=None, beta=30.0, beta_d=8.0,
                 beta_u=8.0, z0_clamp=(-5.0, 5.0), bc_weight=0.0, ref_mode="live",
                 expert_mag=1.0, z0_mode="batch_mean", n_t=8):
        self.ref = ref
        self.m_t = m_t
        # 샘플러가 푸는 것과 **같은 배열**. 기준이 둘이면 배치 구성이 손실에서 재현되지 않는다.
        self.chunk_S = chunk_S
        # 개입이 없던 에피소드를 expert(desirable) 로 읽기 위한 플래그. 없으면 예전 동작.
        self.chunk_has = chunk_has
        self.expert_mag = expert_mag
        # z0_mode: batch_mean = ELBO-KTO 의 Zero Compute Baseline (b0 = 배치 안 r̂ 의 평균).
        #   상수 baseline 중 분산 최적임이 증명돼 있다 [ELBO-KTO Lemma 1, 원문 직접 2026-09-12].
        #   mismatch = KTO 원래의 엇갈린 짝 추정. 우리 실측에서 r 과 척도가 달라 26->462 로
        #   폭주했고 sigmoid 를 완전히 포화시켰다 [측정 2026-09-12].
        self.z0_mode = z0_mode
        # n_t: 샘플당 뽑는 시점 수. 우리 실측에서 t 추첨이 MC 분산의 91% 였고,
        #   v ∝ 1/n_t 가 소수점까지 맞았다 (7838 -> 1930, 예측 4배). n_y 는 1 로 둔다
        #   — eps 축은 분산의 1.4% 뿐 [VRPO Prop 1 + 측정 2026-09-12].
        self.n_t = int(n_t)
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

        # ② 같은 t·eps 를 공유한다 (antithetic sampling — VRPO 의 "free lunch").
        #    n_t 번 뽑아 평균내면 ELBO 추정 분산이 1/n_t 로 준다.
        B = x0.shape[0]
        se_diff, l1_acc = 0.0, 0.0
        for _ in range(self.n_t):
            t = torch.randint(0, T, (B,), device=x0.device, dtype=torch.long)
            eps = torch.randn_like(x0)
            e_the, _, _ = unet_out(m, cond_t, x0, t, eps)
            with torch.no_grad():
                e_ref, _, _ = unet_out(self.ref.diffusion, cond_r, x0, t, eps)
            sq = lambda e: ((eps - e) ** 2).mean(dim=(1, 2))
            se_diff = se_diff + (sq(e_ref) - sq(e_the))
            # 가중치용 오차는 t 정규화 후 평균 — 그래야 "어느 t 가 뽑혔나" 가 안 섞인다
            l1_acc = l1_acc + (eps - e_the).abs().mean(dim=(1, 2)).detach() / self.m_t.to(x0.device)[t]

        # ③ beta 는 sigma **밖에** 둔다. r 안에 넣으면 z0 클램프·grad_clip 같은 상수가
        #    조용히 beta 에 딸려 움직인다 (실제로 beta 1->10 에서 클램프가 10배 조여졌다).
        r = T * se_diff / self.n_t                      # ELBO margin (beta 없음)
        l_tilde = l1_acc / self.n_t

        # ④ 선호 — preference() 한 곳에서만 판정한다
        idx = batch["dataset_index"].cpu().numpy()
        has = None if self.chunk_has is None else self.chunk_has[idx]
        sign_np, mag_np = preference(self.chunk_S[idx], has, expert_mag=self.expert_mag)
        sign = torch.as_tensor(sign_np, device=x0.device)
        mag = torch.as_tensor(mag_np, device=x0.device, dtype=x0.dtype)

        # ⑤ baseline
        if self.z0_mode == "none":
            # U 를 빼면 배치 평균 baseline 이 degenerate 하다 — 전부 D 이면
            # sum(r - b0) = 0 이라 절반은 반드시 평균 아래고, 방향이 사라진다.
            # z0=0 이면 u = sigma(beta*r) 로 "ref 보다 잘해라" 는 절대 기준이 된다.
            z0_raw = torch.zeros((), device=r.device, dtype=r.dtype)
            z0 = z0_raw
        elif self.z0_mode == "batch_mean":
            z0_raw = r.mean().detach()
            z0 = z0_raw
        else:
            mis_the, _, _ = unet_out(m, cond_t.roll(1, 0), x0, t, eps)
            with torch.no_grad():
                mis_ref, _, _ = unet_out(self.ref.diffusion, cond_r.roll(1, 0), x0, t, eps)
            z0_raw = (T * (((eps - mis_ref) ** 2).mean(dim=(1, 2))
                           - ((eps - mis_the) ** 2).mean(dim=(1, 2)))).mean().detach()
            z0 = z0_raw.clamp_min(0.0).clamp(self.z0_lo, self.z0_hi)

        lam, wdiag = apo_weights(l_tilde, None, self.m_t, sign, mag, self.beta_d, self.beta_u)
        u = torch.sigmoid(self.beta * sign.to(r.dtype) * (r - z0))
        loss = -(lam * u).mean()

        keep = sign != 0
        n_keep = keep.sum().clamp_min(1)
        sat = (((u < 0.05) | (u > 0.95)) & keep).sum() / n_keep
        out = {"r_mean": r.mean().item(), "r_std": r.std().item(), "z0": z0.item(),
               "z0_raw": z0_raw.item(), "u_mean": u.mean().item(), "sat_rate": sat.item(),
               "label_frac": keep.float().mean().item(), **wdiag}
        for nm, sel in (("r_d", sign > 0), ("r_u", sign < 0)):
            out[nm] = r[sel].mean().item() if sel.any() else float("nan")

        if self.bc_weight:
            l_bc = ((eps - e_the) ** 2).mean()
            loss = loss + self.bc_weight * l_bc
            out["l_bc"] = l_bc.item()
        out["apo_frac"] = 1.0 if not self.bc_weight else abs(
            (-(lam * u).mean()).item()) / (abs(loss.item()) + 1e-12)
        return loss, out
