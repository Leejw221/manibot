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

import numpy as np
import torch

from manibot.policies.diffusion_ops import prepare_cond, unet_out
from manibot.utils.intervention_labels import preference

from .weighting import apo_weights

__all__ = ["APOLoss"]

# dsig(= sigma'(x)/0.25) 가 0.05 아래로 내려가는 x. 우리가 "그 그룹이 꺼졌다" 고
# 부르는 기준과 같은 값이다 — 데이터가 아니라 규약이므로 라운드가 바뀌어도 그대로다.
X_SAT = 4.36


class APOLoss:
    def __init__(self, ref, m_t, chunk_S, chunk_has=None, beta=30.0, beta_d=8.0,
                 beta_u=8.0, z0_clamp=(-5.0, 5.0), bc_weight=0.0, ref_mode="live",
                 expert_mag=1.0, z0_mode="batch_mean", n_t=8, use_mag=True,
                 mask_gripper_u=True, gripper_dim=-1, undesirable_weight=1.0,
                 z0_min=None, t_mode="uniform", beta_u_sigmoid=None,
                 reward_mode="mse", n_ode=20, n_hutch=1, chunk_is_demo=None,
                 logratio_scale=1.0, t_is=None, p_data_d=None, p_data_u=None,
                 u_grad_cap=False, u_budget_symmetric=False):
        # U 의 변화량 예산을 D 와 대칭으로 — 근거는 __call__ ⑥ 주석.
        self.u_budget_symmetric = bool(u_budget_symmetric)
        # U 기울기 상한 — 근거는 __call__ 의 해당 블록 주석.
        self.u_grad_cap = bool(u_grad_cap)
        # t_is: reward 추첨용 제안분포 p(t). None 이면 균등(= 지금까지의 동작).
        #   iDDPM §3.3 [원문 직접]: L = E_{t~p}[L_t / p_t],  p_t ∝ sqrt(E[L_t²]).
        #   원칙은 "**합산하는 항의 크기**에 비례" 다. iDDPM 이 합산하는 항은 loss 지만
        #   우리가 합산하는 항은 **차** d = s_ref - s_the 다. 펼치면
        #       d = (e_the - e_ref)·(2eps - e_ref - e_the)   ->   |d| ~ |Δe| x ||eps - e_ref||
        #   라 오차 척도를 **한 제곱만** 물려받는다. 그래서 p ∝ sqrt(s_ref).
        #   s_ref 자체(제곱)에 비례시키면 저-t 로 과하게 몰려 이득이 사라진다
        #   [측정 2026-09-20: 신호/잡음 균등 1.44 · sqrt 3.07 · 제곱 1.44].
        #   ⚠ IS 는 **어떤 p 를 써도 무편향**이다 — 표가 부정확해도 편향이 아니라 분산만 는다.
        self.t_is = t_is
        # 자연 비율 z0 용 (z0_mode="natural"). balanced 배치가 U 를 12배 과대표집한다
        # (데이터 2.06% vs 배치 25%) — 실측에서 z0 가 -257 까지 내려갔다 [2026-09-20].
        self.p_data_d, self.p_data_u = p_data_d, p_data_u
        # reward_mode: "mse"(기존) = T*(MSE_ref - MSE_theta) · "logratio" = log pi_theta - log pi_ref
        #   mse 는 MSE_theta >= 0 이라 위로 유계고(r <= T*MSE_ref) 그 천장이 그룹마다 74배 갈린다
        #   — 시연 0.64 / 교정 47.6. 그래서 "기준 모델이 이미 잘하는 샘플"은 아무리 잘해도 reward
        #   를 못 받는다. logratio 는 유도(KL 제약 하 reward 최대화의 최적해)가 요구하는 바로 그
        #   양이고 위로 유계가 아니다. 교정/시연 범위 비가 154배 -> 5.4배 [측정 2026-09-19].
        self.reward_mode = reward_mode
        self.n_ode, self.n_hutch = n_ode, n_hutch
        # 그룹별 wandb 지표용. 시연/롤아웃을 가르려면 에피소드 번호가 필요하다.
        self.chunk_is_demo = chunk_is_demo
        # logratio 손실은 계수(<=0.0375) x l_theta(0.01~0.5) 라 기존 -(lam*u).mean()(약 0.5)보다
        # 50~1000배 작다. 같은 lr 이면 실효 학습률이 그만큼 죽는다. 배치 평균으로 정규화하면
        # **포화 신호가 사라지므로**(다 포화해도 손실이 안 준다) 고정 상수로 둔다. 값은 step 0 의
        # grad_norm 을 기존 설정과 맞춰서 정한다.
        self.logratio_scale = logratio_scale
        self.ref = ref
        self.m_t = m_t
        # 샘플러가 푸는 것과 **같은 배열**. 기준이 둘이면 배치 구성이 손실에서 재현되지 않는다.
        self.chunk_S = chunk_S
        # 개입이 없던 에피소드를 expert(desirable) 로 읽기 위한 플래그. 없으면 예전 동작.
        self.chunk_has = chunk_has
        self.expert_mag = expert_mag
        # mag = |S| 는 APO 원문에 없는 우리 추가물이다. false 면 1 로 두고 부호만 쓴다
        # — kappa 측정에서 초기 업데이트의 48.8% 를 개입직전에 몰아주고 있었다 [2026-09-13].
        self.use_mag = use_mag
        # t_mode: "uniform"(기존) | "lam_is"(reward 를 lambda(t) 가중으로 바꾸고 t 를 lambda 비례 추출)
        self.t_mode = t_mode
        self._is = None
        # U 의 reward 에서 뺄 행동 차원(기본 = 마지막 = gripper). 근거는 __call__ ③ 주석.
        self.mask_gripper_u = mask_gripper_u
        self.gripper_dim = gripper_dim
        # 0 이면 U 가 손실에 기여하지 않는다 — **밀어내기를 끄는 진단용**.
        # 배치 구성은 그대로 두므로(4/2/2) 바뀌는 것이 하나뿐이다. 대신 유효 배치가
        # 8 -> 6 으로 준다 (옛 프로젝트 8-A 와 같은 내재된 부작용).
        self.undesirable_weight = undesirable_weight
        # z0 하한. batch_mean 은 U 에 끌려 음수로 내려가고(R2 최대 -25.5), 그러면
        # desirable 이 ref 보다 나빠도(r<0) "이겼다" 로 판정된다.
        # ⚠ 이건 KL 복원이 아니다 — 우리 z0 는 T x MSE 차이지 KL 이 아니고, 원문의
        #   clamp(-5,5) 도 음수를 막지 않는다 [원문 코드 직접 2026-09-15].
        #   **설계 선택**이다: 기준점이 "ref 대비 개선량 0" 아래로 안 내려가게 한다.
        self.z0_min = z0_min
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
        # undesirable 쪽 효용의 beta. None 이면 beta 하나를 양쪽에 쓴다(원문 그대로).
        #
        # 왜 가를 수 있게 뒀나 — KTO 의 beta 가 하나여도 되는 건 r 이 **무차원 로그비**라
        # 클래스마다 척도가 같기 때문이다. 우리 r 은 T x (MSE_ref - MSE_theta) 라 척도가 갈린다:
        # 실측 l_ref 중앙값이 시연 0.0062 · 배포성공 0.0331 · 교정 0.443 으로 71배 차이다
        # [probe 2026-09-18, logs/apo_probe_r1b].
        # 게다가 부호가 비대칭이다 — desirable 의 r 은 위로 유계(r <= T*MSE_ref, 교정은 이미
        # 천장의 79%)인데 undesirable 은 아래로 무계다(beta=0.3 -> -114 · 0.05 -> -645, 둘 다
        # x=32~34 에서야 멈춘다). 그래서 하나의 beta 로는 "U 를 일찍 포화시키면서 교정을 활성
        # 구간에 두기" 가 불가능하다 — 교정 활성엔 beta <~ 0.057, U 조기 포화엔 큰 beta 가 필요하다.
        # KTO 가 beta 에 준 의미는 "sigma' 가 살아 있는 폭" 이므로, 그 의미를 두 클래스 모두에서
        # 성립시키려면 척도가 갈리는 축을 따라 beta 도 갈라야 한다.
        self.beta_u_sigmoid = beta_u_sigmoid
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

        # ② 선호 — preference() 한 곳에서만 판정한다.  **오차 계산보다 먼저** 구한다:
        #    U 의 reward 에서 gripper 차원을 빼려면 sq() 안에서 부호를 알아야 하기 때문이다.
        idx = batch["dataset_index"].cpu().numpy()
        has = None if self.chunk_has is None else self.chunk_has[idx]
        sign_np, mag_np = preference(self.chunk_S[idx], has, expert_mag=self.expert_mag)
        sign = torch.as_tensor(sign_np, device=x0.device)
        if not self.use_mag:
            mag_np = np.ones_like(mag_np)
        mag = torch.as_tensor(mag_np, device=x0.device, dtype=x0.dtype)

        # ③ gripper 마스크 — U 의 reward 에서 gripper 차원을 뺀다.
        #    gripper 는 거의 열림/닫힘 이진이라 같은 값이 D 와 U 에 함께 나타난다. U 를 밀면
        #    정상 gripper 행동까지 같이 밀린다.  e_ref 와 e_theta 에 **똑같이** 걸어야 r 이
        #    공정한 차로 남고, 유효 원소 수로 나눠야 D 와 척도가 맞는다.
        #    ⚠ 출처: APO **공개 코드에는 없다** — rejected 의 `[:, :-1]` 은 gripper 가 아니라
        #      stop 토큰을 뺀다(labels 는 action 7토큰 + stop, predict_stop_token 기본 True)
        #      [코드 원문 직접 2026-09-14]. 논문 Appendix C 의 gripper 서술은 미확인.
        #      즉 이건 원문 이식이 아니라 **우리 설계 판단**이다.
        B, H, A = x0.shape
        dmask = torch.ones(B, 1, A, device=x0.device, dtype=x0.dtype)
        if self.mask_gripper_u:
            dmask[sign < 0, :, self.gripper_dim] = 0.0
        n_eff = dmask.sum(dim=2).squeeze(1) * H          # 샘플별 유효 원소 수

        # ④ 같은 t·eps 를 공유한다 (antithetic sampling — VRPO 의 "free lunch").
        #    n_t 번 뽑아 평균내면 ELBO 추정 분산이 1/n_t 로 준다.
        if self.t_mode == "lam_is" and self._is is None:
            self._is = _lam_is_tables(m.noise_scheduler, x0.device)

        sq = lambda e, ep: (((ep - e) ** 2) * dmask).sum(dim=(1, 2)) / n_eff

        # ④-1 **reward 추첨** — 기울기는 여기서 안 흐른다. r 은 계수만 정한다.
        #     t_is 가 있으면 그 분포에서 뽑고 1/p 로 보정한다. 균등은 p=1/T 인 특수 경우라
        #     보정이 T 가 되어 기존 `T * mean_k d` 와 **정확히 같은 식**이 된다.
        se_diff, se_mis, se_ref_acc = 0.0, 0.0, 0.0
        probe_k = []  # 진단용: k 번째 추첨의 (t, eps, 청크별 ref 오차). 학습 경로에는 안 쓰인다
        with torch.no_grad():
            for _ in range(self.n_t):
                if self.t_mode == "lam_is":
                    lo, p_t, isw_tab = self._is
                    t = torch.multinomial(p_t.expand(B, -1), 1).squeeze(1) + lo
                    isw = isw_tab[t] * T
                elif self.t_is is not None:
                    p = self.t_is.to(x0.device)
                    t = torch.multinomial(p.expand(B, -1), 1).squeeze(1)
                    isw = 1.0 / p[t]
                else:
                    t = torch.randint(0, T, (B,), device=x0.device, dtype=torch.long)
                    isw = torch.full((B,), float(T), device=x0.device, dtype=x0.dtype)
                eps = torch.randn_like(x0)
                e_the, _, _ = unet_out(m, cond_t, x0, t, eps)
                e_ref, _, _ = unet_out(self.ref.diffusion, cond_r, x0, t, eps)
                s_ref = sq(e_ref, eps)
                se_diff = se_diff + isw.to(x0.dtype) * (s_ref - sq(e_the, eps))
                # r 과 **같은 추정기**로 T*s_ref 도 모은다 (대칭 예산용, ⑥ 참조).
                se_ref_acc = se_ref_acc + isw.to(x0.dtype) * s_ref
                probe_k.append((t, eps, s_ref))
                if self.z0_mode == "mismatch":
                    # KTO 원문의 엇갈린 짝. **r 과 같은 t·eps·보정**을 쓴다 — 2026-09-12 의
                    # "26->462 폭주" 는 방법이 아니라 구현 탓이었다: 그때 z0 는 루프 **밖**의
                    # t·eps 를 써 1회 추첨이었고 r 은 8회였다(MC 분산 8배). 척도도 달랐다
                    # (dmask·n_eff 없이 .mean, 그 위에 T 곱). 여기서는 셋 다 r 과 같게 맞춘다.
                    mis_the, _, _ = unet_out(m, cond_t.roll(1, 0), x0, t, eps)
                    mis_ref, _, _ = unet_out(self.ref.diffusion, cond_r.roll(1, 0), x0, t, eps)
                    se_mis = se_mis + isw.to(x0.dtype) * (sq(mis_ref, eps) - sq(mis_the, eps))

        # ④-2 **기울기 추첨 — 항상 균등**. IS 를 여기 걸면 기울기 질량이 저-t 로 옮겨가는데,
        #     EXP-11 실측에서 그러면 eps 손실은 25~33% 좋아지지만 **x0(행동) 공간으로 안 간다**
        #     (저-t 는 sqrt((1-abar_t)/abar_t) 가 가장 작아 행동 오차로 환산하면 제일 안 중요하다).
        #     그래서 "r 을 어떻게 추정하나" 와 "기울기가 어느 t 에서 오나" 를 **분리**한다.
        #     같은 이유로 adaptive weight 의 l1 도 이 균등 경로에서 뽑는다.
        se_the, l1_acc = 0.0, 0.0
        for _ in range(self.n_t):
            t = torch.randint(0, T, (B,), device=x0.device, dtype=torch.long)
            eps = torch.randn_like(x0)
            e_the, _, _ = unet_out(m, cond_t, x0, t, eps)
            se_the = se_the + sq(e_the, eps)
            # 가중치용 오차에는 마스크를 안 건다 — APO 의 adaptive weight 는 7차원 전부의 L1 이다
            # (ddp_robotic_trainer.py L294, action_space=7) [코드 원문 직접 2026-09-14].
            l1_acc = l1_acc + (eps - e_the).abs().mean(dim=(1, 2)).detach() / self.m_t.to(x0.device)[t]

        # ③ beta 는 sigma **밖에** 둔다. r 안에 넣으면 z0 클램프·grad_clip 같은 상수가
        #    조용히 beta 에 딸려 움직인다 (실제로 beta 1->10 에서 클램프가 10배 조여졌다).
        r_mse = se_diff / self.n_t                      # sum_t d(t) 의 무편향 추정
        r_ref = se_ref_acc / self.n_t                   # = T*s_ref 의 추정 (r 과 같은 단위)
        l_the = se_the / self.n_t                       # 샘플별 디노이징 손실 — **기울기가 산다**
        l_tilde = l1_acc / self.n_t
        r = r_mse

        # ⑤' logratio 모드 — probability flow ODE 로 log pi 를 정확히 구한다 (Eq. 39/40).
        #    파라미터 기울기는 흐르지 않는다: 이 값은 **계수**만 정하고 기울기는 l_the 에서 온다.
        #    그래서 ODE 를 역전파할 필요가 없다 (교수님 43:07 "학습 로스에서는 계산을 못 할 것 같다").
        if self.reward_mode == "logratio":
            from manibot.losses.logprob_reward import log_prob
            sch = m.noise_scheduler
            betas = sch.betas.to(x0.device).double()
            abar = sch.alphas_cumprod.to(x0.device).double()
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,), device="cpu").item())
            with torch.no_grad():
                # theta 와 ref 에 **같은 시드** -> 같은 Hutchinson 벡터 -> 차의 분산이 준다
                g1 = torch.Generator(device=x0.device).manual_seed(seed)
                lp_the = log_prob(m.unet, cond_t, x0, betas, abar, T,
                                  self.n_ode, self.n_hutch, g1)
                g2 = torch.Generator(device=x0.device).manual_seed(seed)
                lp_ref = log_prob(self.ref.diffusion.unet, cond_r, x0, betas, abar, T,
                                  self.n_ode, self.n_hutch, g2)
                r = (lp_the - lp_ref).to(r_mse.dtype)

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
        elif self.z0_mode == "natural":
            # z0 는 "데이터 분포에 대한 기댓값" 인데 balanced 배치가 U 를 12.1배 과대표집한다
            # (데이터 2.06% vs 배치 25%). 그래서 배치 평균 z0 가 U 에 끌려 -257 까지 내려가고,
            # r/demo = -4.2 인 D 의 마진이 +271 이 되어 **D 가 먼저 꺼진다** [측정 2026-09-20].
            # 샘플링·손실은 그대로 두고 **평균 내는 방식만** 자연 비율로 되돌린다.
            #   w = 데이터 비율 / 배치 비율  (역수로 하면 U 가 더 무거워져 방향이 뒤집힌다)
            p_b_d = (sign > 0).to(r.dtype).mean().clamp_min(1e-8)
            p_b_u = (sign < 0).to(r.dtype).mean().clamp_min(1e-8)
            w = torch.where(sign > 0, self.p_data_d / p_b_d,
                            torch.where(sign < 0, self.p_data_u / p_b_u, torch.zeros_like(r)))
            z0_raw = ((w * r).sum() / w.sum().clamp_min(1e-12)).detach()
            z0 = z0_raw
        elif self.z0_mode == "desirable_mean":
            # 배치 평균은 U 에 납치된다 — 개입직전의 r 이 -60 까지 내려가 z0 를 -11 로
            # 끌어내리고, 그러면 desirable 이 ref 보다 나빠도(r<0) "이겼다"로 판정된다.
            # 실측(8k, 128배치): 개입 0-29 의 4u(1-u) 중앙값이 0.0002 까지 죽어 있었고,
            # z0 를 desirable 기준으로 되돌리면 그 샘플들의 kappa 가 173배가 된다 [2026-09-14].
            d = sign > 0
            z0_raw = (r[d].mean() if d.any() else r.mean()).detach()
            z0 = z0_raw
        else:
            # mismatch — 엇갈린 짝. **손실이 밀고 있지 않은 샘플**로 기준점을 잰다는 게 요점이다.
            # batch_mean·natural 은 둘 다 r 자체의 평균이라, 손실이 U 를 밀면 z0 가 따라 내려가는
            # 되먹임이 남는다 (자연비율로도 U 의 |r| 이 demo 의 250배라 2.16%만 실려도 z0 를 끈다
            # — 실측 D팔 z0 가 -32 까지 단조 하락 [2026-09-20]).
            # 엇갈린 짝엔 선호 라벨이 붙지 않으므로 그 되먹임이 구조적으로 끊긴다.
            # ⚠ 클램프를 안 쓴다. KTO 는 max(0,·) 를 쓰지만 그건 z0 를 KL 로 읽기 때문이고,
            #   엇갈린 짝 추정량은 y'~데이터라 **진짜 KL 이 아니고 음수가 자연스럽다**
            #   (KTO 가 max(0,·) 를 붙인 것 자체가 그 증거다). 사용자 판단:
            #   "초반에 ref=theta니까 0이겠네. 그러면 0에서 양수하고 음수를 왔다갔다해야" [2026-09-20].
            z0_raw = (se_mis / self.n_t).mean().detach()
            z0 = z0_raw

        if self.z0_min is not None:
            z0 = z0.clamp_min(self.z0_min)
        lam, wdiag = apo_weights(l_tilde, None, self.m_t, sign, mag, self.beta_d, self.beta_u)
        if self.undesirable_weight != 1.0:
            lam = torch.where(sign < 0, lam * self.undesirable_weight, lam)
        # ⑥ undesirable 쪽 beta — **대칭 예산**에서 유도한다.
        #
        # 왜 필요한가: r_D 는 위로 유계(s_the >= 0 이라 r_D <= T*s_ref)인데 r_U 는 아래로 무계다.
        #   mse 로 옮기면서 생긴 **부호 비대칭**이고(KTO 의 로그비는 양쪽 다 무계라 대칭),
        #   그래서 U 의 변화량이 D 의 350배까지 갔다 [측정 2026-09-21].
        #
        # 대칭 조건: "U 의 오차 증가폭이 D 가 낼 수 있는 최대 감소폭(= s_ref)을 못 넘는다"
        #     -> r_U >= -T*s_ref = -r_ref
        #   그 지점에서 시그모이드가 포화하게 beta_u 를 잡으면 (X_SAT = dsig<0.05 경계)
        #     beta_u,i = X_SAT / r_ref_i
        #   z0 ~ 0 일 때 x_i = X_SAT * |1 - s_the_i/s_ref_i| 가 되어, **포화가 정확히
        #   "그 샘플의 오차가 두 배" 지점**이다 — 샘플의 원래 척도와 무관하다.
        #   상수는 X_SAT 하나뿐이고 그건 데이터가 아니라 "언제를 꺼진 것으로 보나" 라는 규약이다.
        #
        # ⚠ z0 ~ 0 가정이 들어 있다. z0 가 0 에서 멀면 x 가 상대 배수의 함수가 아니게 된다
        #   — 로그의 z0 와 x/U 로 확인할 것.
        if self.u_budget_symmetric:
            assert self.reward_mode == "mse", (
                "대칭 예산은 r 과 r_ref 가 같은 단위여야 한다 — logratio 는 ODE 로그비라 다르다")
            b_u = (X_SAT / r_ref.clamp_min(1e-6)).to(r.dtype)
        elif self.beta_u_sigmoid is not None:
            b_u = torch.full_like(r, float(self.beta_u_sigmoid))
        else:
            b_u = None
        b = self.beta if b_u is None else torch.where(
            sign > 0, torch.full_like(r, float(self.beta)), b_u)
        u = torch.sigmoid(b * sign.to(r.dtype) * (r - z0))
        # KTO 손실을 **계수 x 디노이징 손실** 로 쓴다. 원래 기울기는
        #     dL/dtheta = -( lam*b*sign*sigma'(x) * dr/dtheta ).mean()
        # 이고 대괄호 안은 고정 시점에서 상수로 둘 수 있다.
        #   mse      : dr/dtheta = -T * d(l_the)/dtheta  ->  scale = T. **유도된 값**이다.
        #              (기존 `-(lam*u).mean()` 과 같은 기울기를 준다 — 근사가 아니라 항등)
        #   logratio : dr/dtheta = d(log pi_theta)/dtheta 이고 **로그확률을 올리는 표준 기울기가
        #              디노이징 손실을 내리는 기울기**다 [Diffusion-DPO Eq.14].
        #   desirable(sign=+1): 계수>0 x l_the -> l_the 를 줄인다(배운다)
        #   undesirable(sign=-1): 계수<0 x l_the -> l_the 를 키운다(밀어낸다)
        # 이 형태로 쓰는 이유는 **l_the 의 t 를 r 의 t 와 분리**하기 위해서다 (④-2 참조).
        coef = (lam * b * u * (1.0 - u)).detach()
        # ── U 기울기 상한 ────────────────────────────────────────────────
        # **APO 원문이 갖고 있는데 우리 이식본이 잃어버린 성질을 되살린다.**
        # 이산 토큰: d(log pi)/d(logits) = e_y - pi  ->  ‖∇‖ <= sqrt(2). 아무리 틀려도 안 커진다.
        # 연속(우리): d(log pi)/d(eps_theta) = 2(eps - eps_theta)  ->  **오차에 비례해 커진다.**
        # 실측 [2026-09-21]: U 의 오차가 ref 의 155배까지 가고 eps 공간 이동량이 intv 의 350배.
        #   그 결과 demo 가 step 100 에 이미 7.4배 나빠졌다 — 기울기가 sqrt(l_the) 에 비례해
        #   **제일 잘 맞는 샘플이 제일 약한 방어력**을 갖기 때문(demo 0.042 vs U 6.9, 164배).
        # 규칙: "밀기가 지키기를 못 넘는다" — undesirable 의 기울기 크기가 desirable 쪽
        #   **중앙값**을 넘지 않게 계수를 줄인다. 임의 상수가 없고 배치마다 스스로 맞춘다.
        #   (demo 하나만 기준으로 잡는 변형도 있으나, 그러려면 l_ref/demo 를 라운드마다
        #    다시 재서 넘겨야 한다 — 여기서는 배치 안에서 닫히는 쪽을 골랐다.)
        u_cap_med = float("nan")
        if self.u_grad_cap:
            g = coef.abs() * l_the.detach().clamp_min(0).sqrt()   # 기울기 크기 대용
            d_sel, u_sel = sign > 0, sign < 0
            if d_sel.any():
                g_d = g[d_sel].median()
                cap = (g_d / g.clamp_min(1e-12)).clamp(max=1.0)
                coef = torch.where(u_sel, coef * cap, coef)
                if u_sel.any():
                    u_cap_med = cap[u_sel].median().item()   # 1 이면 상한이 안 걸린 것
        scale = self.logratio_scale if self.reward_mode == "logratio" else float(T)
        loss = (coef * sign.to(r.dtype) * l_the).mean() * scale
        # 진단용 샘플별 값 — 배치 평균 로그만으로는 그룹별 포화 방향을 못 가른다 (2026-09-17)
        # l_the 는 **기울기가 살아 있는 채로** 남긴다 — 그룹별 기울기 분해 진단에 쓴다
        # (detach 하면 분해가 불가능하다). 학습 경로는 안 바뀐다.
        self.last = {"l_the": l_the, "coef_b": b,
                     "r": r.detach(), "z0": z0.detach(), "z0_raw": z0_raw.detach(), "lam": lam.detach(),
                     "u": u.detach(), "sign": sign, "cond_t": cond_t.detach(), "cond_r": cond_r,
                     "x0": x0, "dmask": dmask, "n_eff": n_eff, "k": probe_k}

        # D 와 U 가 손실을 나눠 갖는 비율. **KTO 식 (9) 의 개수 비율이 아니다** — 그쪽 n_D, n_U 는
        # 데이터 개수이고 lam 은 클래스 상수다. 여기 lam 은 APO 의 샘플별 중요도라 물건이 다르다.
        # 진단용으로만 본다 (2026-09-18 에 이 둘을 혼동해 balance_ratio 를 만들었다가 되돌렸다).
        with torch.no_grad():
            _d, _u = lam[sign > 0].sum(), lam[sign < 0].sum()
            bal = float(_d / _u) if _u > 0 else float("nan")
        keep = sign != 0
        n_keep = keep.sum().clamp_min(1)
        sat = (((u < 0.05) | (u > 0.95)) & keep).sum() / n_keep
        out = {"r_mean": r.mean().item(), "r_std": r.std().item(), "z0": z0.item(),
               "z0_raw": z0_raw.item(), "u_mean": u.mean().item(), "sat_rate": sat.item(),
               "label_frac": keep.float().mean().item(), "bal": bal,
               "u_cap": u_cap_med, **wdiag}
        if self.u_budget_symmetric:
            _u = sign < 0
            if _u.any():
                # r/r_ref = 1 - s_the/s_ref.  -1 이면 오차가 딱 2배 = 예산 소진 지점.
                out["b_u"] = b[_u].median().item()
                out["u_spent"] = (-(r / r_ref.clamp_min(1e-6)))[_u].median().item()
        for nm, sel in (("r_d", sign > 0), ("r_u", sign < 0)):
            out[nm] = r[sel].mean().item() if sel.any() else float("nan")

        # 그룹별 지표 — 배치 평균만 보면 "어느 그룹이 꺼졌나" 를 못 가른다. beta 를 포화로 정했으므로
        # dsig(= sigma'(x)/0.25) 가 그 선택의 직접 검정이다: U 는 꺼지고 시연은 살아 있어야 한다.
        with torch.no_grad():
            x_sig = (b * sign.to(r.dtype) * (r - z0)).detach()
            dsig = (u * (1.0 - u) / 0.25).detach()
            coef_abs = (lam * b * u * (1.0 - u)).detach().abs()
            tot = coef_abs.sum().clamp_min(1e-12)
            has_np = np.zeros(len(sign_np), dtype=bool) if has is None else np.asarray(has)
            dem = (np.asarray(self.chunk_is_demo)[idx] if self.chunk_is_demo is not None
                   else np.ones(len(sign_np), dtype=bool))
            # ⚠ 그룹은 **승격 전 S** 로 가른다. preference() 가 S=0 을 +1 로 올리므로
            #   (sign>0) 로 나누면 "개입에서 먼 정책 구간" 이 개입 그룹에 섞인다
            #   [2026-09-20: 고친 직후 intv 16 -> 29 로 보여 잡았다].
            S_np = np.asarray(self.chunk_S[idx], dtype=np.float64)
            _t = 1e-6
            groups = {"intv": has_np & (S_np > _t), "U": has_np & (S_np < -_t),
                      "policy": has_np & (np.abs(S_np) <= _t),   # 개입에서 먼 정책 구간
                      "demo": (~has_np) & dem, "rollout": (~has_np) & (~dem)}
            for g, msk in groups.items():
                if not msk.any():
                    continue
                sel = torch.as_tensor(msk, device=r.device)
                out[f"r/{g}"] = r[sel].median().item()
                out[f"x/{g}"] = x_sig[sel].median().item()
                out[f"dsig/{g}"] = dsig[sel].median().item()
                out[f"coef_share/{g}"] = (coef_abs[sel].sum() / tot).item()
                out[f"l_the/{g}"] = l_the[sel].detach().median().item()
            if self.reward_mode == "logratio":
                out["r_mse_mean"] = r_mse.mean().item()      # 옛 축도 같이 본다 (대조용)
                out["lp_the"] = lp_the.mean().item()
                out["lp_ref"] = lp_ref.mean().item()

        if self.bc_weight:
            l_bc = ((eps - e_the) ** 2).mean()
            loss = loss + self.bc_weight * l_bc
            out["l_bc"] = l_bc.item()
        out["apo_frac"] = 1.0 if not self.bc_weight else abs(
            (-(lam * u).mean()).item()) / (abs(loss.item()) + 1e-12)
        return loss, out


def _lam_is_tables(sched, device):
    """lambda(t) 비례 추출용 (lo, p_t, isw) 를 만든다.

    lambda(t) = beta_t^2 / (2 * beta~_t * alpha_t * (1-abar_t))  — 고정분산·eps 파라미터화에서
    lambda(t)*[||eps-eps_ref||^2 - ||eps-eps_theta||^2] 가 **reverse-KL 항의 차**와 같다 [검산 2026-09-15].
    ⚠ 이건 로그확률비가 아니다 (log p = ELBO + G, gap 차의 부호 미정).
    ⚠ S = 코드 1..98 (논문 t=2..99).  제외: L_0(decoder) · L_{T-1}(alpha_T 퇴화) · L_T(prior, 차에서 상쇄).
      끝점 제외는 **우리 판단**이다.
    """
    ac = sched.alphas_cumprod.to(device).double()
    ac_prev = torch.cat([torch.ones(1, device=device, dtype=ac.dtype), ac[:-1]])
    beta = (1 - ac / ac_prev).clamp(max=0.999)
    alpha = 1 - beta
    bt = (1 - ac_prev) / (1 - ac) * beta
    lam = beta ** 2 / (2 * bt * alpha * (1 - ac))
    lo, hi = 1, len(ac) - 1                       # 코드 1..98
    ls = lam[lo:hi]
    lam_n = ls / ls.mean()                        # lambda~,  평균 1
    p = ls / ls.sum()                             # p_t
    P = hi - lo
    isw_s = lam_n / (P * p)                       # == 1 (수치로 확인)
    isw = torch.ones_like(lam)
    isw[lo:hi] = isw_s
    return lo, p.float(), isw.float()
