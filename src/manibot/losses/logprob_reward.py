"""probability flow ODE 로 diffusion 정책의 log pi(a|o) 를 정확히 구한다.

왜 이게 필요한가: 지금 reward 는 T*(MSE_ref - MSE_theta) 인데 MSE_theta >= 0 이라 위로 유계다
  (r <= T*MSE_ref). 그 천장이 그룹마다 74배 갈려서 "기준 모델이 이미 잘하는 샘플"은 아무리 잘해도
  reward 를 못 받는다 — 시연의 천장이 0.64 인데 교정은 47.6 이다. 그래서 reward 가 샘플 품질이
  아니라 **기준 모델의 여유**를 재게 된다. Preference optimization 의 유도가 요구하는 양은
  log(pi_theta/pi_ref) 이고, 그건 위로 유계가 아니다.

근거 [Song et al., Score-Based Generative Modeling through SDEs, 원문 직접 2026-09-19]:
  - DDPM 은 VP SDE 의 이산화 (§3.2, Eq. 10->11):  dx = -0.5*beta(t)*x dt + sqrt(beta(t)) dw
  - 같은 주변분포를 갖는 결정론적 ODE (§4.3, Eq. 13):  dx = [f - 0.5*g^2*score] dt
  - 로그밀도 (Appendix D.2, Eq. 39):  log p_0(x0) = log p_T(x_T) + int div(f_tilde) dt
  - 발산 추정 (Eq. 40):  div = E_u[u^T (df/dx) u]   -- VJP 1회, forward 와 비슷한 비용

틀렸다면 어디서:
  1. t->0 특이점을 t0 = 1/T 에서 끊는다. 학습 격자의 첫 점이라 외삽은 아니지만, 여기서 나오는 건
     log p_0 가 아니라 log p_{t0} 다. theta 와 ref 에 **같은 t0** 를 쓰므로 차에서 상당 부분
     상쇄되지만 완전히는 아니다.
  2. RK4 고정 스텝. n_ode=20 과 80 의 차이가 그룹 중앙값 기준 0~7% 였다 [측정 2026-09-19].
  3. Hutchinson 은 불편추정이지만 분산이 있다. theta 와 ref 가 **같은 u** 를 쓰게 해 차의 분산을 줄인다.

검산 (두 개 다 통과했다 [2026-09-19]):
  - theta = ref 이면 두 로그밀도가 같아 차가 **정확히 0** 이어야 한다 -> 0.00 나옴
  - ODE 스텝 2배(40->80)에도 값이 안정해야 한다 -> 0.0~6.6% 변화
"""
import torch


def _interp(table, t, n_train):
    """연속 t in (0,1] 을 이산 인덱스 t*T-1 로 보고 선형보간. 학습 격자와 맞춘다."""
    p = (t * n_train - 1.0).clamp(0.0, n_train - 1.0)
    lo = p.floor().long()
    hi = (lo + 1).clamp(max=n_train - 1)
    w = p - lo.double()
    return table[lo] * (1 - w) + table[hi] * w


def _drift(unet, cond, x, t, betas, abar, n_train):
    """f_tilde = -0.5*beta(t)*(x - eps_theta/sigma_t)

    f = -0.5*beta*x, g^2 = beta, score = -eps/sigma 를 Eq. 13 에 넣어 정리한 것.
    beta(t) = T*beta_i 는 이산->연속 극한(Eq. 10->11)에서 온다.
    """
    tt = torch.as_tensor([t], device=x.device, dtype=torch.float64)
    b = (_interp(betas, tt, n_train) * n_train).squeeze()
    sig = (1.0 - _interp(abar, tt, n_train)).clamp_min(1e-12).sqrt().squeeze()
    idx = (tt * n_train - 1.0).clamp(0.0, n_train - 1.0).float().expand(x.shape[0])
    eps = unet(x.float(), idx, global_cond=cond)
    d = x.shape[-1]
    if eps.shape[-1] == 2 * d:                      # 학습된 분산 팔이 있으면 앞 절반만
        eps = eps[..., :d]
    return -0.5 * b * (x - eps.double() / sig)


def _drift_div(unet, cond, x, t, us, betas, abar, n_train):
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        f = _drift(unet, cond, x, t, betas, abar, n_train)
        div = 0.0
        for i, u in enumerate(us):
            g, = torch.autograd.grad((f * u).sum(), x, retain_graph=(i < len(us) - 1))
            div = div + (g * u).sum(dim=(1, 2))
    return f.detach(), (div / len(us)).detach()


def log_prob(unet, cond, a0, betas, abar, n_train, n_ode=20, n_hutch=1, generator=None):
    """log pi(a0|o) 를 샘플별로 돌려준다. shape (B,).

    파라미터 기울기는 흐르지 않는다 — 이 값은 **계수**를 정하는 데만 쓰고, 기울기는 디노이징
    손실에서 온다. ODE 를 역전파할 필요가 없는 이유가 이것이다.
    """
    t0, t1 = 1.0 / n_train, 1.0
    h = (t1 - t0) / n_ode
    x = a0.double().clone()
    acc = torch.zeros(a0.shape[0], device=a0.device, dtype=torch.float64)
    for k in range(n_ode):
        t = t0 + k * h
        # 스텝마다 u 를 고정 -> 같은 generator 를 넘기면 theta 와 ref 가 같은 u 를 본다
        us = [torch.randint(0, 2, a0.shape, generator=generator, device=a0.device,
                            dtype=torch.float64) * 2 - 1 for _ in range(n_hutch)]
        k1, d1 = _drift_div(unet, cond, x, t, us, betas, abar, n_train)
        k2, d2 = _drift_div(unet, cond, x + 0.5 * h * k1, t + 0.5 * h, us, betas, abar, n_train)
        k3, d3 = _drift_div(unet, cond, x + 0.5 * h * k2, t + 0.5 * h, us, betas, abar, n_train)
        k4, d4 = _drift_div(unet, cond, x + h * k3, t + h, us, betas, abar, n_train)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        acc = acc + h / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
    d = x[0].numel()
    log_pT = -0.5 * (x ** 2).sum(dim=(1, 2)) - 0.5 * d * torch.log(
        torch.tensor(2 * torch.pi, device=x.device, dtype=torch.float64))
    return log_pT + acc
