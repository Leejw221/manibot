"""Diffusion 정책의 공통 연산 — 여러 곳에서 같은 코드를 다시 쓰지 않으려고 모았다.

이 파일이 없으면 `iddpm_policy.py` · APO 손실 · 측정 스크립트가 각자
"배치를 조건으로 만들고 → 노이즈를 섞고 → UNet 을 부른다" 를 따로 구현하게 된다.
실제로 그런 사본이 세 벌 있었다 (2026-09-12 통합).

담는 것:
  prepare_cond   배치 -> global conditioning (이미지 스택 처리 포함)
  unet_out       (eps 예측, v, x_t) — 학습된 분산 팔이면 v 가 따라온다
  Posterior      q(x_(t-1)|x_t,x_0) 의 평균·분산 계수
  normal_kl      대각 가우시안 KL
  lambda_t       DDPM Eq.12 의 t 계수 (L_simple 이 떼어낸 것)
"""

import torch

__all__ = ["prepare_cond", "unet_out", "Posterior", "normal_kl", "extract", "lambda_t"]


def extract(arr, t, ndim):
    """(T,) 계수 배열에서 배치별 t 를 뽑아 (B, 1, ...) 로 만든다."""
    return arr.to(t.device).gather(0, t).reshape(-1, *([1] * (ndim - 1)))


def normal_kl(mean_q, logvar_q, mean_p, logvar_p):
    """두 대각 가우시안 사이의 KL. 원소별."""
    return 0.5 * (logvar_p - logvar_q + torch.exp(logvar_q - logvar_p)
                  + (mean_q - mean_p) ** 2 * torch.exp(-logvar_p) - 1.0)


def prepare_cond(policy, batch):
    """배치 -> (global_cond, x0).

    LeRobot 은 이미지 특징을 `OBS_IMAGES` 한 키에 쌓아서 받는다. n_obs_steps=1 일 때
    차원이 하나 빠져 있어 그대로 넘기면 조용히 틀린 축으로 쌓인다 — 그 처리까지 여기서 한다.
    """
    from lerobot.utils.constants import ACTION, OBS_IMAGES

    cfg, m = policy.config, policy.diffusion
    b = dict(batch)
    if cfg.image_features:
        for key in cfg.image_features:
            if cfg.n_obs_steps == 1 and b[key].ndim == 4:
                b[key] = b[key].unsqueeze(1)
        b[OBS_IMAGES] = torch.stack([b[key] for key in cfg.image_features], dim=-4)
    return m._prepare_global_conditioning(b), b[ACTION]


def unet_out(model, cond, x0, t, eps):
    """(eps_pred, v, x_t) 를 돌려준다.  `model` 은 정책이 아니라 **DiffusionModel** 이다
    — 그래야 `IDDPMModel.compute_loss` 안에서도 같은 함수를 쓸 수 있다.

    v 는 학습된 분산 팔(UNet 출력이 2*action_dim)일 때만 채워지고 아니면 None.
    ⚠ 두 정책(pi_theta, pi_ref)은 비전 인코더가 서로 달라 cond 를 **공유하면 안 된다**.
    """
    xt = model.noise_scheduler.add_noise(x0, eps, t)
    pred = model.unet(xt, t, global_cond=cond)
    d = x0.shape[-1]
    if pred.shape[-1] == 2 * d:
        return pred[..., :d], (pred[..., d:] + 1.0) / 2.0, xt
    return pred, None, xt


class Posterior:
    """q(x_(t-1)|x_t,x_0) 의 계수. 스케줄러 **객체 종류에 의존하지 않게** 여기서 한 번만 만든다
    (baseline 은 DDIM, iDDPM 팔은 DDPM 이라 클래스가 다르지만 계수는 같아야 한다)."""

    def __init__(self, scheduler, device="cpu"):
        ac = scheduler.alphas_cumprod.to(device)
        ac_prev = torch.cat([torch.ones(1, device=device), ac[:-1]])
        betas = scheduler.betas.to(device)
        alphas = scheduler.alphas.to(device)
        post_var = betas * (1.0 - ac_prev) / (1.0 - ac)

        self.ac, self.ac_prev, self.betas, self.alphas = ac, ac_prev, betas, alphas
        self.log_betas = torch.log(betas)
        # t=0 은 posterior 분산이 0 이라 로그가 발산한다 — iDDPM 과 같이 t=1 값으로 채운다
        self.log_post_var = torch.log(torch.cat([post_var[1:2], post_var[1:]]))
        self.post_var = post_var
        self.c0 = betas * ac_prev.sqrt() / (1.0 - ac)
        self.ct = (1.0 - ac_prev) * alphas.sqrt() / (1.0 - ac)

    def mean(self, x0, xt, t):
        """q 의 posterior 평균."""
        return extract(self.c0, t, x0.dim()) * x0 + extract(self.ct, t, x0.dim()) * xt


def lambda_t(posterior):
    """DDPM Eq.12 의 t 계수 — `L_simple` 이 떼어낸 것. sigma^2 = beta~_t 기준.

    ⚠ t=T-1 근처에서 alpha_t -> 0 이라 발산한다. 우리 T=100 코사인에서 t=99 하나가
    전체 합의 97.9% 를 먹는다 [측정 2026-09-12] — 쓸 거면 t 범위를 잘라야 한다.
    """
    p = posterior
    sig2 = torch.cat([p.post_var[1:2], p.post_var[1:]])
    return p.betas ** 2 / (2.0 * sig2 * p.alphas * (1.0 - p.ac))
