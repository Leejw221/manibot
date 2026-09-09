"""iDDPM (arXiv:2102.09672) 의 **학습된 분산 + L_hybrid** 를 Diffusion Policy 에 얹는다.

배경: 원본 Diffusion Policy 는 iDDPM 에서 **코사인 스케줄만** 가져왔다. 분산 학습도, t 의
importance sampling 도 안 쓴다. 그중 importance sampling 은 우리가 이미 기각했다
(EXP-11: eps 손실이 25~33% 좋아져도 x0·성공률로는 안 갔다). 남은 것이 분산 학습이고,
이 파일이 그걸 시험하기 위한 팔이다.

가설: 우도를 좋게 만드는 요소는 조작 성공률로 가지 않는다 — 그래서 DP 가 안 가져왔다.

원문과 다른 점 (명시):
  · t=0 항을 **연속 가우시안 NLL** 로 둔다. iDDPM 은 픽셀이 이산이라 discretized Gaussian
    을 쓰지만 행동은 연속이다.
  · L_hybrid = L_simple + λ·L_vlb 의 λ 는 논문값 0.001 을 기본으로 둔다. 공식 구현은
    대신 `T/1000` 으로 재조정한다 — 둘은 다른 정규화다.

⚠ 학습한 분산은 **DDPM 샘플링에서만** 쓰인다. DDIM(eta=0)은 역과정이 결정론적이라
   분산이 들어갈 자리가 없다.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from lerobot.policies.diffusion.modeling_diffusion import (DiffusionModel, DiffusionPolicy,
                                                           _make_noise_scheduler)


def _extract(arr: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """(T,) 계수 배열에서 배치별 t 를 뽑아 (B, 1, 1) 로 만든다."""
    return arr.to(t.device).gather(0, t).reshape(-1, *([1] * (ndim - 1)))


def _normal_kl(mean_q, logvar_q, mean_p, logvar_p):
    """두 대각 가우시안 사이의 KL. 원소별로 돌려준다."""
    return 0.5 * (logvar_p - logvar_q + torch.exp(logvar_q - logvar_p)
                  + (mean_q - mean_p) ** 2 * torch.exp(-logvar_p) - 1.0)


class IDDPMModel(DiffusionModel):
    """UNet 출력을 2배로 늘려 뒤쪽 절반을 분산 보간 계수로 쓴다."""

    def __init__(self, config, vlb_weight: float = 0.001):
        super().__init__(config)
        self.vlb_weight = float(vlb_weight)

        # ① UNet 이 (eps, v) 를 함께 내도록 마지막 1x1 conv 만 교체한다
        d = config.action_feature.shape[0]
        last = self.unet.final_conv[-1]
        self.unet.final_conv[-1] = nn.Conv1d(last.in_channels, 2 * d, 1)
        self.action_dim = d

        # ② 학습된 분산을 쓰려면 DDPM + learned_range 여야 한다
        self.noise_scheduler = _make_noise_scheduler(
            "DDPM",
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
            variance_type="learned_range",
        )
        # q(x_{t-1}|x_t, x_0) 의 분산·평균 계수를 미리 만들어 둔다
        ac = self.noise_scheduler.alphas_cumprod
        ac_prev = torch.cat([torch.ones(1, dtype=ac.dtype), ac[:-1]])
        betas = self.noise_scheduler.betas
        post_var = betas * (1.0 - ac_prev) / (1.0 - ac)
        self.register_buffer("_ac", ac, persistent=False)
        self.register_buffer("_ac_prev", ac_prev, persistent=False)
        self.register_buffer("_betas", betas, persistent=False)
        self.register_buffer("_log_betas", torch.log(betas), persistent=False)
        # t=0 에서 posterior 분산이 0 이라 로그가 발산한다 — 첫 항만 t=1 값으로 채운다
        self.register_buffer("_log_post_var",
                             torch.log(torch.cat([post_var[1:2], post_var[1:]])), persistent=False)
        self.register_buffer("_post_c0", betas * ac_prev.sqrt() / (1.0 - ac), persistent=False)
        self.register_buffer("_post_ct", (1.0 - ac_prev) * self.noise_scheduler.alphas.sqrt()
                             / (1.0 - ac), persistent=False)

    # ── 손실 ────────────────────────────────────────────────────────────
    def _split(self, pred):
        """UNet 출력 (B,T,2D) -> (eps, v).  v 는 [-1,1] 로 나와 [0,1] 로 옮긴다."""
        eps, v = pred[..., :self.action_dim], pred[..., self.action_dim:]
        return eps, (v + 1.0) / 2.0

    def _model_logvar(self, v, t):
        """Σ_θ = exp(v·log β_t + (1−v)·log β̃_t) — iDDPM Eq.15 의 보간."""
        lo = _extract(self._log_post_var, t, v.dim())
        hi = _extract(self._log_betas, t, v.dim())
        return v * hi + (1.0 - v) * lo

    def compute_loss(self, batch):
        global_cond = self._prepare_global_conditioning(batch)
        x0 = batch["action"]
        eps = torch.randn_like(x0)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                          (x0.shape[0],), device=x0.device).long()
        xt = self.noise_scheduler.add_noise(x0, eps, t)

        pred = self.unet(xt, t, global_cond=global_cond)
        eps_pred, v = self._split(pred)

        # L_simple — 평균(=eps) 만 학습한다.  원본 DP 의 손실과 같다
        l_simple = F.mse_loss(eps_pred, eps)

        # L_vlb — **분산만** 학습한다.  평균은 detach (iDDPM 3.2)
        eps_frozen = eps_pred.detach()
        ac = _extract(self._ac, t, x0.dim())
        x0_hat = (xt - (1 - ac).sqrt() * eps_frozen) / ac.sqrt()
        if self.noise_scheduler.config.clip_sample:
            r = self.noise_scheduler.config.clip_sample_range
            x0_hat = x0_hat.clamp(-r, r)
        c0 = _extract(self._post_c0, t, x0.dim())
        ct = _extract(self._post_ct, t, x0.dim())
        mean_p = c0 * x0_hat + ct * xt                 # p_θ 의 평균 (분산 학습용)
        mean_q = c0 * x0 + ct * xt                     # q 의 posterior 평균
        logvar_q = _extract(self._log_post_var, t, x0.dim())
        logvar_p = self._model_logvar(v, t)

        kl = _normal_kl(mean_q, logvar_q, mean_p, logvar_p)
        # t=0 은 KL 이 아니라 −log p(x_0|x_1).  행동은 연속이라 가우시안 NLL 을 쓴다
        nll = 0.5 * (math.log(2 * math.pi) + logvar_p + (x0 - mean_p) ** 2 * torch.exp(-logvar_p))
        vlb = torch.where((t == 0).reshape(-1, 1, 1), nll, kl).mean()

        # 두 항을 따로 남긴다 — 이게 없으면 "분산이 안 배워져서 차이가 없었다" 와
        # "배웠는데 성공률이 안 올랐다" 를 구별할 수 없다. 그게 이 실험의 핵심 질문이다.
        self.last_parts = {"l_simple": l_simple.detach(), "l_vlb": vlb.detach(),
                           "var_v_mean": v.detach().mean(), "var_v_std": v.detach().std()}
        return l_simple + self.vlb_weight * vlb

    # ── 샘플링 ──────────────────────────────────────────────────────────
    def conditional_sample(self, batch_size, global_cond=None, generator=None, noise=None):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        sample = noise if noise is not None else torch.randn(
            (batch_size, self.config.horizon, self.action_dim),
            dtype=dtype, device=device, generator=generator)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            out = self.unet(sample, torch.full(sample.shape[:1], t, dtype=torch.long,
                                               device=device), global_cond=global_cond)
            # ⚠ diffusers 는 dim=1 에서 분산을 뗀다(`torch.split(..., sample.shape[1], dim=1)`).
            #    우리 텐서는 (B,T,D) 라 dim=1 이 horizon 이므로 (B,D,T) 로 돌려서 넘긴다
            step = self.noise_scheduler.step(out.transpose(1, 2), t, sample.transpose(1, 2),
                                             generator=generator).prev_sample
            sample = step.transpose(1, 2)
        return sample


class IDDPMPolicy(DiffusionPolicy):
    """LeRobot DiffusionPolicy 와 인터페이스가 같다 — 모델만 바꿔 끼운다."""

    def __init__(self, config, dataset_stats=None, vlb_weight: float = 0.001, **kw):
        super().__init__(config, dataset_stats=dataset_stats, **kw)
        self.diffusion = IDDPMModel(config, vlb_weight=vlb_weight)

    def forward(self, batch):
        """부모와 같되 손실의 두 항을 로그로 내보낸다 (train.py 가 output_dict 를 wandb 로 보낸다)."""
        loss, _ = super().forward(batch)
        parts = getattr(self.diffusion, "last_parts", {})
        return loss, {k: float(v) for k, v in parts.items()}
