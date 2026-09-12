"""APO 의 adaptive reweighting + 우리가 추가한 t 정규화.

원문 [원문 직접 2026-09-12, APO §3.2 + Algorithm 1]:
    l_i = |pi_theta(o_i) - a_i|_1
    w_i = l_i / sum_j l_j                      (배치 정규화)
    lambda_D = 1 - exp(-beta_D * w_i)          desirable:   오차 클수록 크게
    lambda_U =     exp(-beta_U * w_i)          undesirable: 오차 작을수록 크게
                                               (= 정책이 이미 잘 내는 실패 행동을 더 민다)

**우리가 더한 것 — t 정규화.** APO 는 autoregressive VLA 라 디퓨전 t 가 없다. 우리는 배치에
t 가 섞여 있고, 예측 오차의 크기가 t 에 따라 **38배** 다르다 [측정 2026-09-12]. 정규화하지
않으면 w_i 가 "이 샘플이 어려운가" 가 아니라 **"어느 t 가 뽑혔나"** 를 잰다 — t 간 변동이
샘플 간 변동의 4.6배였다. m[t] 로 나누면 그 축이 사라진다 (t 구간 평균의 변동 0.982 -> 0.007).

m[t] 는 pi_ref 로 **라운드 시작에 한 번** 재서 고정한다. 갱신하지 않는 근거:
  · 모양이 정책에 거의 무관하다 (다른 시드 정책과 상관 0.9998)
  · 전체 스케일은 뒤따르는 배치 정규화가 지운다

beta_D=beta_U=8 은 APO 의 1.0 을 batch 8 -> 64 로 보정한 값이다 (beta*w~ = beta/B 를 보존).
"""

import torch

__all__ = ["apo_weights"]


def apo_weights(l1_err, t, m_t, sign, mag, beta_d: float = 8.0, beta_u: float = 8.0):
    """샘플별 최종 가중치 lambda(x) 와 진단값을 돌려준다.

    l1_err (B,)   |eps - eps_theta|_1 의 샘플별 평균
    t      (B,)   뽑힌 디퓨전 스텝
    m_t    (T,)   t 별 평균 오차 (사전계산, 고정)
    sign   (B,)   +1 desirable · -1 undesirable · 0 보류
    mag    (B,)   |S| — 라벨의 확신도
    """
    # ⚠ 가중치는 importance weight 다. gradient 를 흘리면 "오차를 키워 가중치를 올리는"
    #   역방향 지름길이 생긴다 — 반드시 detach.
    l = l1_err.detach()
    l_tilde = l / m_t.to(l.device)[t]                       # t 정규화
    w = l_tilde / l_tilde.sum().clamp_min(1e-12)            # 배치 정규화 (APO 원문)

    lam_d = 1.0 - torch.exp(-beta_d * w)
    lam_u = torch.exp(-beta_u * w)
    lam_apo = torch.where(sign > 0, lam_d, lam_u)
    lam = mag * lam_apo                                      # 확신도 x 중요도
    lam = torch.where(sign == 0, torch.zeros_like(lam), lam)  # 보류 청크는 학습에서 뺀다

    d = sign > 0
    diag = {
        "w_max_over_mean": (w.max() / w.mean()).item(),
        "lam_d_mean": lam_d[d].mean().item() if d.any() else 0.0,
        "lam_u_mean": lam_u[~d].mean().item() if (~d).any() else 0.0,
        "lam_mean": lam.mean().item(),
    }
    return lam, diag
