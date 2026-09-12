"""m[t] — t 별 평균 L1 예측 오차를 pi_ref 로 미리 재서 고정한다 (라운드당 1회).

왜: 가중치 w_i = l_i/sum(l_j) 가 "이 샘플이 어려운가" 를 재야 하는데, 예측 오차의 크기가
t 에 따라 **38배** 다르다. 정규화 없이는 "어느 t 가 뽑혔나" 를 잰다 — t 간 변동이 샘플 간
변동의 4.6배 [측정 2026-09-12].

왜 고정: 모양이 정책에 거의 무관하고(다른 시드 정책과 상관 0.9998), 전체 스케일은 뒤따르는
배치 정규화가 지운다. 갱신하면 초기 불안정 + 되먹임만 생긴다.

표본 크기: 8배치(1,024 샘플)면 모양 오차 1.4% — 정책이 바뀔 때의 변화(8.84%)보다 작다.
사용:
    python -m manibot.scripts.precompute_t_stats task=square_apo_r1 policy=diffusion \
        +ref_checkpoint=outputs/.../step_0000050000 +n_batches=8 +out=.../apo_mt.npy
"""

import hydra
import numpy as np
import torch

from manibot.policies.diffusion_ops import prepare_cond, unet_out
from manibot.utils.checkpoints import load_ema_weights, load_model_weights
from manibot.utils.dataset_utils import create_dataloader, create_dataset, create_dataset_stats
from manibot.policies.factory import make_policy
from manibot.utils.task_utils import derive_task_meta


@hydra.main(version_base="1.3", config_path="../configs", config_name="default_policy")
def main(cfg):
    from lerobot.utils.constants import ACTION

    ckpt = cfg.ref_checkpoint
    n_batches = int(cfg.get("n_batches", 8))
    out = cfg.get("out") or f"{cfg.task.dataset_root}/apo_mt.npy"
    device = cfg.get("device", "cuda")

    meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, meta)
    policy, pre, _ = make_policy(cfg, meta, stats)
    policy = policy.to(device).eval()
    load_model_weights(policy, ckpt, device)
    used = load_ema_weights(policy, ckpt, device)
    print(f"pi_ref = {ckpt}  (EMA {'적용' if used else '없음'})")

    loader = create_dataloader(create_dataset(policy, cfg), cfg, is_training=False)
    T = policy.diffusion.noise_scheduler.config.num_train_timesteps

    acc, n = np.zeros(T), 0
    it = iter(loader)
    with torch.no_grad():
        for _ in range(n_batches):
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in next(it).items()}
            b = pre(b)
            cond, x0 = prepare_cond(policy, b)
            eps = torch.randn_like(x0)
            for k in range(T):
                t = torch.full((x0.shape[0],), k, device=device, dtype=torch.long)
                e, _, _ = unet_out(policy.diffusion, cond, x0, t, eps)
                acc[k] += (e - eps).abs().mean(dim=(1, 2)).sum().item()
            n += x0.shape[0]
    m_t = acc / n
    np.save(out, m_t)
    print(f"저장: {out}   표본 {n} 샘플 x t {T}")
    print(f"  m[t]  t=0 {m_t[0]:.4f}   t=50 {m_t[50]:.4f}   t={T-1} {m_t[-1]:.4f}")
    print(f"  최대/최소 {m_t.max()/m_t.min():.1f}배  (정규화가 걷어낼 t 축의 크기)")


if __name__ == "__main__":
    main()
