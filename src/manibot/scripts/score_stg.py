"""라운드 데이터 채점 — 직전 라운드 모델의 STG(EMA)로 프레임마다 남은 스텝 기댓값 d 를 계산해 저장한다.

노션 계획 페이지 결정 7: 라운드 N 데이터는 라운드 N-1 모델의 STG 로 채점한다 (R1 = base, R2 = R1 모델).
진척률·가중은 학습(train_vip_stg 의 round_weights)이 이 d 로 계산한다 — 양 끝·할인 두 방식이 같은 점수를 쓴다.
d_i = sum_b p_b * b, p = stg_distribution(프레임 i 에서 끝나는 관측 창). 정책은 eval(센터 크롭).

산출: <out>.npz — d (N,) · episode_ends · checkpoint · dataset_root
사용:
    python -m manibot.scripts.score_stg task=square_ph50_vip_r1 policy=diffusion_vip_stg \
        +checkpoint=outputs/square_ph50/diffusion_vip_stg/vipstg-base-noval/checkpoints/step_0000050000 \
        +out=data/square_ph50_vip_r1_stg_base.npz
"""

from pathlib import Path

import hydra
import numpy as np
import torch
from safetensors import safe_open

from manibot.policies.vip_stg import make_vip_stg_policy
from manibot.utils.checkpoints import load_ema_weights, load_model_weights
from manibot.utils.dataset_utils import create_dataset, create_dataset_stats
from manibot.utils.task_utils import derive_task_meta


@hydra.main(version_base="1.3", config_path="../configs", config_name="default_policy")
def main(cfg):
    ckpt, out, device = Path(cfg.checkpoint), Path(cfg.out), cfg.device
    meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, meta)
    policy, pre, _ = make_vip_stg_policy(cfg, meta, stats)
    policy = policy.to(device)
    # load_model_weights 는 strict=False — 키가 안 맞으면 무작위 STG 로 채점하게 된다
    with safe_open(str(ckpt / "model.safetensors"), "pt") as f:
        diff = set(f.keys()) ^ set(policy.state_dict())
    if diff:
        raise ValueError(f"체크포인트 키가 정책과 다르다 ({len(diff)} 개, 예: {sorted(diff)[:3]}): {ckpt}")
    load_model_weights(policy, ckpt, device)          # VIP 특징 통계(버퍼)는 여기서 온다 — EMA 엔 파라미터만 있다
    if not load_ema_weights(policy, ckpt, device):
        raise ValueError(f"EMA 가중치를 싣지 못했다 — 채점은 EMA 로 한다(결정 7): {ckpt}")
    policy.eval()

    ds = create_dataset(policy, cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=cfg.train.num_workers)
    bins = torch.arange(cfg.policy.vip_stg.num_bins, device=device, dtype=torch.float32)
    d = []
    with torch.no_grad():
        for b in loader:
            b = pre(b)
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            d.append((policy.stg_distribution(b) * bins).sum(-1).cpu())
            print(f"  {sum(len(x) for x in d)}/{len(ds)}", end="\r", flush=True)
    d = torch.cat(d).numpy().astype(np.float64)
    if len(d) != len(ds) or not np.isfinite(d).all():
        raise RuntimeError(f"d 가 프레임 수({len(ds)})와 다르거나 유한하지 않다 — 저장하지 않는다")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, d=d, episode_ends=np.asarray(ds.episode_data_index["to"]), checkpoint=str(ckpt),
             dataset_root=str(cfg.task.dataset_root))
    print(f"\n저장: {out}   프레임 {len(d)} · d 범위 [{d.min():.1f}, {d.max():.1f}]  (채점 {ckpt})")


if __name__ == "__main__":
    main()
