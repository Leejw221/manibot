"""청크 점수 S 를 데이터셋 인덱스별로 미리 계산한다 (학습 전 1회).

왜 미리: c(t) 는 **개입 경계까지의 거리**로 정해지는데, 그 경계가 16프레임 윈도 밖일 수
있어 배치 안에서는 계산할 수 없다. 그리고 **샘플러와 손실이 같은 배열을 봐야** 한다 —
기준을 따로 정의하면 샘플러가 목표한 배치 구성이 손실에서 재현되지 않는다.

윈도 매핑은 데이터셋의 `_get_query_indices` 를 그대로 쓴다 (직접 재현하면 어긋난다).

산출: <out>.npz  — S (N,) · has_intv (N,) bool
사용:
    python -m manibot.scripts.precompute_apo_labels task=square_apo_r1 \
        +gamma=0.95 +out=data/square_apo_r1_zarr/apo_labels.npz
"""

import numpy as np
import hydra
import torch
from lerobot.utils.constants import ACTION

from manibot.utils.dataset_utils import create_dataset, create_dataset_stats
from manibot.utils.intervention_labels import LABEL_INTV, preference, recovery_confidence
from manibot.utils.task_utils import derive_task_meta
from manibot.policies.factory import make_policy


@hydra.main(version_base="1.3", config_path="../configs", config_name="default_policy")
def main(cfg):
    gamma = float(cfg.get("gamma", 0.95))
    k_pre = int(cfg.get("k_pre", 15))
    out = cfg.get("out") or f"{cfg.task.dataset_root}/apo_labels.npz"

    meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, meta)
    policy, _, _ = make_policy(cfg, meta, stats)
    ds = create_dataset(policy, cfg)

    # action_mode 는 학습에 안 쓰는 키라 replay buffer 에 안 실린다 — zarr 에서 직접 읽는다
    import zarr
    root = zarr.open(str(cfg.task.dataset_root), "r")
    ep_idx = np.asarray(root["data"]["episode_index"])
    mode = np.asarray(root["data"]["action_mode"])

    # 프레임별 c(t) 를 에피소드마다 계산
    c = np.zeros(len(mode), dtype=np.float64)
    has_intv_ep = {}
    for e in np.unique(ep_idx):
        sel = ep_idx == e
        has_intv_ep[int(e)] = bool((mode[sel] == LABEL_INTV).any())
        c[sel] = recovery_confidence(mode[sel], gamma, k_pre)

    # 인덱스별 S — 데이터셋이 쓰는 바로 그 윈도로
    N = len(ds)
    S = np.zeros(N, dtype=np.float64)
    has = np.zeros(N, dtype=bool)
    for i in range(N):
        e = int(ep_idx[i])
        qi, _ = ds._get_query_indices(i, e)
        S[i] = c[np.asarray(qi[ACTION])].sum()
        has[i] = has_intv_ep[e]

    sign, mag = preference(S)
    np.savez(out, S=S, has_intv=has, gamma=gamma, k_pre=k_pre)
    print(f"저장: {out}   샘플 {N}  (gamma={gamma}, k_pre={k_pre})")
    print(f"  개입 있는 에피소드의 청크 {has.sum()}  ·  없는 청크 {(~has).sum()}")
    print(f"  D {int((sign>0).sum())}  U {int((sign<0).sum())}  보류 {int((sign==0).sum())}")
    print(f"  |S| 중앙 {np.median(mag[has]):.3f}  90분위 {np.percentile(mag[has],90):.3f}")


if __name__ == "__main__":
    main()
