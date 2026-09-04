"""세 정책을 make_policy 로 만들고 랜덤 배치로 forward/loss/backward 1스텝.

diffusion 은 LeRobot 구현, act·cfm 은 이전 계열이다. make_policy 가 두 계열을
같은 (policy, preprocessor, postprocessor) 계약으로 내주는지도 여기서 확인된다.
"""
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir

from manibot.policies.factory import make_policy

CFG_DIR = str(Path(__file__).resolve().parents[1] / "src" / "manibot" / "configs")
DEV, B, HW = "cpu", 2, 96


def stub_meta(cfg):
    """dataset_meta 대신 — LeRobot config 는 이미지 shape 만 여기서 읽는다."""
    return SimpleNamespace(features={
        k: {"dtype": "image", "shape": [HW, HW, 3]} for k in cfg.task.image_keys
    })


def make_stats(cfg):
    sd, ad = cfg.task.state_dim, cfg.task.action_dim
    stats = {
        cfg.task.state_key: {"mean": torch.zeros(sd), "std": torch.ones(sd),
                             "min": -torch.ones(sd), "max": torch.ones(sd)},
        cfg.task.action_key: {"mean": torch.zeros(ad), "std": torch.ones(ad),
                              "min": -torch.ones(ad), "max": torch.ones(ad)},
    }
    for k in cfg.task.image_keys:
        stats[k] = {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1),
                    "min": torch.zeros(3, 1, 1), "max": torch.ones(3, 1, 1)}
    return stats


def make_batch(cfg):
    S, P = cfg.policy.obs_horizon, cfg.policy.pred_horizon
    batch = {
        cfg.task.state_key: torch.randn(B, S, cfg.task.state_dim),
        cfg.task.action_key: torch.randn(B, P, cfg.task.action_dim),
        # LeRobot 은 패딩 마스크를 필수로 본다 — ZarrDataset 이 실제로 넣어주는 키다.
        f'{cfg.task.action_key}_is_pad': torch.zeros(B, P, dtype=torch.bool),
    }
    for k in cfg.task.image_keys:
        batch[k] = torch.rand(B, S, 3, HW, HW)
    return batch


for name in ["diffusion", "cfm", "act"]:
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(config_name="default_policy",
                      overrides=[f"policy={name}", "task=piper_cube_stack", f"device={DEV}",
                                 f"resize_shape=[{HW},{HW}]", f"crop_shape=[{HW-8},{HW-8}]"])
    try:
        policy, pre, post = make_policy(cfg, stub_meta(cfg), make_stats(cfg))
        policy = policy.to(DEV)
        n = sum(p.numel() for p in policy.parameters())
        out = policy.forward(pre(make_batch(cfg)))
        loss = out[0] if isinstance(out, tuple) else out
        loss.backward()
        gnorm = sum(p.grad.norm().item() ** 2 for p in policy.parameters() if p.grad is not None) ** 0.5
        src = "LeRobot" if type(policy).__module__.startswith("lerobot") else "manibot"
        print(f"  ✓ {name:10s} {src:8s} params {n/1e6:7.2f}M  loss {loss.item():8.4f}  grad_norm {gnorm:.3f}")
    except Exception as e:
        import traceback
        print(f"  ✗ {name:10s} {type(e).__name__}: {e}")
        traceback.print_exc(limit=4)
