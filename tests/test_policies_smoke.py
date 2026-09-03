"""세 정책(diffusion·cfm·act)을 설정에서 만들고 랜덤 배치로 forward/loss 1스텝."""
import torch
from pathlib import Path
from hydra import compose, initialize_config_dir

CFG_DIR = (Path(__file__).resolve().parents[1] / "src" / "manibot" / "configs")
DEV = "cpu"
B = 2


def make_stats(cfg):
    """meanstd 와 minmax 를 둘 다 채운다 — 정책마다 쓰는 쪽이 다르다."""
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
    S = cfg.policy.obs_horizon
    P = cfg.policy.pred_horizon
    H, W = tuple(cfg.resize_shape)
    batch = {
        cfg.task.state_key: torch.randn(B, S, cfg.task.state_dim, device=DEV),
        cfg.task.action_key: torch.randn(B, P, cfg.task.action_dim, device=DEV),
    }
    for k in cfg.task.image_keys:
        batch[k] = torch.rand(B, S, 3, H, W, device=DEV)
    return batch


for name in ["diffusion", "cfm", "act"]:
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="default_policy",
                      overrides=[f"policy={name}", "task=piper_cube_stack", f"device={DEV}"])
    try:
        from manibot.policies.factory import registry
        cls = registry.get_policy(name)
        policy = cls(cfg, make_stats(cfg)).to(DEV)
        n = sum(p.numel() for p in policy.parameters())
        out = policy.compute_loss(make_batch(cfg))
        loss = out[0] if isinstance(out, tuple) else out
        loss.backward()
        gnorm = sum(p.grad.norm().item() ** 2 for p in policy.parameters() if p.grad is not None) ** 0.5
        print(f"  ✓ {name:10s} params {n/1e6:7.2f}M   loss {loss.item():.4f}   grad_norm {gnorm:.3f}")
    except Exception as e:
        import traceback
        print(f"  ✗ {name:10s} {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
