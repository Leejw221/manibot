"""공유 백본(VIP) DP + STG 헤드 — 라벨·풀·배치 구성·손실·그래디언트 경로 검증.

기대값은 희민 self-improving-gym 원본(diffusion_policy_stg.py · joint_stg_dataset.py)과 설정에서 정한 의도다.
실데이터(data/square_ph50_zarr)·VIP 가중치(data/vip/model.pt)가 없으면 그 테스트만 건너뛴다.
실행 (manibot 루트에서):
    CUDA_VISIBLE_DEVICES="" ~/miniconda3/envs/manibot/bin/python -m pytest tests/test_vip_stg.py
"""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

from manibot.datasets.joint_stg import (
    JointBatchSampler,
    JointStgDataset,
    stg_pool,
    stg_val_episodes,
    steps_to_go,
)

REPO = Path(__file__).resolve().parents[1]
CFG_DIR = str(REPO / "src" / "manibot" / "configs")
PH50 = REPO / "data" / "square_ph50_zarr"
VIP = REPO / "data" / "vip" / "model.pt"
EDI = {"from": np.array([0, 3]), "to": np.array([3, 7])}


def test_steps_to_go_counts_down_to_zero_per_episode():
    assert steps_to_go(EDI).tolist() == [2, 1, 0, 3, 2, 1, 0]


def test_stg_pool_drops_only_val_episodes():
    val = stg_val_episodes(50, 0.2, 0)
    assert len(val) == 10 and val == stg_val_episodes(50, 0.2, 0)
    assert stg_pool(EDI, {1}).tolist() == [0, 1, 2]
    assert stg_val_episodes(50, 0.0, 0) == set()


def test_batch_sampler_composition():
    bc, stg, n = np.arange(0, 20), np.arange(5, 15), 30
    s = JointBatchSampler(bc, stg, n, bc_batch=4, stg_batch=3, seed=0)
    batches = list(s)
    assert len(batches) == len(s) == 5
    bc_part = [i for b in batches for i in b[:4]]
    assert sorted(bc_part) == bc.tolist()                     # 한 epoch 안에서 BC 는 겹침 없이 전부
    assert all(n + 5 <= i < n + 15 for b in batches for i in b[4:])   # STG 는 풀 + N
    assert list(s) != batches                                 # epoch 마다 다시 섞는다


class _Fake(torch.utils.data.Dataset):
    num_frames, num_episodes = 7, 2

    def __len__(self):
        return 7

    def __getitem__(self, i):
        return {"x": torch.tensor(i)}


def test_joint_dataset_roles_and_labels():
    ds = JointStgDataset(_Fake(), stg_idx=[0, 1, 2], labels=steps_to_go(EDI), num_bins=3)
    a, b, c = ds[1], ds[7 + 1], ds[7 + 4]
    assert a["bc_mask"] and not a["stg_mask"]
    assert not b["bc_mask"] and b["stg_mask"] and int(b["stg_label"]) == 1
    assert not c["stg_mask"]                                  # 풀 밖 프레임은 STG 역할이어도 손실에 안 들어간다
    assert int(ds[7 + 3]["stg_label"]) == 2                   # 라벨 3 -> num_bins-1 로 자른다
    assert len(ds) == 14


def test_normalize_then_resize_equals_resize_then_normalize():
    x = torch.rand(2, 3, 76, 76)
    m, s = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    up = lambda t: F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)  # noqa: E731
    assert torch.allclose(up((x - m) / s), (up(x) - m) / s, atol=1e-5)


@pytest.fixture(scope="module")
def built():
    if not (PH50 / "config.json").exists():
        pytest.skip("data/square_ph50_zarr 없음")
    from manibot.policies.vip_stg import make_vip_stg_policy
    from manibot.utils.dataset_utils import create_dataset_stats
    from manibot.utils.task_utils import derive_task_meta
    with initialize_config_dir(version_base="1.3", config_dir=CFG_DIR):
        cfg = compose("default_policy", overrides=[
            "task=square_ph50", "policy=diffusion_vip_stg", f"base_dir={REPO}", "device=cpu",
            "policy.vip_stg.weight=0.5", f"task.dataset_root={PH50}"])
    meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, meta)
    policy, pre, post = make_vip_stg_policy(cfg, meta, stats)
    return cfg, policy


def _batch(policy, n_bc, n_stg):
    torch.manual_seed(0)
    b = n_bc + n_stg
    batch = {k: torch.randn(b, 2, 3, 84, 84) for k in policy.config.image_features}
    batch["observation.state"] = torch.randn(b, 2, 9)
    batch["action"] = torch.randn(b, 16, 7).clamp(-1, 1)
    batch["action_is_pad"] = torch.zeros(b, 16, dtype=torch.bool)
    batch["bc_mask"] = torch.tensor([True] * n_bc + [False] * n_stg)
    batch["stg_mask"] = ~batch["bc_mask"]
    batch["stg_label"] = torch.tensor([0] * n_bc + list(range(5, 5 + n_stg)))
    return batch


def test_encoder_freeze_bn_and_optimizer_groups(built):
    cfg, policy = built
    conv = policy.diffusion.rgb_encoder.convnet
    frozen = {n.split(".")[0] for n, p in conv.named_parameters() if not p.requires_grad}
    trainable = {n.split(".")[0] for n, p in conv.named_parameters() if p.requires_grad}
    assert frozen == {"conv1", "bn1", "layer1", "layer2"} and trainable == {"layer3", "layer4", "fc"}
    policy.train()
    assert all(not m.training for m in conv.modules() if isinstance(m, torch.nn.BatchNorm2d))
    opt = policy.get_optimizer()
    lrs = [g["lr"] for g in opt.param_groups]
    assert lrs[1] == pytest.approx(lrs[0] * 0.1)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) not in in_opt for p in conv.parameters() if not p.requires_grad)
    assert {id(p) for p in policy.parameters() if p.requires_grad} == in_opt
    # 조건 차원 = (상태 9 + 카메라 2 x 1024) x 관측 2
    assert policy.diffusion.stg_head.in_features == (9 + 2 * 1024) * 2
    assert policy.diffusion.stg_head.out_features == 512 and policy.diffusion.stg_head.bias is None


def test_joint_loss_routes_rows_and_gradients(built):
    cfg, policy = built
    policy.train()
    policy.stg_weight = 0.5
    batch = _batch(policy, 2, 2)
    torch.manual_seed(1)
    loss, out = policy(batch)
    assert float(loss) == pytest.approx(out["bc_loss"] + 0.5 * out["stg_loss"], rel=1e-5)
    # STG 손실은 STG 행만으로 계산된다 — 헤드 출력과 라벨로 다시 계산해 맞춘다
    policy.eval()
    with torch.no_grad():
        sb = policy._stack_images(batch)
        gc = policy.diffusion._prepare_global_conditioning(sb)
        ce = F.cross_entropy(policy.diffusion.stg_head(gc[2:]).float(), batch["stg_label"][2:])
        p = policy.stg_distribution(batch)
    assert p.shape == (4, 512) and torch.allclose(p.sum(-1), torch.ones(4))
    assert float(ce) > 0
    policy.train()
    policy.zero_grad()
    loss, _ = policy(batch)
    loss.backward()
    conv = policy.diffusion.rgb_encoder.convnet
    assert conv.layer1[0].conv1.weight.grad is None and conv.layer3[0].conv1.weight.grad is not None
    assert policy.diffusion.stg_head.weight.grad.abs().sum() > 0
    # stg_weight = 0 이면 STG 헤드에 그래디언트가 안 간다 (warmup 시작)
    policy.zero_grad()
    policy.stg_weight = 0.0
    loss, _ = policy(batch)
    loss.backward()
    g = policy.diffusion.stg_head.weight.grad
    assert g is None or float(g.abs().sum()) == 0.0


def test_vip_weights_load_exactly(built):
    if not VIP.exists():
        pytest.skip("data/vip/model.pt 없음")
    cfg, policy = built
    enc = policy.diffusion.rgb_encoder
    enc.load_vip(VIP)
    ref = torch.load(VIP, map_location="cpu", weights_only=False)["vip"]
    assert torch.equal(enc.convnet.layer4[2].conv3.weight, ref["module.convnet.layer4.2.conv3.weight"])
    assert torch.equal(enc.convnet.fc.weight, ref["module.convnet.fc.weight"])


def test_grad_checkpoint_gives_same_features_and_grads(built):
    cfg, policy = built
    enc = policy.diffusion.rgb_encoder
    enc.train()
    x = torch.randn(2, 3, 84, 84)
    outs = []
    for flag in (False, True):
        enc.grad_checkpoint = flag
        enc.zero_grad()
        torch.manual_seed(3)                                   # 같은 랜덤 크롭
        f = enc(x, 0)
        f.square().mean().backward()
        outs.append((f.detach(), enc.convnet.layer3[0].conv1.weight.grad.clone()))
    enc.grad_checkpoint = False
    assert torch.allclose(outs[0][0], outs[1][0], atol=1e-5)
    assert torch.allclose(outs[0][1], outs[1][1], atol=1e-5)
