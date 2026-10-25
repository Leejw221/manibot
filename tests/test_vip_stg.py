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
    chunk_rates,
    linear_weights,
    round_pools,
    sirius_weights,
    stg_pool,
    stg_val_episodes,
    steps_to_go,
)
from manibot.policies.vip_stg import batch_shares

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


DELTA = np.arange(-1, 15)          # 행동 창 t-1..t+14 (create_dataset 의 LeRobot 규약)


def test_chunk_rates_with_perfect_stg():
    # STG 가 정확하면(d = 남은 스텝) 창이 에피소드 안에선 1, 끝 구간은 (R+1)/16 (결정 4), 첫 샘플은 t-1 패딩으로 15/16
    edi = {"from": np.array([0]), "to": np.array([40])}
    d = steps_to_go(edi).astype(float)
    two = chunk_rates(d, edi, DELTA, "two", 0.9)
    R = 39 - np.arange(40)
    assert two[0] == pytest.approx(15 / 16)
    assert np.allclose(two[1:25], 1.0)                        # R >= 15
    assert np.allclose(two[R <= 14], (R[R <= 14] + 1) / 16)
    assert np.allclose(chunk_rates(d, edi, DELTA, "disc", 1.0), two)   # gamma = 1 이면 두 방식이 같다


def test_chunk_rates_discount_weights_each_step():
    # 창 안 k 번째 행동에서만 d 가 3 떨어지면 two = 3/16, disc = 0.9^k * 3 / sum 0.9^j
    edi = {"from": np.array([0]), "to": np.array([60])}
    d = np.where(np.arange(60) <= 25, 50.0, 47.0)             # 프레임 25 -> 26 행동에서 3 감소
    t = 20                                                    # 창 첫 행동 = 프레임 19 -> k = 25 - 19 = 6
    g = 0.9 ** np.arange(16)
    assert chunk_rates(d, edi, DELTA, "two", 0.9)[t] == pytest.approx(3 / 16)
    assert chunk_rates(d, edi, DELTA, "disc", 0.9)[t] == pytest.approx(g[6] * 3 / g.sum())
    # 에피소드 경계를 넘어 다음 에피소드의 d 를 읽지 않는다
    edi2 = {"from": np.array([0, 30]), "to": np.array([30, 60])}
    assert chunk_rates(d, edi2, DELTA, "two", 0.9)[29] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        chunk_rates(d, edi, DELTA, "both", 0.9)


def test_round_pools_rules():
    # 시연 1 · 개입 에피소드(robot 3 -> 개입 3) · 무개입 robot 1
    edi = {"from": np.array([0, 4, 10]), "to": np.array([4, 10, 14])}
    mode = np.array([0] * 4 + [0, 0, 0, 1, 1, 1] + [0] * 4)
    keep, floor = round_pools(edi, mode, n_demo=1)
    assert keep.tolist() == [1] * 4 + [0, 0, 0, 1, 1, 1] + [1] * 4
    assert floor.tolist() == [1] * 4 + [0, 0, 0, 1, 1, 1] + [0] * 4
    with pytest.raises(ValueError):                           # 개입 뒤 robot 복귀 (결정 6: 멈춘다)
        round_pools(edi, np.array([0] * 4 + [0, 1, 1, 0, 1, 1] + [0] * 4), n_demo=1)
    with pytest.raises(ValueError):                           # 시연 수를 잘못 주면 개입 에피소드가 시연이 된다
        round_pools(edi, mode, n_demo=2)
    with pytest.raises(ValueError):                           # 개입 직전 라벨(2) 등 규칙 없는 라벨
        round_pools(edi, np.where(np.arange(14) == 6, 2, mode), n_demo=1)


def test_batch_shares_floor_without_renormalizing():
    share, info = batch_shares(torch.tensor([2.0, 1.0, 0.5, -1.0]), torch.tensor([True, False, True, False]))
    # v = [2, 1, 0.5, 0] / 3.5 -> 시연·개입 몫 하한 1/4, robot 음수 -> 0, 다시 나누지 않는다
    assert share.tolist() == pytest.approx([2 / 3.5, 1 / 3.5, 0.25, 0.0])
    assert info["share_sum"] == pytest.approx(2 / 3.5 + 1 / 3.5 + 0.25)
    assert info["floor_hit"] == pytest.approx(0.5)
    s, _ = batch_shares(torch.full((8,), 0.7), torch.ones(8, dtype=torch.bool))
    assert torch.allclose(s, torch.full((8,), 1 / 8))          # 모두 같으면 가중 없는 평균과 같다


def test_batch_shares_equal_user_rule_when_mean_rate_is_one():
    # 배치 평균 진척률이 1 이면(합 = B) 몫 x B 가 사용자 규칙 그대로: 시연·개입 max(1, Δ) · robot Δ (10-22)
    rate = torch.tensor([2.5, 0.5, 0.6, 0.4])                 # 합 4 = B
    floor = torch.tensor([True, True, False, False])
    share, _ = batch_shares(rate, floor)
    assert (share * 4).tolist() == pytest.approx([2.5, 1.0, 0.6, 0.4])


def test_linear_weights_rule():
    # 시연 1 고정(Δ 무관) · 개입 max(1,Δ) · robot max(0,Δ) · 제외 0 (사용자 10-22)
    rate = np.array([0.4, 2.5, 0.4, 2.5, -0.7, 0.6, -0.3, 3.0])
    kind = np.array([0, 0, 1, 1, 1, 2, 2, -1])
    assert linear_weights(rate, kind).tolist() == pytest.approx([1, 1, 1, 2.5, 1, 0.6, 0, 0])


def test_sirius_weights_rule():
    # w = P*/P: 시연 1 · 개입 0.5/P(intv) · robot (0.5 - P(demo))/P(robot) · preintv 0 (논문 §IV-D) · 제외 0 (사용자 10-24)
    kind = np.array([-1, 0, 0, 0, 1, 2, 2, 2, 2, 2])          # 풀 9 = 시연 3 · 개입 1 · robot 5
    assert sirius_weights(kind).tolist() == pytest.approx([0, 1, 1, 1, 4.5, 0.3, 0.3, 0.3, 0.3, 0.3])
    # 시연이 풀의 절반을 넘으면 robot 몫이 음수 -> 0 (R1 풀의 실제 상황)
    assert sirius_weights(np.array([0, 0, 0, 1, 2])).tolist() == pytest.approx([1, 1, 1, 2.5, 0])


def test_joint_dataset_carries_round_rate():
    rate, floor = np.linspace(0, 1, 7), np.arange(7) % 2 == 0
    ds = JointStgDataset(_Fake(), stg_idx=[0, 1], labels=steps_to_go(EDI), num_bins=8, bc_rate=rate, bc_floor=floor)
    item = ds[3]
    assert float(item["bc_rate"]) == pytest.approx(0.5) and not bool(item["bc_floor"])
    assert "bc_rate" not in JointStgDataset(_Fake(), [0], steps_to_go(EDI), 8)[3]   # base 는 그대로
    with pytest.raises(ValueError):
        JointStgDataset(_Fake(), [0], steps_to_go(EDI), 8, bc_rate=np.full(7, np.nan), bc_floor=floor)


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


def test_round_loss_reduces_to_unweighted_when_shares_are_uniform(built):
    # 진척률이 모두 같고 전부 하한 대상이면 몫 = 1/B — 가중 경로의 손실이 기존 경로와 같아야 한다(같은 노이즈)
    cfg, policy = built
    policy.train()
    policy.stg_weight = 0.5
    batch = _batch(policy, 3, 2)
    torch.manual_seed(1)
    base_loss, base_out = policy(batch)
    batch["bc_rate"] = torch.full((5,), 0.3)
    batch["bc_floor"] = torch.ones(5, dtype=torch.bool)
    torch.manual_seed(1)
    loss, out = policy(batch)
    assert float(loss) == pytest.approx(float(base_loss), rel=1e-5)
    assert out["share_sum"] == pytest.approx(1.0) and "share_sum" not in base_out
    # robot 하나(floor=False)의 진척률이 0 이면 그 샘플은 손실에 안 들어간다
    batch["bc_rate"] = torch.tensor([1.0, 1.0, 0.0, 0.3, 0.3])
    batch["bc_floor"] = torch.tensor([True, True, False, True, True])
    torch.manual_seed(1)
    _, out = policy(batch)
    assert out["share_sum"] == pytest.approx(1.0) and out["floor_hit"] == 0.0


def test_linear_weighting_loss_is_batch_mean_of_weighted_samples(built):
    # weighting=linear: BC 손실 = (1/B) Σ w_i l_i — 가중이 모두 1 이면 기존 경로와 같고, 배치 합으로 나누지 않는다
    cfg, policy = built
    policy.train()
    policy.stg_weight = 0.5
    batch = _batch(policy, 3, 2)
    torch.manual_seed(1)
    base_loss, _ = policy(batch)
    batch["bc_weight"] = torch.ones(5)
    torch.manual_seed(1)
    loss, out = policy(batch)
    assert float(loss) == pytest.approx(float(base_loss), rel=1e-5)
    assert out["w_mean"] == pytest.approx(1.0)
    # 샘플별 손실을 따로 구해 손으로 (1/3)(2·l0 + 0·l1 + 1·l2) 와 맞춘다 (같은 시드 = 같은 노이즈·timestep)
    batch["bc_weight"] = torch.tensor([2.0, 0.0, 1.0, 9.0, 9.0])     # STG 행(뒤 2개)의 값은 쓰이지 않아야 한다
    torch.manual_seed(1)
    _, out = policy(batch)
    torch.manual_seed(1)
    sb = policy._stack_images(batch)
    gc = policy.diffusion._prepare_global_conditioning(sb)
    bc = batch["bc_mask"]
    per = policy.diffusion.bc_loss(gc[bc], sb["action"][bc], sb["action_is_pad"][bc], per_sample=True)
    assert out["bc_loss"] == pytest.approx(float((2 * per[0] + 0 * per[1] + 1 * per[2]) / 3), rel=1e-5)
    assert out["w_mean"] == pytest.approx(1.0) and out["w_max_share"] == pytest.approx(2 / 3)


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
