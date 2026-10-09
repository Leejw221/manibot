"""관점 C — 샘플 가중 손실(SampleWeightLoss)과 학습 통합(train.py · dataset_utils) 검증.

기대값은 현재 코드가 아니라 **의도**다 (검토 요청의 번호):
  (1) 시연 w=1   (3) PI 판정이 가중 파일과 학습 sampler 에서 같고 drop 샘플은 batch 에 절대 안 나온다
  (4) prev 범위는 그대로   (6) 손실 = sum(w * 청크 MSE) / sum(w) — w 가 큰 샘플이 비례해 더 반영된다
  (7) 기존 sirius/bc 경로는 동작이 안 바뀐다
  (8) 잘못된 입력은 조용히 넘어가지 않고 분명한 에러 (메시지가 있는 AssertionError · ValueError 등)

합성 데이터셋·출력은 pytest tmp_path 아래에만 만든다. 실데이터(data/square_ph50_r1_zarr · r2)는
읽기만 하고, 없으면 그 테스트만 건너뛴다.
실행 (manibot 루트에서):
    CUDA_VISIBLE_DEVICES="" ~/miniconda3/envs/manibot/bin/python -m pytest tests/test_sample_weight_train.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
from hydra import compose, initialize_config_dir

from manibot.datasets.replay_buffer import ReplayBuffer
from manibot.losses.sample_weight import SampleWeightLoss
from manibot.losses.sirius import LABELS, SiriusLoss, action_windows, frame_labels
from manibot.policies.diffusion_ops import prepare_cond
from manibot.policies.factory import make_policy
from manibot.scripts.train import PolicyTrainer, _build_finetune_loss
from manibot.utils.dataset_utils import (
    EpisodeAwareSampler,
    create_dataloader,
    create_dataset,
    create_dataset_stats,
)
from manibot.utils.task_utils import derive_task_meta

REPO = Path(__file__).resolve().parents[1]
CFG_DIR = str(REPO / "src" / "manibot" / "configs")
R1 = REPO / "data" / "square_ph50_r1_zarr"
R2 = REPO / "data" / "square_ph50_r2_zarr"

S_DIM, A_DIM, HW, FPS = 7, 7, 32, 20
H, OBS, ACT = 16, 2, 8           # policy=diffusion 의 pred / obs / action horizon
DEFAULT_DROP_LAST = H - ACT - OBS + 1
N_DEMO = 2
K = 4                            # build_sample_weights 의 기본 drop_preintv_min
JUDGE = "t..t+14,no_pad"         # 가중·점수 파일의 판정 규약 (sample_weight 의 PI 판정 칸)


# ── 합성 데이터 ────────────────────────────────────────────────────────────

def synth_modes():
    """에피소드별 action_mode (0 정책 · 1 개입). 앞 N_DEMO 개는 시연."""
    z, o = np.zeros, np.ones
    return [
        z(30), z(32),                                   # 시연
        np.r_[z(30), o(20)],                            # 개입 1번 — 직전 15칸이 preintv
        np.r_[z(10), o(10), z(15), o(10)],              # 개입 2번 — 첫 개입 앞은 10칸뿐
        z(28),                                          # 개입 없는 롤아웃
    ]


def build_dataset(root, modes, seed=0):
    """convert 가 만드는 모양(zarr + config.json)에 action_mode 를 더한 작은 데이터셋."""
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer.create_from_path(str(root), mode="a")
    states, actions = [], []
    for i, m in enumerate(modes):
        T = len(m)
        st = rng.standard_normal((T, S_DIM)).astype(np.float32)
        ac = rng.standard_normal((T, A_DIM)).astype(np.float32)
        buf.add_episode({
            "observation.state": st,
            "action": ac,
            "action_mode": np.asarray(m, dtype=np.int64),
            "observation.images.main": rng.integers(0, 255, (T, HW, HW, 3), dtype=np.uint8),
            "observation.images.wrist": rng.integers(0, 255, (T, HW, HW, 3), dtype=np.uint8),
            "episode_index": np.full(T, i, dtype=np.int64),
            "timestamp": np.arange(T, dtype=np.float32) / FPS,
        })
        states.append(st), actions.append(ac)

    def stat(a):
        return {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
                "min": a.min(0).tolist(), "max": a.max(0).tolist()}

    imgs = ["observation.images.main", "observation.images.wrist"]
    json.dump({
        "repo_id": None,
        "stats": {"observation.state": stat(np.concatenate(states)),
                  "action": stat(np.concatenate(actions))},
        "num_frames": int(buf.n_steps), "num_episodes": len(modes),
        "features": {"observation.state": {"dtype": "float32", "shape": [S_DIM]},
                     "action": {"dtype": "float32", "shape": [A_DIM]},
                     **{k: {"dtype": "image", "shape": [HW, HW, 3]} for k in imgs}},
        "camera_keys": imgs, "video_keys": [], "image_keys": imgs,
        "fps": FPS, "tasks": {0: "fake"},
    }, open(Path(root) / "config.json", "w"), indent=2)
    return root


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("synth") / "ds", synth_modes())


# ── 구현과 독립인 기준 (프레임 라벨 · 행동 창 · PI 판정) ──────────────────

def ref_windows(lengths):
    """(N, H) 샘플 i 의 행동 창 — 관측 창 시작(i-OBS+1)부터 H 칸, 에피소드 안으로 clamp."""
    rows, s = [], 0
    for L in lengths:
        for i in range(s, s + L):
            rows.append([min(max(i + d, s), s + L - 1) for d in range(1 - OBS, 1 - OBS + H)])
        s += L
    return np.asarray(rows, dtype=np.int64)


def ref_frame_labels(modes, n_demo):
    """시연 -1 · 정책 0 · 개입 1 · 각 개입 시작 직전의 정책 프레임 최대 15개 -10."""
    out = []
    for e, m in enumerate(modes):
        m = np.asarray(m, dtype=np.int64)
        if e < n_demo:
            out.append(np.full(len(m), LABELS["demo"]))
            continue
        lab = m.copy()
        for o in range(1, len(m)):
            if m[o] == 1 and m[o - 1] != 1:
                for f in range(max(0, o - 15), o):
                    if m[f] == 0:
                        lab[f] = LABELS["preintv"]
        out.append(lab)
    return np.concatenate(out)


def ref_n_pre(modes, n_demo):
    """sirius 의 PI 판정: 행동 창 16칸(고정된 복사 칸 포함)의 preintv 칸 수."""
    lab = ref_frame_labels(modes, n_demo)[ref_windows([len(m) for m in modes])]
    return (lab == LABELS["preintv"]).sum(1)


def ref_judge(lengths):
    """(N, H) sample_weight 의 판정 칸 — 창 칸 d 는 프레임 i+d (d = 1-OBS .. H-OBS). 프레임 i 부터(d >= 0)이고
    에피소드 안인 칸 [사용자 결정 2026-10-09: t-1 칸·패딩 칸은 판정에 안 쓴다]."""
    rows, s = [], 0
    for L in lengths:
        for i in range(s, s + L):
            rows.append([d >= 0 and i + d < s + L for d in range(1 - OBS, 1 - OBS + H)])
        s += L
    return np.asarray(rows, dtype=bool)


def ref_n_pre_judge(modes, n_demo):
    """sample_weight 의 PI 판정: 판정 칸(프레임 t..t+14 중 에피소드 안)의 preintv 칸 수."""
    lengths = [len(m) for m in modes]
    lab = ref_frame_labels(modes, n_demo)[ref_windows(lengths)]
    return ((lab == LABELS["preintv"]) & ref_judge(lengths)).sum(1)


def ref_episode_index(modes):
    return np.repeat(np.arange(len(modes)), [len(m) for m in modes])


def write_weights(path, modes, n_demo=N_DEMO, k=K, w=None, drop=None, seed=0, root="", judge=JUDGE):
    """build_sample_weights 와 같은 규칙·키로 가중 파일을 쓴다 (기준은 위의 독립 구현).

    시연 w=1 · 새 PI 샘플 w=0·drop · 나머지 무작위 양수를 평균 1 로.  w/drop 을 주면 그대로 쓴다.
    judge=None 이면 판정 규약 키를 쓰지 않는다 (2026-10-09 이전 파일).
    """
    ep = ref_episode_index(modes)
    pi = (ref_n_pre_judge(modes, n_demo) >= k) & (ep >= n_demo)
    if w is None:
        rng = np.random.default_rng(seed)
        w = np.ones(len(ep), dtype=np.float32)
        ok = (ep >= n_demo) & ~pi
        r = rng.uniform(0.4, 3.0, ok.sum())
        w[ok] = r / r.mean()
        w[pi] = 0.0
    if drop is None:
        drop = pi
    w, drop = np.asarray(w, dtype=np.float32), np.asarray(drop, dtype=bool)
    new = ep >= n_demo
    extra = {} if judge is None else {"judge": judge}
    np.savez(path, w=w, drop=drop, episode_index=ep, n_prev=0, n_new=int(new.sum()),
             n_new_pi=int((new & drop).sum()), w_new_quantiles=np.zeros(7), clip_lo_frac=0.0,
             clip_hi_frac=0.0, lo=0.4, hi=3.0, drop_preintv_min=k, n_demo_episodes=n_demo,
             scores="", prev="", dataset_root=str(root), **extra)
    return w, drop


# ── 설정 · 정책 · 학습기 ────────────────────────────────────────────────────

def synth_overrides(root, out, over=None):
    """합성 데이터 · 작은 정책 · sample_weight 기본 설정. over 의 값이 None 이면 그 키를 뺀다."""
    ov = {
        "task": "piper_cube_stack", "policy": "diffusion", "device": "cpu",
        "task.dataset_root": str(root), "task.dataset_repo_id": "null", "task.fps": FPS,
        "resize_shape": f"[{HW},{HW}]", "crop_shape": f"[{HW - 4},{HW - 4}]",
        "policy.unet.down_dims": "[32,64]",
        "train.batch_size": 4, "train.num_workers": 0, "train.use_amp": "false",
        "train.log_freq": 1, "wandb.enable": "false",
        "base_dir": str(out), "output_dir": str(Path(out) / "run"),
        "finetune.enabled": "true", "finetune.loss": "sample_weight", "finetune.balanced": "null",
        "finetune.n_demo_episodes": N_DEMO, "finetune.sirius_drop_preintv_min": K,
        "+policy.drop_n_last_frames": 0,
    }
    ov.update(over or {})
    return [f"{k}={v}" for k, v in ov.items() if v is not None]


def compose_cfg(overrides):
    with initialize_config_dir(config_dir=CFG_DIR, version_base="1.3"):
        return compose(config_name="default_policy", overrides=overrides)


def build_policy_dataset(cfg, episodes=None):
    meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, meta)
    torch.manual_seed(0)
    policy, pre, _ = make_policy(cfg, meta, stats)
    return policy, pre, create_dataset(policy, cfg, episodes)


def make_trainer(cfg, policy, pre, ds):
    """main() 과 같은 순서 — 샘플러를 만든 뒤 PolicyTrainer 가 손실을 만들고 PI 샘플을 뺀다."""
    dl = create_dataloader(ds, cfg, is_training=True)
    return PolicyTrainer(cfg, policy, device="cpu", train_dataloader=dl, preprocessor=pre)


def setup_synth(root, out, over=None):
    cfg = compose_cfg(synth_overrides(root, out, over))
    policy, pre, ds = build_policy_dataset(cfg)
    return cfg, policy, pre, ds


def raises_clear(fn, types=(AssertionError, ValueError)):
    """분명한 에러 = 정해진 종류 + 비어 있지 않은 메시지 (맨 assert 의 빈 메시지는 분명하지 않다)."""
    with pytest.raises(types) as e:
        fn()
    assert str(e.value).strip(), f"에러 메시지가 비어 있다: {type(e.value).__name__}"
    return e.value


def batch_of(ds, pre, idx):
    items = [ds[int(i)] for i in idx]
    b = {k: torch.stack([it[k] for it in items]) for k in items[0]
         if isinstance(items[0][k], torch.Tensor)}
    di = b["dataset_index"]
    b = pre(b)
    b["dataset_index"] = di
    return b


def hand_chunk_mse(policy, batch, seed):
    """샘플별 청크 MSE (B,) — 손실과 같은 난수 순서(조건 -> t -> eps)로 직접 계산한다."""
    torch.manual_seed(seed)
    m = policy.diffusion
    cond, x0 = prepare_cond(policy, batch)
    T = m.noise_scheduler.config.num_train_timesteps
    t = torch.randint(0, T, (x0.shape[0],), dtype=torch.long)
    eps = torch.randn_like(x0)
    e = m.unet(m.noise_scheduler.add_noise(x0, eps, t), t, global_cond=cond)
    return ((eps - e) ** 2).mean(dim=(1, 2))


def flat_grad(policy):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                      for p in policy.parameters()])


def rel_err(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def new_ok_indices(w, drop, n_demo_frames):
    return np.flatnonzero((np.arange(len(w)) >= n_demo_frames) & ~drop)


def run_module(mod, overrides):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-m", mod, *overrides], capture_output=True, text=True,
                       cwd=str(REPO), env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"{mod} 실패 (code {r.returncode}):\n{out[-4000:]}"
    return out


def logged_losses(out):
    return [float(v) for v in re.findall(r"\bloss:(\S+)", out)]


def fake_scores(path, ep, index, seed):
    rng = np.random.default_rng(seed)
    l1 = rng.gamma(2.0, 0.05, (len(index), 2))
    np.savez(path, index=np.asarray(index), episode_index=ep[index], l1=l1,
             l1_mean=l1.mean(1), l1_min=l1.min(1), n_valid_slots=np.full(len(index), H), judge=JUDGE)


N_DEMO_FRAMES = sum(len(m) for m in synth_modes()[:N_DEMO])


# ── (6) 손실 식 ────────────────────────────────────────────────────────────

def test_loss_equals_weighted_mean_of_chunk_mse(synth_root, tmp_path):
    """loss == sum(w_i * 청크MSE_i) / sum(w_i) 를 손계산과 대조. 배치는 섞인 비연속 인덱스."""
    w, drop = write_weights(tmp_path / "w.npz", synth_modes())
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path,
                                       {"finetune.sample_weights": tmp_path / "w.npz"})
    loss_fn = _build_finetune_loss(cfg, policy, ds)
    assert isinstance(loss_fn, SampleWeightLoss)
    ok = new_ok_indices(w, drop, N_DEMO_FRAMES)
    idx = np.array([ok[17], 3, ok[2], ok[40], ok[9]])        # 시연 하나 + 새 비-PI 넷
    policy.eval()
    b = batch_of(ds, pre, idx)
    with torch.no_grad():
        torch.manual_seed(11)
        loss, out = loss_fn(policy, b)
        lbar = hand_chunk_mse(policy, b, 11).double()
    wi = torch.as_tensor(w[idx], dtype=torch.float64)
    expect = float((wi * lbar).sum() / wi.sum())
    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(expect, rel=1e-5)
    assert out["w_mean"] == pytest.approx(float(w[idx].mean()), rel=1e-6)


def test_loss_is_invariant_to_global_weight_scale(synth_root, tmp_path):
    """sum(w*l)/sum(w) 이므로 모든 w 에 같은 상수를 곱해도 손실이 같다 (상대 가중만 의미가 있다)."""
    w, drop = write_weights(tmp_path / "w1.npz", synth_modes())
    write_weights(tmp_path / "w3.npz", synth_modes(), w=w * 3.0, drop=drop)
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path,
                                       {"finetune.sample_weights": tmp_path / "w1.npz"})
    l1 = _build_finetune_loss(cfg, policy, ds)
    cfg.finetune.sample_weights = str(tmp_path / "w3.npz")
    l3 = _build_finetune_loss(cfg, policy, ds)
    ok = new_ok_indices(w, drop, N_DEMO_FRAMES)
    policy.eval()
    b = batch_of(ds, pre, ok[[5, 30, 1, 22]])
    with torch.no_grad():
        torch.manual_seed(3)
        a, _ = l1(policy, b)
        torch.manual_seed(3)
        c, _ = l3(policy, b)
    assert float(a) == pytest.approx(float(c), rel=1e-6)


def test_doubling_a_weight_doubles_its_gradient_share(synth_root, tmp_path):
    """grad L = sum_i w_i g_i / sum_i w_i (g_i = 샘플 i 청크 MSE 의 기울기, 손계산).
    샘플 a 의 w 를 2배로 하면 a 의 계수/b 의 계수 비가 정확히 2배가 된다."""
    w, drop = write_weights(tmp_path / "w.npz", synth_modes())
    ok = new_ok_indices(w, drop, N_DEMO_FRAMES)
    ia, ib = ok[4], ok[33]
    w2 = w.copy()
    w2[ia] *= 2.0
    write_weights(tmp_path / "w2.npz", synth_modes(), w=w2, drop=drop)
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path,
                                       {"finetune.sample_weights": tmp_path / "w.npz"})
    lf1 = _build_finetune_loss(cfg, policy, ds)
    cfg.finetune.sample_weights = str(tmp_path / "w2.npz")
    lf2 = _build_finetune_loss(cfg, policy, ds)
    policy.eval()
    b = batch_of(ds, pre, [ia, ib])

    g = []
    for i in range(2):                                   # 샘플별 기울기 (같은 t·eps)
        policy.zero_grad(set_to_none=True)
        hand_chunk_mse(policy, b, 5)[i].backward()
        g.append(flat_grad(policy).double())
    for lf, ww in ((lf1, w), (lf2, w2)):
        policy.zero_grad(set_to_none=True)
        torch.manual_seed(5)
        loss, _ = lf(policy, b)
        loss.backward()
        got = flat_grad(policy).double()
        wa, wb = float(ww[ia]), float(ww[ib])
        expect = (wa * g[0] + wb * g[1]) / (wa + wb)
        assert rel_err(got, expect) < 1e-4, (wa, wb, rel_err(got, expect))


def test_loss_matches_sirius_normalized_chunk_weights(synth_root, tmp_path):
    """같은 샘플 가중이면 SiriusLoss(majority · normalize) 와 손실이 같다 (sample_weight.py 의 설계 문장)."""
    modes = synth_modes()
    fl = ref_frame_labels(modes, N_DEMO)
    act = ref_windows([len(m) for m in modes])
    sl = SiriusLoss(fl, act, chunk_label="majority", normalize=True)
    _, drop = write_weights(tmp_path / "tmp.npz", modes)
    w = sl.w[:, 0].copy()
    w[drop] = 0.0
    assert (w[~drop] > 0).all()
    write_weights(tmp_path / "w.npz", modes, w=w, drop=drop)
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path,
                                       {"finetune.sample_weights": tmp_path / "w.npz"})
    swl = _build_finetune_loss(cfg, policy, ds)
    ok = np.flatnonzero(~drop)
    cls = sl.cls[ok]
    idx = [ok[cls == LABELS[c]][0] for c in ("demo", "robot", "intv")] + [ok[-1]]
    policy.eval()
    b = batch_of(ds, pre, idx)
    with torch.no_grad():
        torch.manual_seed(21)
        a, _ = swl(policy, b)
        torch.manual_seed(21)
        c, _ = sl(policy, b)
    assert float(a) == pytest.approx(float(c), rel=1e-6)


def test_all_zero_weight_batch_is_error_or_defined_zero(synth_root, tmp_path):
    """배치의 w 가 전부 0 — 올바른 파일에선 drop 만 w=0 이라 생길 수 없다(다음 테스트가 확인).
    명세 밖 경계이므로 기대는 둘 중 하나: (a) 비-drop 샘플에 w=0 인 파일을 분명한 에러로 거절,
    (b) 받아들인다면 손실이 정의된 값 0 이고 기울기가 유한(0). 최소 기대 = 조용한 NaN 이 없다."""
    modes = synth_modes()
    w, drop = write_weights(tmp_path / "tmp.npz", modes)
    ok = new_ok_indices(w, drop, N_DEMO_FRAMES)
    zero = ok[[0, 7, 14, 21]]
    w = w.copy()
    w[zero] = 0.0
    write_weights(tmp_path / "w.npz", modes, w=w, drop=drop)
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path,
                                       {"finetune.sample_weights": tmp_path / "w.npz"})
    try:
        loss_fn = _build_finetune_loss(cfg, policy, ds)
    except (AssertionError, ValueError) as e:
        assert str(e).strip()
        return
    policy.eval()
    b = batch_of(ds, pre, zero)
    policy.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    loss, out = loss_fn(policy, b)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() == 0.0
    assert all(np.isfinite(v) for v in out.values() if isinstance(v, float))
    assert torch.isfinite(flat_grad(policy)).all()


def test_every_sampled_index_has_positive_weight(synth_root, tmp_path):
    """올바른 파일 + PI 제외 샘플러에서는 뽑힐 수 있는 모든 샘플의 w > 0 — 전부 0 인 배치가 불가능하다."""
    w, drop = write_weights(tmp_path / "w.npz", synth_modes())
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path,
                                       {"finetune.sample_weights": tmp_path / "w.npz"})
    tr = make_trainer(cfg, policy, pre, ds)
    ind = np.asarray(tr.train_dataloader.sampler.indices)
    assert len(ind) == len(w) - drop.sum()
    assert (w[ind] > 0).all()


@pytest.mark.parametrize("bad", ["nan", "inf", "negative"])
def test_nonfinite_or_negative_weight_file_is_rejected(synth_root, tmp_path, bad):
    """비-drop 샘플의 w 가 NaN · inf · 음수인 파일 — 손실이 조용히 NaN/inf 가 되거나 부호가 뒤집힌다.
    기대: 학습 시작 전(손실·샘플러 구성 중) 분명한 에러."""
    modes = synth_modes()
    w, drop = write_weights(tmp_path / "tmp.npz", modes)
    i = new_ok_indices(w, drop, N_DEMO_FRAMES)[3]
    w = w.copy()
    w[i] = {"nan": np.nan, "inf": np.inf, "negative": -0.5}[bad]
    write_weights(tmp_path / "w.npz", modes, w=w, drop=drop)
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path,
                                       {"finetune.sample_weights": tmp_path / "w.npz"})
    raises_clear(lambda: make_trainer(cfg, policy, pre, ds))


# ── (3) PI 제외 · 샘플러 · dataset_index ──────────────────────────────────

@pytest.mark.parametrize("workers", [0, 2])
def test_drop_samples_never_appear_over_two_epochs(synth_root, tmp_path, workers):
    """학습 dataloader 를 두 에폭 돌려 모은 dataset_index: drop 은 0 번, 나머지는 에폭마다 정확히 1 번.
    실제 학습은 워커(persistent)를 쓰므로 샘플러를 DataLoader 생성 뒤에 고치는 방식이 워커에서도 먹는지 본다."""
    w, drop = write_weights(tmp_path / "w.npz", synth_modes())
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path, {
        "finetune.sample_weights": tmp_path / "w.npz", "train.num_workers": workers})
    tr = make_trainer(cfg, policy, pre, ds)
    assert drop.sum() > 0
    keep = np.flatnonzero(~drop)
    orders = []
    for _ in range(2):
        seen = np.concatenate([b["dataset_index"].numpy() for b in tr.train_dataloader])
        assert not drop[seen].any(), f"drop 샘플이 batch 에 나왔다: {seen[drop[seen]][:10]}"
        assert np.array_equal(np.sort(seen), keep)
        orders.append(seen)
    assert not np.array_equal(orders[0], np.sort(orders[0])), "학습 샘플러가 섞지 않는다"


def test_trainer_excluded_set_equals_file_drop_and_reference(synth_root, tmp_path):
    """학습이 빼는 샘플 = 가중 파일의 drop = 독립 구현의 (판정 칸 preintv 칸 >= k) & 새 프레임."""
    w, drop = write_weights(tmp_path / "w.npz", synth_modes())
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path,
                                       {"finetune.sample_weights": tmp_path / "w.npz"})
    tr = make_trainer(cfg, policy, pre, ds)
    ind = np.asarray(tr.train_dataloader.sampler.indices)
    excluded = np.setdiff1d(np.arange(len(ds)), ind)
    assert np.array_equal(excluded, np.flatnonzero(drop))
    assert np.array_equal(tr.loss_fn.drop, drop)
    ep = ref_episode_index(synth_modes())
    assert np.array_equal(excluded, np.flatnonzero((ref_n_pre_judge(synth_modes(), N_DEMO) >= K) & (ep >= N_DEMO)))


def test_train_step_picks_weight_by_dataset_index_from_shuffled_batch(synth_root, tmp_path):
    """섞인 학습 batch 로 train_step — 전처리기가 dataset_index 를 떨어뜨려도 w 는 그 인덱스로 집는다."""
    w, drop = write_weights(tmp_path / "w.npz", synth_modes(), seed=3)
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path,
                                       {"finetune.sample_weights": tmp_path / "w.npz"})
    tr = make_trainer(cfg, policy, pre, ds)
    batch = next(iter(tr.train_dataloader))
    idx = batch["dataset_index"].clone().numpy()
    assert not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))), "batch 가 연속 구간이다"
    out = tr.train_step(batch)
    assert out["w_mean"] == pytest.approx(float(w[idx].mean()), rel=1e-6)
    assert np.isfinite(tr.train_metrics["loss"].val)


def test_sampler_choice_with_and_without_pick(synth_root, tmp_path):
    """drop_n_last_frames=0 일 때: PI 제외가 있으면 전 프레임 EpisodeAwareSampler(섞음),
    없으면 기존대로 DataLoader 기본 샘플러 · 검증 loader 는 PI 제외와 무관하게 기존대로."""
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path)
    N = len(ds)
    dl = create_dataloader(ds, cfg, is_training=True)
    assert isinstance(dl.sampler, EpisodeAwareSampler) and dl.sampler.shuffle
    assert dl.sampler.indices == list(range(N))
    dl_val = create_dataloader(ds, cfg, is_training=False)
    assert not isinstance(dl_val.sampler, EpisodeAwareSampler) and len(dl_val.sampler) == N

    cfg2 = compose_cfg(synth_overrides(synth_root, tmp_path, {
        "finetune.loss": "bc", "finetune.sirius_drop_preintv_min": "null"}))
    dl2 = create_dataloader(ds, cfg2, is_training=True)
    assert not isinstance(dl2.sampler, EpisodeAwareSampler) and len(dl2.sampler) == N


# ── (8) 잘못된 입력 -> 분명한 에러 ─────────────────────────────────────────

@pytest.mark.parametrize("case", ["shorter_prefix", "longer", "truncated_one"])
def test_weight_length_mismatch_is_rejected(synth_root, tmp_path, case):
    """길이가 다른 가중 파일 — 이전 라운드 파일(앞부분)을 그대로 넣음 · 다음 라운드 파일 · 한 칸 모자람."""
    modes = synth_modes()
    if case == "shorter_prefix":
        m = modes[:4]
    elif case == "longer":
        m = modes + [np.zeros(20)]
    else:
        m = modes[:-1] + [modes[-1][:-1]]
    write_weights(tmp_path / "w.npz", m)
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path,
                                       {"finetune.sample_weights": tmp_path / "w.npz"})
    raises_clear(lambda: make_trainer(cfg, policy, pre, ds))


def test_weights_from_other_dataset_with_same_length_is_rejected(synth_root, tmp_path):
    """프레임 수는 같지만 에피소드 구성이 다른 데이터셋의 가중 (시연 두 판 길이만 바뀜 -> PI 집합도 같다).
    가중 파일에 episode_index 가 있으므로 기대: 분명한 에러 (길이·drop 이 우연히 같아도)."""
    other = synth_modes()
    other[0], other[1] = other[1], other[0]               # 30,32 -> 32,30
    write_weights(tmp_path / "w.npz", other)
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path,
                                       {"finetune.sample_weights": tmp_path / "w.npz"})
    raises_clear(lambda: make_trainer(cfg, policy, pre, ds))


@pytest.mark.parametrize("flip", ["extra_drop", "missing_drop"])
def test_drop_flags_disagreeing_with_trainer_pi_rule_are_rejected(synth_root, tmp_path, flip):
    """가중 파일의 drop 이 학습의 PI 판정과 한 샘플이라도 다르면 분명한 에러."""
    modes = synth_modes()
    w, drop = write_weights(tmp_path / "tmp.npz", modes)
    w, drop = w.copy(), drop.copy()
    if flip == "extra_drop":
        i = new_ok_indices(w, drop, N_DEMO_FRAMES)[0]
        drop[i], w[i] = True, 0.0
    else:
        i = np.flatnonzero(drop)[0]
        drop[i], w[i] = False, 1.0
    write_weights(tmp_path / "w.npz", modes, w=w, drop=drop)
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path,
                                       {"finetune.sample_weights": tmp_path / "w.npz"})
    raises_clear(lambda: make_trainer(cfg, policy, pre, ds))


def test_k_mismatch_changing_drop_set_is_rejected(synth_root, tmp_path):
    """가중 파일 k=4 · 학습 k=9 — PI 집합이 달라진다 -> 분명한 에러."""
    write_weights(tmp_path / "w.npz", synth_modes(), k=4)
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path, {
        "finetune.sample_weights": tmp_path / "w.npz", "finetune.sirius_drop_preintv_min": 9})
    raises_clear(lambda: make_trainer(cfg, policy, pre, ds))


def test_k_mismatch_with_same_drop_set_is_still_rejected(synth_root, tmp_path):
    """가중 파일은 k=16 으로 만들었다고 적혀 있는데 학습은 k=17 (합성 데이터의 preintv 칸은 최대 15라
    둘 다 PI 가 없다 -> drop 집합은 우연히 같다). 의도 (8) 'k 불일치 -> 분명한 에러' 를 기대한다 —
    이 데이터에선 결과가 같아도 설정이 파일과 어긋났다는 사실은 알려야 한다."""
    assert ref_n_pre_judge(synth_modes(), N_DEMO).max() < 16
    write_weights(tmp_path / "w.npz", synth_modes(), k=16)
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path, {
        "finetune.sample_weights": tmp_path / "w.npz", "finetune.sirius_drop_preintv_min": 17})
    raises_clear(lambda: make_trainer(cfg, policy, pre, ds))


@pytest.mark.parametrize("n_demo_cfg", [1, 3])
def test_n_demo_mismatch_is_rejected(synth_root, tmp_path, n_demo_cfg):
    """가중 파일 n_demo_episodes=2 · 학습 설정은 1 또는 3. 3 이면 개입 판이 시연으로 바뀌어 PI 집합이
    달라지고, 1 이면 시연 한 판이 롤아웃으로 바뀌지만 PI 집합은 그대로다(조용히 넘어갈 수 있는 쪽).
    기대: 둘 다 분명한 에러."""
    write_weights(tmp_path / "w.npz", synth_modes(), n_demo=N_DEMO)
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path, {
        "finetune.sample_weights": tmp_path / "w.npz", "finetune.n_demo_episodes": n_demo_cfg})
    raises_clear(lambda: make_trainer(cfg, policy, pre, ds))


@pytest.mark.parametrize("over", [
    {"finetune.sample_weights": "null"},
    {"finetune.sirius_drop_preintv_min": "null"},
    {"finetune.n_demo_episodes": "null"},
    {"+policy.drop_n_last_frames": DEFAULT_DROP_LAST},
    {"+policy.drop_n_last_frames": None},                 # 지정 안 함 -> 기본 7
    {"finetune.balanced": "[0.5,0.25,0.25]"},
], ids=["no_weights", "no_k", "no_n_demo", "drop_last_7", "drop_last_unset", "balanced"])
def test_missing_or_conflicting_config_is_rejected(synth_root, tmp_path, over):
    write_weights(tmp_path / "w.npz", synth_modes())
    o = {"finetune.sample_weights": tmp_path / "w.npz", **over}
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path, o)
    raises_clear(lambda: make_trainer(cfg, policy, pre, ds))


@pytest.mark.parametrize("judge", [None, "t-1..t+14,clamp"], ids=["none", "other"])
def test_weights_with_other_judge_rule_are_rejected(synth_root, tmp_path, judge):
    """판정 규약이 없거나(2026-10-09 이전 16칸 판정) 다른 가중 파일 — drop 이 우연히 같아도 분명한 에러."""
    write_weights(tmp_path / "w.npz", synth_modes(), judge=judge)
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path,
                                       {"finetune.sample_weights": tmp_path / "w.npz"})
    raises_clear(lambda: make_trainer(cfg, policy, pre, ds))


def test_missing_weight_file_is_rejected(synth_root, tmp_path):
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path,
                                       {"finetune.sample_weights": tmp_path / "nope.npz"})
    raises_clear(lambda: make_trainer(cfg, policy, pre, ds),
                 (AssertionError, ValueError, FileNotFoundError))


def test_validation_split_subset_is_rejected(synth_root, tmp_path):
    """val.num_episodes>0 이면 학습 데이터셋이 앞 에피소드만이라 전체 데이터로 만든 가중과 어긋난다 -> 분명한 에러."""
    write_weights(tmp_path / "w.npz", synth_modes())
    cfg = compose_cfg(synth_overrides(synth_root, tmp_path, {
        "finetune.sample_weights": tmp_path / "w.npz", "val.num_episodes": 1}))
    policy, pre, ds = build_policy_dataset(cfg, episodes=list(range(len(synth_modes()) - 1)))
    raises_clear(lambda: make_trainer(cfg, policy, pre, ds))


# ── 끝에서 끝까지 (build_sample_weights -> train -> 체크포인트 -> 재개) ─────

def test_build_then_train_save_and_resume_end_to_end(synth_root, tmp_path):
    """합성 데이터: 가짜 점수 -> build_sample_weights -> train.py 3 스텝(저장) -> 재개 5 스텝.
    빌더의 drop 을 학습이 그대로 받아들이고(PI 판정 일치), 손실이 유한하고, 체크포인트로 이어진다."""
    modes = synth_modes()
    ep = ref_episode_index(modes)
    fake_scores(tmp_path / "s.npz", ep, np.flatnonzero(ep >= N_DEMO), seed=0)
    base = synth_overrides(synth_root, tmp_path)
    run_module("manibot.scripts.build_sample_weights", base + [
        f"+scores={tmp_path / 's.npz'}", f"+n_demo_episodes={N_DEMO}", f"+drop_preintv_min={K}",
        f"+out={tmp_path / 'w.npz'}", f"hydra.run.dir={tmp_path}"])
    W = np.load(tmp_path / "w.npz")
    ref_drop = (ref_n_pre_judge(modes, N_DEMO) >= K) & (ep >= N_DEMO)
    assert np.array_equal(W["drop"], ref_drop)
    assert (W["w"][ep < N_DEMO] == 1).all()

    train = synth_overrides(synth_root, tmp_path, {
        "finetune.sample_weights": tmp_path / "w.npz", "train.batch_size": 2,
        "train.save_freq": 3, "train.steps": 3})
    out = run_module("manibot.scripts.train", train + [f"hydra.run.dir={tmp_path}"])
    assert "Training completed" in out
    n_drop, N = int(W["drop"].sum()), len(W["w"])
    assert f"PI 제외 {n_drop}" in out
    assert f"preintv {K}칸 이상 샘플 제외: {N} -> {N - n_drop}" in out
    losses = logged_losses(out)
    assert len(losses) == 3 and all(np.isfinite(losses)), losses
    ckpt = tmp_path / "run" / "checkpoints" / "step_0000000003"
    assert (ckpt / "training_state.pt").exists() and (ckpt / "model.safetensors").exists()

    train[train.index("train.steps=3")] = "train.steps=5"
    out = run_module("manibot.scripts.train", train + ["resume=true", f"hydra.run.dir={tmp_path}"])
    assert "Resumed from checkpoint at step 3" in out and "Training completed" in out
    losses = logged_losses(out)
    assert len(losses) == 2 and all(np.isfinite(losses)), losses
    assert (tmp_path / "run" / "checkpoints" / "step_0000000005").exists()


# ── (7) 기존 경로 회귀 (합성) ──────────────────────────────────────────────

def old_inline_frame_labels(root, n_demo, use_preintv=True):
    """train.py 에서 frame_labels 로 옮기기 전의 인라인 코드 그대로 (회귀 기준)."""
    from manibot.utils.intervention_labels import LABEL_PREINTV, relabel_preintv
    z = zarr.open(str(root), "r")["data"]
    ep = np.asarray(z["episode_index"]).ravel()
    mode = np.asarray(z["action_mode"]).ravel()
    fl = np.empty(len(ep), dtype=np.int64)
    for e in np.unique(ep):
        s = ep == e
        if e < n_demo:
            fl[s] = LABELS["demo"]
        elif not use_preintv:
            fl[s] = mode[s]
        else:
            m = relabel_preintv(mode[s], k=15)
            fl[s] = np.where(m == LABEL_PREINTV, LABELS["preintv"], m)
    return fl, ep


@pytest.mark.parametrize("use_preintv", [True, False])
def test_frame_labels_and_windows_match_old_inline_and_reference(synth_root, tmp_path, use_preintv):
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path)
    fl, ep = frame_labels(synth_root, N_DEMO, use_preintv)
    fl_old, ep_old = old_inline_frame_labels(synth_root, N_DEMO, use_preintv)
    assert np.array_equal(fl, fl_old) and np.array_equal(ep, ep_old)
    if use_preintv:
        assert np.array_equal(fl, ref_frame_labels(synth_modes(), N_DEMO))
    assert np.array_equal(action_windows(ds, ep), ref_windows([len(m) for m in synth_modes()]))


def test_sirius_drop9_first_label_unchanged(synth_root, tmp_path):
    """기존 sirius drop9 (첫 칸 라벨 · drop_n_last_frames 기본 7): 라벨·클래스 수는 전체 데이터 기준,
    샘플러 = 에피소드 끝 7 프레임 제외 후 preintv 9칸 이상 제외."""
    modes = synth_modes()
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path, {
        "finetune.loss": "sirius", "finetune.sirius_drop_preintv_min": 9,
        "+policy.drop_n_last_frames": None})
    tr = make_trainer(cfg, policy, pre, ds)
    assert isinstance(tr.loss_fn, SiriusLoss)
    lab = ref_frame_labels(modes, N_DEMO)[ref_windows([len(m) for m in modes])]
    assert np.array_equal(tr.loss_fn.lab, lab)
    cls = lab[:, 0]
    assert tr.loss_fn.n == {c: int((cls == v).sum()) for c, v in LABELS.items()}
    n_pre = ref_n_pre(modes, N_DEMO)
    base = EpisodeAwareSampler(ds.episode_data_index, drop_n_last_frames=DEFAULT_DROP_LAST).indices
    assert tr.train_dataloader.sampler.indices == [i for i in base if n_pre[i] < 9]


def test_sirius_majority_drop_with_all_frames(synth_root, tmp_path):
    """sirius majority + drop9 + drop_n_last_frames=0: 클래스 수(P)는 전체 데이터 기준 그대로,
    샘플러는 전 프레임에서 preintv 9칸 이상만 뺀다."""
    modes = synth_modes()
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path, {
        "finetune.loss": "sirius", "finetune.sirius_drop_preintv_min": 9,
        "finetune.sirius_chunk_label": "majority"})
    tr = make_trainer(cfg, policy, pre, ds)
    lab = ref_frame_labels(modes, N_DEMO)[ref_windows([len(m) for m in modes])]
    vals = np.array(list(LABELS.values()))
    cnt = (lab[:, :, None] == vals).sum(1).astype(float)
    cnt[:, vals == LABELS["preintv"]] -= 0.5
    cls = vals[cnt.argmax(1)]
    assert tr.loss_fn.n == {c: int((cls == v).sum()) for c, v in LABELS.items()}
    assert sum(tr.loss_fn.n.values()) == len(ds)
    n_pre = ref_n_pre(modes, N_DEMO)
    assert tr.train_dataloader.sampler.indices == [i for i in range(len(ds)) if n_pre[i] < 9]


@pytest.mark.parametrize("drop_last", [None, 0])
def test_bc_path_unchanged(synth_root, tmp_path, drop_last):
    """bc: 손실 객체 없음(정책 기본 BC) · 샘플 수 = 기본이면 에피소드마다 끝 7 프레임 제외, 0 이면 전부."""
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path, {
        "finetune.loss": "bc", "finetune.sirius_drop_preintv_min": "null",
        "+policy.drop_n_last_frames": drop_last})
    tr = make_trainer(cfg, policy, pre, ds)
    assert tr.loss_fn is None
    n_ep = len(synth_modes())
    expect = len(ds) - (DEFAULT_DROP_LAST * n_ep if drop_last is None else 0)
    assert len(tr.train_dataloader.sampler) == expect


def test_drop_with_loss_that_cannot_drop_is_rejected(synth_root, tmp_path):
    """sirius_drop_preintv_min 을 bc 와 함께 주면 (라벨이 없다) 분명한 에러 — 기존 동작."""
    cfg, policy, pre, ds = setup_synth(synth_root, tmp_path, {
        "finetune.loss": "bc", "+policy.drop_n_last_frames": None})
    raises_clear(lambda: make_trainer(cfg, policy, pre, ds))


# ── 실데이터 (읽기만 · 없으면 건너뜀) ───────────────────────────────────────

def real_overrides(root, out, over=None):
    ov = {
        "task": "square_ph50_r1", "policy": "diffusion", "device": "cpu",
        "task.dataset_root": str(root), "policy.unet.down_dims": "[32,64]",
        "train.batch_size": 2, "train.num_workers": 0, "train.use_amp": "false",
        "train.log_freq": 1, "wandb.enable": "false",
        "base_dir": str(out), "output_dir": str(Path(out) / "run"),
        "finetune.enabled": "true", "finetune.balanced": "null", "finetune.n_demo_episodes": 50,
    }
    ov.update(over or {})
    return [f"{k}={v}" for k, v in ov.items() if v is not None]


def real_modes(root):
    z = zarr.open(str(root), "r")["data"]
    ep = np.asarray(z["episode_index"]).ravel()
    mode = np.asarray(z["action_mode"]).ravel()
    return [mode[ep == e] for e in np.unique(ep)], ep


need_r1 = pytest.mark.skipif(not R1.exists(), reason="data/square_ph50_r1_zarr 없음")
need_r2 = pytest.mark.skipif(not (R1.exists() and R2.exists()), reason="r1·r2 zarr 없음")


@need_r1
def test_real_r1_sirius_drop9_matches_old_run(tmp_path):
    """outputs/ph50-r1-sirius-drop9 (2026-10-01, 리팩터 전) 의 로그 값과 같아야 한다:
    SIRIUS 가중 demo n=7468 P=0.5066 w=1.000 · robot 4491 0.3047 0.000 · intv 2527 0.1714 2.917 ·
    preintv 255 0.0173 0.116 · 'preintv 9칸 이상 샘플 제외: 14083 -> 13845'."""
    cfg = compose_cfg(real_overrides(R1, tmp_path, {
        "finetune.loss": "sirius", "finetune.sirius_drop_preintv_min": 9}))
    policy, pre, ds = build_policy_dataset(cfg)
    dl = create_dataloader(ds, cfg, is_training=True)
    assert len(dl.sampler.indices) == 14083
    tr = PolicyTrainer(cfg, policy, device="cpu", train_dataloader=dl, preprocessor=pre)
    expect = {"demo": (7468, 0.5066, 1.000), "robot": (4491, 0.3047, 0.000),
              "intv": (2527, 0.1714, 2.917), "preintv": (255, 0.0173, 0.116)}
    for c, (n, p, w) in expect.items():
        assert tr.loss_fn.n[c] == n, c
        assert round(tr.loss_fn.P[c], 4) == p, c
        assert round(tr.loss_fn.w_cls[c], 3) == w, c
    assert len(tr.train_dataloader.sampler.indices) == 13845


@need_r1
def test_real_r1_sirius_majority_drop9_all_frames(tmp_path):
    """majority · p_star_auto=0.002: 클래스 n·P·w 는 outputs/ph50-r1-sirius-human-v2 (2026-10-05, drop 없음)
    로그와 같아야 하고 (P 는 전체 데이터 기준), 샘플러는 전 14741 프레임에서 preintv 9칸 이상만 뺀다."""
    cfg = compose_cfg(real_overrides(R1, tmp_path, {
        "finetune.loss": "sirius", "finetune.sirius_drop_preintv_min": 9,
        "finetune.sirius_chunk_label": "majority", "finetune.sirius_p_star_auto": 0.002,
        "+policy.drop_n_last_frames": 0}))
    policy, pre, ds = build_policy_dataset(cfg)
    tr = make_trainer(cfg, policy, pre, ds)
    expect = {"demo": (7468, 0.5066, 1.000), "robot": (4372, 0.2966, 0.006),
              "intv": (2663, 0.1807, 2.768), "preintv": (238, 0.0161, 0.006)}
    for c, (n, p, w) in expect.items():
        assert tr.loss_fn.n[c] == n, c
        assert round(tr.loss_fn.P[c], 4) == p, c
        assert round(tr.loss_fn.w_cls[c], 3) == w, c
    modes, _ = real_modes(R1)
    n_pre = ref_n_pre(modes, 50)
    assert tr.train_dataloader.sampler.indices == [i for i in range(len(ds)) if n_pre[i] < 9]


@need_r1
def test_real_r1_bc_sample_count_unchanged(tmp_path):
    """bc · drop_n_last_frames 기본: 14741 - 94*7 = 14083 샘플 (drop9 run 의 제외 전 수와 같다)."""
    cfg = compose_cfg(real_overrides(R1, tmp_path, {"finetune.loss": "bc"}))
    policy, pre, ds = build_policy_dataset(cfg)
    tr = make_trainer(cfg, policy, pre, ds)
    assert tr.loss_fn is None
    assert len(tr.train_dataloader.sampler) == 14083


@need_r2
def test_real_r2_chained_weights_train(tmp_path):
    """r1 가중 -> (prev) r2 가중 -> r2 에서 sample_weight 2 스텝 학습.  점수는 가짜(무작위 양수)다 —
    여기서 보는 것은 연쇄 파일을 학습이 받아들이는지(길이·PI 판정 일치)와 prev 범위가 그대로인지."""
    ep1 = np.asarray(zarr.open(str(R1), "r")["data"]["episode_index"]).ravel()
    ep2 = np.asarray(zarr.open(str(R2), "r")["data"]["episode_index"]).ravel()
    fake_scores(tmp_path / "s1.npz", ep1, np.flatnonzero(ep1 >= 50), seed=1)
    run_module("manibot.scripts.build_sample_weights", real_overrides(R1, tmp_path) + [
        f"+scores={tmp_path / 's1.npz'}", "+n_demo_episodes=50",
        f"+out={tmp_path / 'w1.npz'}", f"hydra.run.dir={tmp_path}"])
    W1 = np.load(tmp_path / "w1.npz")
    n1 = len(W1["w"])
    fake_scores(tmp_path / "s2.npz", ep2, np.arange(n1, len(ep2)), seed=2)
    run_module("manibot.scripts.build_sample_weights", real_overrides(R2, tmp_path, {
        "task": "square_ph50_r2"}) + [
        f"+scores={tmp_path / 's2.npz'}", f"+prev={tmp_path / 'w1.npz'}", "+n_demo_episodes=50",
        f"+out={tmp_path / 'w2.npz'}", f"hydra.run.dir={tmp_path}"])
    W2 = np.load(tmp_path / "w2.npz")
    assert np.array_equal(W2["w"][:n1], W1["w"]) and np.array_equal(W2["drop"][:n1], W1["drop"])

    out = run_module("manibot.scripts.train", real_overrides(R2, tmp_path, {
        "task": "square_ph50_r2", "finetune.loss": "sample_weight",
        "finetune.sample_weights": tmp_path / "w2.npz", "finetune.sirius_drop_preintv_min": 4,
        "+policy.drop_n_last_frames": 0, "train.steps": 2, "train.save_freq": 2}) + [
        f"hydra.run.dir={tmp_path}"])
    N, n_drop = len(W2["w"]), int(W2["drop"].sum())
    assert N == len(ep2)
    assert "Training completed" in out
    assert f"PI 제외 {n_drop}" in out
    assert f"preintv 4칸 이상 샘플 제외: {N} -> {N - n_drop}" in out
    losses = logged_losses(out)
    assert len(losses) == 2 and all(np.isfinite(losses)), losses
    assert (tmp_path / "run" / "checkpoints" / "step_0000000002" / "training_state.pt").exists()
