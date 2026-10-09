"""관점 B — score_samples(샘플별 재현 L1 채점)가 의도대로 도는지.

의도 (기대값의 근거):
  - 재현 가능: 같은 시드면 같은 값. 배치 크기·워커 수·처리 범위·전역 RNG 와 무관.
    시드가 다르면 값이 달라지고, 한 샘플의 M 개 생성은 서로 다른 노이즈를 받는다.
  - L1 = |생성 - 라벨| 을 판정 칸(프레임 t..t+14 중 에피소드 안 — t-1 칸·패딩 칸 제외, 사용자 결정 2026-10-09)
    x 행동 차원 평균, 정규화(min-max) 공간. 생성이 라벨과 같으면 0, 일정 오프셋 d 면 d.
  - 정책은 EMA 가중치 · eval 모드(랜덤 크롭 꺼짐).
  - 출력 npz: index · episode_index · l1 (N, M) · l1_mean · l1_min · n_valid_slots · judge · 메타.
  - 잘못된 입력은 조용히 넘어가지 않고 분명한 에러.

방법: 작은 합성 데이터셋(짧은 에피소드 포함)과 작은 정책 체크포인트를 만들어 main 을 프로세스 안에서
부른다. L1 공식은 conditional_sample 을 가짜로 바꿔(생성 = 라벨 + 알려진 오프셋) 기대값을 손으로 정한다.
실데이터(square r1 + base 체크포인트)는 8 프레임 x M=2 만 쓴다 — 없으면 건너뛴다.
"""
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from safetensors.torch import save_file

from manibot.datasets.replay_buffer import ReplayBuffer
from manibot.policies.factory import make_policy
from manibot.scripts import score_samples as ss
from manibot.utils.dataset_utils import create_dataset_stats
from manibot.utils.task_utils import derive_task_meta

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "manibot"
CFG_DIR = str(SRC / "configs")

# 이미지 96 -> 크롭 84 (resize 없음, square 와 같은 경로): ResNet 특징맵이 3x3 이라 크롭 위치가
# 조건을 바꾼다. 32 -> 28 이면 1x1 이라 이미지와 무관해지고, resize_shape 를 주면 LeRobot 이
# crop_ratio=1.0 이라며 크롭을 끈다 — 둘 다 랜덤 크롭(train 모드)을 못 잡는다.
S, A, HW, CROP, FPS = 7, 7, 96, 84, 20
H = 16                       # pred_horizon (policy=diffusion)
# 길이 3 에피소드: 첫 프레임과 끝 프레임의 패딩이 한 창 안에서 겹친다
EPISODES = [20, 18, 3, 17]
N_DEMO = 1
N = sum(EPISODES)
START = EPISODES[0]          # 시연 에피소드가 앞에 있으므로 첫 비시연 프레임
STARTS = np.cumsum([0] + EPISODES[:-1])
ENDS = np.cumsum(EPISODES)

REQUIRED_KEYS = {"index", "episode_index", "l1", "l1_mean", "l1_min", "n_valid_slots",
                 "checkpoint", "M", "seed", "num_inference_steps", "dataset_root", "judge"}
JUDGE = "t..t+14,no_pad"

REAL_DS = ROOT / "data" / "square_ph50_r1_zarr"
REAL_CKPT = (ROOT / "outputs" / "ph50-base-v2" / "diffusion" / "diffusion-20260929_121432"
             / "checkpoints" / "step_0000050000")
REAL_FULL = Path("/tmp/claude-1000/-home-jungwook-workspace-ljw-workspace/"
                 "3c0919a6-1ea8-45d3-b605-feec3e75212e/scratchpad/l1review/scores_r1_full.npz")
REAL_RANGE = (7599, 7607)    # ep 50 끝 4 프레임 + ep 51 첫 4 프레임


# ── 합성 데이터셋 · 체크포인트 ────────────────────────────────────────────

def build_dataset(root):
    shutil.rmtree(root, ignore_errors=True)
    buf = ReplayBuffer.create_from_path(str(root), mode="a")
    rng = np.random.default_rng(0)
    states, actions = [], []
    for i, T in enumerate(EPISODES):
        st = rng.standard_normal((T, S)).astype(np.float32)
        ac = rng.standard_normal((T, A)).astype(np.float32)
        buf.add_episode({
            "observation.state": st,
            "action": ac,
            "observation.images.main": rng.integers(0, 255, (T, HW, HW, 3), dtype=np.uint8),
            "observation.images.wrist": rng.integers(0, 255, (T, HW, HW, 3), dtype=np.uint8),
            "episode_index": np.full(T, i, dtype=np.int64),
            "timestamp": np.arange(T, dtype=np.float32) / FPS,
        })
        states.append(st)
        actions.append(ac)

    def stat(a):
        return {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
                "min": a.min(0).tolist(), "max": a.max(0).tolist()}

    imgs = ["observation.images.main", "observation.images.wrist"]
    stats = {"observation.state": stat(np.concatenate(states)),
             "action": stat(np.concatenate(actions))}
    json.dump({
        "repo_id": None, "stats": stats,
        "num_frames": int(buf.n_steps), "num_episodes": len(EPISODES),
        "features": {"observation.state": {"dtype": "float32", "shape": [S]},
                     "action": {"dtype": "float32", "shape": [A]},
                     **{k: {"dtype": "image", "shape": [HW, HW, 3]} for k in imgs}},
        "camera_keys": imgs, "video_keys": [], "image_keys": imgs,
        "fps": FPS, "tasks": {0: "fake"},
    }, open(root / "config.json", "w"), indent=2)
    return np.concatenate(actions), stats


def make_cfg(root, ckpt=None, out=None, overrides=(), **plus):
    ov = ["task=piper_cube_stack", "policy=diffusion", "device=cpu",
          f"task.dataset_root={root}", "task.dataset_repo_id=null", f"task.fps={FPS}",
          "resize_shape=null", f"crop_shape=[{CROP},{CROP}]", "policy.unet.down_dims=[64,128]",
          "train.num_workers=0", *overrides]
    if ckpt is not None:
        ov.append(f"+checkpoint={ckpt}")
    if out is not None:
        ov.append(f"+out={out}")
    ov += [f"+{k}={v}" for k, v in plus.items()]
    with initialize_config_dir(config_dir=CFG_DIR, version_base="1.3"):
        return compose("default_policy", overrides=ov)


def build_policy(root, seed):
    cfg = make_cfg(root)
    meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, meta)
    torch.manual_seed(seed)
    policy, _, _ = make_policy(cfg, meta, stats)
    return policy


def save_ckpt(path, raw, ema="same"):
    """raw 가중치는 model.safetensors, EMA 는 training_state.pt 의 shadow_params (save_checkpoint 와 같은 배치).
    ema: "same" = raw 와 같은 값 · 정책 객체 = 그 정책의 파라미터 · None = training_state.pt 없음."""
    path.mkdir(parents=True, exist_ok=True)
    raw.save_pretrained(path)
    if ema is None:
        return path
    src = raw if isinstance(ema, str) else ema
    shadow = [p.detach().clone() for p in src.parameters()]
    torch.save({"ema": {"shadow_params": shadow}, "step": 0}, path / "training_state.pt")
    return path


def run(cfg):
    ss.main.__wrapped__(cfg)
    with np.load(cfg.out) as d:
        return {k: d[k] for k in d.files}


def is_clear(exc, arg_names=()):
    """분명한 에러 = 우리 코드(src/manibot)의 raise·assert 문에서 나왔거나, 메시지가 틀린 인자 이름을 말한다.
    torch.cat 의 'non-empty list' · numpy 의 'zero-size array' · 배열 IndexError 처럼 라이브러리 깊은 곳의
    에러는 어느 입력이 틀렸는지 말하지 않으므로 분명한 에러로 치지 않는다.
    (C 함수의 에러는 마지막 파이썬 프레임이 우리 코드라서, 파일만 보지 않고 그 줄이 raise·assert 인지 본다.)"""
    last = traceback.extract_tb(exc.__traceback__)[-1]
    explicit = str(SRC) in last.filename and (last.line or "").lstrip().startswith(("raise", "assert"))
    return explicit or any(n in str(exc) for n in arg_names)


def expect_clear_error(cfg, out_path, types=(AssertionError, ValueError), arg_names=()):
    with pytest.raises(types) as ei:
        ss.main.__wrapped__(cfg)
    assert is_clear(ei.value, arg_names), \
        f"원인을 말하지 않는 내부 에러: {type(ei.value).__name__}: {ei.value}"
    assert not out_path.exists()


def bounds(i):
    e = int(np.searchsorted(ENDS, i, side="right"))
    return int(STARTS[e]), int(ENDS[e]), e


def valid_slots(i):
    """창 칸 k 는 프레임 i-1+k (관측 2칸 -> 행동 창이 i-1 에서 시작). 유효 = 판정 칸: 프레임 i 부터(k >= 1)이고
    에피소드 안인 칸. 칸 0(프레임 i-1)은 에피소드 안이어도 뺀다."""
    _, e1, _ = bounds(i)
    lo, hi = 1, min(H - 1, e1 - i)
    return lo, hi


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("score")
    root = tmp / "ds"
    actions, stats = build_dataset(root)
    ckpt = save_ckpt(tmp / "ckpt", build_policy(root, seed=0))
    return {"tmp": tmp, "root": root, "ckpt": ckpt, "actions": actions, "stats": stats}


@pytest.fixture(scope="module")
def base(syn):
    """기본 채점: 기본 범위(첫 비시연 프레임~끝) · M=2 · 시드 0 · 배치 7 (N 을 나누지 않는다)."""
    cfg = make_cfg(syn["root"], syn["ckpt"], syn["tmp"] / "base.npz",
                   n_demo_episodes=N_DEMO, M=2, batch_size=7, noise_seed=0)
    return run(cfg)


# ── 가짜 생성기: 생성 = f(라벨) 로 L1 의 기대값을 손으로 정한다 ─────────────────

class FakeSampler:
    """conditional_sample 을 가로챈다. 행마다 global_cond 를 배치의 cond 와 맞춰 어느 샘플(b)의
    몇 번째(m) 생성인지 정하고, gen_fn(x0[b], 프레임, m, pad[b]) 를 돌려준다.
    cond 와 label 의 짝·노이즈의 짝이 어긋나면 기대 L1 과 달라지므로 그 배선까지 검사된다."""

    def __init__(self, monkeypatch, gen_fn):
        self.gen_fn = gen_fn
        self.calls = 0
        self.noises = {}
        self.x0s = {}
        self.train_flags = []
        orig_make, orig_prep = ss.make_policy, ss.prepare_cond

        def make(*a, **k):
            policy, pre, post = orig_make(*a, **k)

            def pre_w(batch):
                # 전처리기가 dataset_index 를 버리므로 그 전에 잡아 둔다
                self.idx = batch["dataset_index"].cpu()
                self.pad = batch["action_is_pad"].cpu()
                return pre(batch)
            return policy, pre_w, post

        def prep(policy, batch):
            cond, x0 = orig_prep(policy, batch)
            self.cond, self.x0 = cond, x0
            self.train_flags.append(any(m.training for m in policy.modules()))
            return cond, x0

        def cs(model, batch_size, global_cond=None, generator=None, noise=None):
            assert noise is not None, "고정 노이즈가 넘어오지 않았다"
            assert batch_size == global_cond.shape[0] == noise.shape[0]
            out = torch.empty_like(noise)
            seen = {}
            for r in range(batch_size):
                hit = (global_cond[r] == self.cond).all(-1).nonzero().flatten()
                assert len(hit) == 1, "생성 행의 조건이 배치의 한 샘플과 정확히 대응하지 않는다"
                b = int(hit[0])
                m = seen.get(b, 0)
                seen[b] = m + 1
                f = int(self.idx[b])
                self.noises[(f, m)] = noise[r].detach().cpu().clone()
                self.x0s[f] = self.x0[b].detach().cpu().clone()
                out[r] = self.gen_fn(self.x0[b], f, m, self.pad[b]).to(out)
            self.calls += 1
            return out

        monkeypatch.setattr(ss, "make_policy", make)
        monkeypatch.setattr(ss, "prepare_cond", prep)
        monkeypatch.setattr(DiffusionModel, "conditional_sample", cs)


def identity(x0, f, m, pad):
    return x0.clone()


# ── 재현성 ───────────────────────────────────────────────────────────────

def test_same_seed_identical_regardless_of_global_rng(syn, base, tmp_path):
    """같은 인자면 비트 단위로 같다. 전역 RNG 를 다르게 두어도 같아야 한다 — 랜덤 크롭(train 모드)이나
    전역 RNG 노이즈가 섞이면 여기서 갈린다."""
    torch.manual_seed(12345)
    np.random.seed(12345)
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   n_demo_episodes=N_DEMO, M=2, batch_size=7, noise_seed=0)
    again = run(cfg)
    np.testing.assert_array_equal(again["index"], base["index"])
    np.testing.assert_array_equal(again["l1"], base["l1"])


@pytest.mark.parametrize("bs", [1, 4, 64])
def test_batch_size_does_not_change_scores(syn, base, tmp_path, bs):
    """배치 크기는 샘플이 받는 노이즈·조건을 바꾸지 않는다 (부동소수 합 순서 차이만 허용)."""
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   n_demo_episodes=N_DEMO, M=2, batch_size=bs, noise_seed=0)
    out = run(cfg)
    np.testing.assert_array_equal(out["index"], base["index"])
    np.testing.assert_allclose(out["l1"], base["l1"], rtol=1e-5, atol=1e-6)


def test_num_workers_keeps_order_and_values(syn, base, tmp_path):
    """워커가 여럿이어도 index 순서가 프레임 순서이고 값이 같다."""
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz", overrides=["train.num_workers=2"],
                   n_demo_episodes=N_DEMO, M=2, batch_size=7, noise_seed=0)
    out = run(cfg)
    np.testing.assert_array_equal(out["index"], np.arange(START, N))
    np.testing.assert_allclose(out["l1"], base["l1"], rtol=1e-5, atol=1e-6)


def test_subrange_matches_full_range_rows(syn, base, tmp_path):
    """처리 범위가 달라도 같은 프레임은 같은 노이즈를 받아 같은 값 (프레임 인덱스로 시드)."""
    a, b = 37, 45            # ep1 끝 · 짧은 ep2 전체 · ep3 시작을 걸친다
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   start_index=a, end_index=b, M=2, batch_size=3, noise_seed=0)
    out = run(cfg)
    np.testing.assert_array_equal(out["index"], np.arange(a, b))
    np.testing.assert_allclose(out["l1"], base["l1"][a - START:b - START], rtol=1e-5, atol=1e-6)


def test_different_seed_changes_scores(syn, base, tmp_path):
    """노이즈 시드가 다르면 다른 생성 -> 다른 L1. 메타에도 시드가 남는다."""
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   n_demo_episodes=N_DEMO, M=2, batch_size=7, noise_seed=1)
    out = run(cfg)
    assert int(out["seed"]) == 1
    assert np.mean(out["l1"] != base["l1"]) > 0.9


def test_m_generations_are_distinct(base):
    """한 샘플의 M 개 생성은 서로 다른 노이즈 -> 서로 다른 L1 (같다면 M 이 의미가 없다)."""
    assert np.mean(base["l1"][:, 0] != base["l1"][:, 1]) > 0.9


def test_noise_is_seeded_by_frame_index(syn, monkeypatch, tmp_path):
    """프레임 i 의 m 번째 노이즈 = randn((M,H,D), CPU 생성기 seed*1e8 + i)[m] — 배치 위치와 무관."""
    fake = FakeSampler(monkeypatch, identity)
    seed, M = 3, 3
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   start_index=30, end_index=45, M=M, batch_size=4, noise_seed=seed)
    run(cfg)
    assert set(fake.noises) == {(f, m) for f in range(30, 45) for m in range(M)}
    for f in range(30, 45):
        want = torch.randn((M, H, A), generator=torch.Generator().manual_seed(seed * 10**8 + f))
        for m in range(M):
            torch.testing.assert_close(fake.noises[(f, m)], want[m], rtol=0, atol=0)
    assert not torch.equal(fake.noises[(30, 0)], fake.noises[(31, 0)])


def test_m_equals_one(syn, tmp_path):
    """M=1 이면 l1 은 (N, 1) 이고 l1_mean = l1_min = l1[:, 0]."""
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   n_demo_episodes=N_DEMO, M=1, batch_size=5)
    out = run(cfg)
    assert out["l1"].shape == (N - START, 1)
    np.testing.assert_array_equal(out["l1_mean"], out["l1"][:, 0])
    np.testing.assert_array_equal(out["l1_min"], out["l1"][:, 0])
    assert int(out["M"]) == 1


# ── 범위 ─────────────────────────────────────────────────────────────────

def test_default_start_from_n_demo(base):
    """기본 범위 = 첫 비시연 프레임부터 끝까지."""
    np.testing.assert_array_equal(base["index"], np.arange(START, N))
    assert (base["episode_index"] >= N_DEMO).all()


def test_n_demo_from_finetune_section(syn, base, tmp_path):
    """+n_demo_episodes 가 없으면 finetune.n_demo_episodes 를 쓴다."""
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   overrides=[f"finetune.n_demo_episodes={N_DEMO}"], M=2, batch_size=7)
    out = run(cfg)
    np.testing.assert_array_equal(out["index"], base["index"])
    np.testing.assert_array_equal(out["l1"], base["l1"])


def test_start_inside_demo_is_scored_consistently(syn, base, tmp_path):
    """start_index 가 시연 안이면 시연 프레임도 채점한다 (명세상 금지가 없다). 시연 행의 episode_index 는
    시연이고, 비시연 행은 기본 범위 결과와 같다."""
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   start_index=5, M=2, batch_size=7, noise_seed=0)
    out = run(cfg)
    np.testing.assert_array_equal(out["index"], np.arange(5, N))
    assert (out["episode_index"][:START - 5] == 0).all()
    np.testing.assert_allclose(out["l1"][START - 5:], base["l1"], rtol=1e-5, atol=1e-6)


def test_missing_start_and_n_demo_errors(syn, tmp_path):
    """시작점을 정할 정보가 없으면 분명한 에러."""
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz", M=1)
    expect_clear_error(cfg, tmp_path / "s.npz")


def test_n_demo_zero_defined(syn, tmp_path):
    """경계(명세 밖): n_demo_episodes=0. 기대 = 분명한 에러, 또는 정의된 값(시작 0 = 전부 채점)."""
    out_path = tmp_path / "s.npz"
    cfg = make_cfg(syn["root"], syn["ckpt"], out_path, n_demo_episodes=0, M=1, batch_size=16)
    try:
        ss.main.__wrapped__(cfg)
    except (AssertionError, ValueError) as e:
        assert is_clear(e), f"원인을 말하지 않는 내부 에러: {e}"
        assert not out_path.exists()
        return
    with np.load(out_path) as d:
        np.testing.assert_array_equal(d["index"], np.arange(0, N))


def assert_empty_or_clear_error(cfg, out_path):
    """빈 범위: 기대 = 분명한 에러(AssertionError/ValueError, 파일 없음) 또는 정의된 값(0 행 npz, 필수 키).
    torch.cat 의 'non-empty list' 같은 내부 에러는 원인(범위)을 말하지 않으므로 분명한 에러로 치지 않는다."""
    try:
        ss.main.__wrapped__(cfg)
    except (AssertionError, ValueError) as e:
        assert is_clear(e, ("start_index", "end_index", "n_demo")), f"원인을 말하지 않는 내부 에러: {e}"
        assert not out_path.exists()
        return
    with np.load(out_path) as d:
        assert REQUIRED_KEYS <= set(d.files)
        assert len(d["index"]) == 0 and d["l1"].shape[0] == 0


def test_start_equals_end(syn, tmp_path):
    """start_index = end_index -> 빈 범위 (경계: 에러 또는 0 행)."""
    out_path = tmp_path / "s.npz"
    cfg = make_cfg(syn["root"], syn["ckpt"], out_path, start_index=25, end_index=25, M=1)
    assert_empty_or_clear_error(cfg, out_path)


def test_end_index_zero_is_not_full_dataset(syn, tmp_path):
    """+end_index=0 은 빈 범위다 — 0 을 '없음' 으로 읽어 데이터셋 전체를 채점하면 안 된다."""
    out_path = tmp_path / "s.npz"
    cfg = make_cfg(syn["root"], syn["ckpt"], out_path, start_index=0, end_index=0, M=1,
                   batch_size=64)
    assert_empty_or_clear_error(cfg, out_path)


def test_start_greater_than_end_errors(syn, tmp_path):
    """start > end 는 잘못된 입력 -> 분명한 에러 (AssertionError/ValueError), 파일 없음."""
    out_path = tmp_path / "s.npz"
    cfg = make_cfg(syn["root"], syn["ckpt"], out_path, start_index=30, end_index=25, M=1)
    expect_clear_error(cfg, out_path, arg_names=("start_index", "end_index"))


def test_end_beyond_dataset_errors(syn, tmp_path):
    """end_index > N 은 잘못된 입력 -> 분명한 에러 (AssertionError/ValueError), 파일 없음.
    조용히 잘라 쓰는 것도, 데이터셋 깊은 곳의 IndexError 도 기대가 아니다 (어느 인자가 틀렸는지 안 보인다)."""
    out_path = tmp_path / "s.npz"
    cfg = make_cfg(syn["root"], syn["ckpt"], out_path, start_index=50, end_index=N + 5, M=1)
    expect_clear_error(cfg, out_path, arg_names=("end_index",))


def test_negative_start_errors(syn, tmp_path):
    """음수 start_index 는 잘못된 입력 -> 분명한 에러. (그대로 두면 파이썬 음수 인덱싱으로 끝 프레임을
    엉뚱한 index·노이즈로 채점해 저장한다.)"""
    out_path = tmp_path / "s.npz"
    cfg = make_cfg(syn["root"], syn["ckpt"], out_path, start_index=-3, end_index=N, M=1)
    expect_clear_error(cfg, out_path, arg_names=("start_index",))


def test_n_demo_beyond_episodes_errors(syn, tmp_path):
    """n_demo_episodes 가 에피소드 수 이상이면 채점할 프레임이 없다 -> 분명한 에러 (경계: 0 행도 허용)."""
    out_path = tmp_path / "s.npz"
    cfg = make_cfg(syn["root"], syn["ckpt"], out_path, n_demo_episodes=len(EPISODES) + 6, M=1)
    assert_empty_or_clear_error(cfg, out_path)


@pytest.mark.parametrize("key,val", [("M", 0), ("M", -1), ("batch_size", 0)])
def test_bad_counts_error(syn, monkeypatch, tmp_path, key, val):
    """M<=0 · batch_size<=0 -> 분명한 에러 (AssertionError/ValueError), 파일 없음.
    M=0 이 NaN l1_mean 으로 저장되면 안 된다."""
    FakeSampler(monkeypatch, identity)
    out_path = tmp_path / "s.npz"
    kw = {"M": 1, "batch_size": 4, key: val}
    cfg = make_cfg(syn["root"], syn["ckpt"], out_path, start_index=20, end_index=26, **kw)
    # batch_size 는 DataLoader 의 에러가 인자 이름을 말한다 — M 은 이름이 짧아 메시지로 가리지 않는다
    expect_clear_error(cfg, out_path, arg_names=("batch_size",) if key == "batch_size" else ())


# ── 패딩 칸 · L1 공식 (가짜 생성기) ────────────────────────────────────────

def test_n_valid_slots_formula(base):
    """n_valid_slots = 판정 칸 수 (프레임 t..t+14 중 에피소드 안, 최대 15). 에피소드 앞쪽은 15 (첫 프레임도 15),
    끝 프레임 1, 길이 3 에피소드는 3·2·1."""
    want = np.array([valid_slots(i)[1] - valid_slots(i)[0] + 1 for i in base["index"]])
    np.testing.assert_array_equal(base["n_valid_slots"], want)
    nv = dict(zip(base["index"].tolist(), base["n_valid_slots"].tolist()))
    assert nv[20] == 15 and nv[37] == 1 and nv[21] == 15
    assert [nv[38], nv[39], nv[40]] == [3, 2, 1]


def test_identity_generation_gives_zero(syn, monkeypatch, tmp_path):
    """생성 = 라벨 -> L1 = 0 (모든 샘플·모든 m)."""
    FakeSampler(monkeypatch, identity)
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   n_demo_episodes=N_DEMO, M=2, batch_size=5)
    out = run(cfg)
    assert np.all(out["l1"] == 0)
    assert np.all(out["l1_mean"] == 0) and np.all(out["l1_min"] == 0)


@pytest.mark.parametrize("d", [0.25, -0.4])
def test_constant_offset_gives_d(syn, monkeypatch, tmp_path, d):
    """생성 = 라벨 + d (모든 칸·차원) -> L1 = |d|."""
    FakeSampler(monkeypatch, lambda x0, f, m, pad: x0 + d)
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   n_demo_episodes=N_DEMO, M=2, batch_size=5)
    out = run(cfg)
    np.testing.assert_allclose(out["l1"], abs(d), atol=1e-6)


def test_per_m_offset_maps_to_columns(syn, monkeypatch, tmp_path):
    """m 번째 생성의 오프셋이 l1[:, m] 에 간다 (M 축 배치 순서). l1_mean·l1_min 은 그 평균·최소."""
    offs = [0.1, 0.3, 0.2]
    FakeSampler(monkeypatch, lambda x0, f, m, pad: x0 + offs[m])
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   n_demo_episodes=N_DEMO, M=3, batch_size=4)
    out = run(cfg)
    np.testing.assert_allclose(out["l1"], np.tile(offs, (N - START, 1)), atol=1e-6)
    np.testing.assert_allclose(out["l1_mean"], np.mean(offs), atol=1e-6)
    np.testing.assert_allclose(out["l1_min"], min(offs), atol=1e-6)


def test_per_dim_offset_averaged_over_dims(syn, monkeypatch, tmp_path):
    """차원별 오프셋 -> L1 = 차원 평균 |delta| (차원 축 평균)."""
    delta = torch.tensor([0.1 * (j + 1) * (-1) ** j for j in range(A)])
    FakeSampler(monkeypatch, lambda x0, f, m, pad: x0 + delta.to(x0))
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   n_demo_episodes=N_DEMO, M=1, batch_size=5)
    out = run(cfg)
    np.testing.assert_allclose(out["l1"], float(delta.abs().mean()), atol=1e-6)


def test_pad_slots_do_not_enter_l1(syn, monkeypatch, tmp_path):
    """패딩 칸에 큰 차이(100)를 넣어도 L1 은 유효 칸의 차이(0.1) 그대로."""
    def gen(x0, f, m, pad):
        off = torch.where(pad.to(x0.device)[:, None], torch.tensor(100.0), torch.tensor(0.1))
        return x0 + off.to(x0)
    FakeSampler(monkeypatch, gen)
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   n_demo_episodes=N_DEMO, M=2, batch_size=5)
    out = run(cfg)
    np.testing.assert_allclose(out["l1"], 0.1, atol=1e-6)


def test_slot_offsets_averaged_over_valid_slots_only(syn, monkeypatch, tmp_path):
    """칸 k 에 오프셋 k -> L1 = 유효 칸 번호의 평균 (lo+hi)/2. 분모가 16 이거나 마스크 방향이 틀리면 갈린다.
    패딩 판정을 action_is_pad 가 아니라 에피소드 경계에서 독립적으로 계산해 비교한다."""
    slot = torch.arange(H, dtype=torch.float32)[:, None]
    FakeSampler(monkeypatch, lambda x0, f, m, pad: x0 + slot.to(x0))
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   start_index=0, M=1, batch_size=6)
    out = run(cfg)
    want = np.array([sum(valid_slots(i)) / 2 for i in range(N)])
    np.testing.assert_allclose(out["l1"][:, 0], want, atol=1e-5)


def test_l1_in_minmax_normalized_space(syn, monkeypatch, tmp_path):
    """생성 = 0 (정규화 공간의 0) -> L1 = 유효 칸 평균 |x0|, x0 = 2(a-min)/(max-min)-1 을 zarr 원자료와
    config.json stats 로 따로 계산한 값. 라벨이 [-1, 1] 안인지도 본다."""
    fake = FakeSampler(monkeypatch, lambda x0, f, m, pad: torch.zeros_like(x0))
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   start_index=0, M=1, batch_size=8)
    out = run(cfg)
    a = syn["actions"].astype(np.float64)
    lo = np.asarray(syn["stats"]["action"]["min"])
    hi = np.asarray(syn["stats"]["action"]["max"])
    want = []
    for i in range(N):
        e0, e1, _ = bounds(i)
        win = np.clip(np.arange(i - 1, i - 1 + H), e0, e1 - 1)
        x0 = 2 * (a[win] - lo) / (hi - lo) - 1
        k0, k1 = valid_slots(i)
        want.append(np.abs(x0[k0:k1 + 1]).mean())
        np.testing.assert_allclose(fake.x0s[i].double().numpy()[k0:k1 + 1], x0[k0:k1 + 1], atol=1e-5)
    np.testing.assert_allclose(out["l1"][:, 0], want, atol=1e-5)
    assert all(float(x.abs().max()) <= 1 + 1e-5 for x in fake.x0s.values())


def test_nonfinite_generation_errors(syn, monkeypatch, tmp_path):
    """경계(명세 밖): 생성에 NaN 이 섞이면 L1 이 NaN. 기대 = 분명한 에러, 파일 없음 —
    NaN 을 저장하면 build_sample_weights 의 평균까지 조용히 NaN 이 된다."""
    def gen(x0, f, m, pad):
        return torch.full_like(x0, float("nan")) if f == 25 else x0.clone()
    FakeSampler(monkeypatch, gen)
    out_path = tmp_path / "s.npz"
    cfg = make_cfg(syn["root"], syn["ckpt"], out_path, start_index=20, end_index=30, M=1)
    expect_clear_error(cfg, out_path, types=(AssertionError, ValueError, FloatingPointError))


# ── EMA · eval 모드 ──────────────────────────────────────────────────────

def test_policy_in_eval_mode_while_scoring(syn, monkeypatch, tmp_path):
    """채점 중 정책의 모든 모듈이 eval (랜덤 크롭 대신 센터 크롭)."""
    fake = FakeSampler(monkeypatch, identity)
    cfg = make_cfg(syn["root"], syn["ckpt"], tmp_path / "s.npz",
                   start_index=20, end_index=30, M=1, batch_size=4)
    run(cfg)
    assert fake.train_flags and not any(fake.train_flags)


@pytest.fixture(scope="module")
def ema_ckpts(syn):
    p1, p2 = build_policy(syn["root"], seed=1), build_policy(syn["root"], seed=2)
    t = syn["tmp"]
    return {
        "raw1_ema2": save_ckpt(t / "raw1_ema2", p1, ema=p2),
        "raw2_ema2": save_ckpt(t / "raw2_ema2", p2, ema=p2),
        "raw1_ema1": save_ckpt(t / "raw1_ema1", p1, ema=p1),
    }


def test_ema_weights_are_used(syn, ema_ckpts, tmp_path):
    """raw=정책1 · EMA=정책2 체크포인트는 정책2 로 채점한 것과 같다 (정책1 과는 다르다)."""
    def score(name):
        cfg = make_cfg(syn["root"], ema_ckpts[name], tmp_path / f"{name}.npz",
                       start_index=20, end_index=28, M=1, batch_size=4)
        return run(cfg)["l1"]
    l1_mix, l1_b, l1_a = score("raw1_ema2"), score("raw2_ema2"), score("raw1_ema1")
    np.testing.assert_array_equal(l1_mix, l1_b)
    assert not np.allclose(l1_mix, l1_a)


def test_ema_count_mismatch_errors(syn, tmp_path):
    """training_state 에 EMA 가 있는데 정책에 못 실으면(파라미터 수 불일치) 분명한 에러.
    raw 가중치로 조용히 채점하면 '직전 정책(EMA)' 이라는 의도와 다른 값이 저장된다."""
    p = build_policy(syn["root"], seed=0)
    ck = save_ckpt(tmp_path / "ck", p)
    st = torch.load(ck / "training_state.pt", weights_only=False)
    st["ema"]["shadow_params"] = st["ema"]["shadow_params"][:-1]
    torch.save(st, ck / "training_state.pt")
    out_path = tmp_path / "s.npz"
    cfg = make_cfg(syn["root"], ck, out_path, start_index=20, end_index=24, M=1)
    expect_clear_error(cfg, out_path, types=(AssertionError, ValueError, RuntimeError))


def test_checkpoint_without_ema_errors_or_recorded(syn, tmp_path):
    """경계(명세 밖): EMA 가 없는 체크포인트(training_state.pt 없음). 기대 = 분명한 에러, 또는 정의된 값
    (raw 로 채점하되 npz 에 EMA 미사용이 남는다 — 이름에 'ema' 가 든 키가 False)."""
    ck = save_ckpt(tmp_path / "ck", build_policy(syn["root"], seed=0), ema=None)
    out_path = tmp_path / "s.npz"
    cfg = make_cfg(syn["root"], ck, out_path, start_index=20, end_index=24, M=1)
    try:
        ss.main.__wrapped__(cfg)
    except (AssertionError, ValueError, FileNotFoundError) as e:
        assert is_clear(e), f"원인을 말하지 않는 내부 에러: {e}"
        assert not out_path.exists()
        return
    with np.load(out_path) as d:
        keys = [k for k in d.files if "ema" in k.lower()]
        assert keys, "EMA 없이 채점했는데 npz 에 그 사실이 남지 않았다"
        assert not bool(d[keys[0]])


# ── 체크포인트 · 인자 오류 ─────────────────────────────────────────────────

def test_missing_checkpoint_dir_errors(syn, tmp_path):
    """체크포인트 경로가 없으면 분명한 에러."""
    out_path = tmp_path / "s.npz"
    cfg = make_cfg(syn["root"], tmp_path / "nope", out_path, start_index=20, end_index=24, M=1)
    expect_clear_error(cfg, out_path, types=(FileNotFoundError, AssertionError, ValueError))


def test_checkpoint_without_weights_errors(syn, tmp_path):
    """폴더는 있는데 가중치 파일이 없으면 분명한 에러."""
    (tmp_path / "empty").mkdir()
    out_path = tmp_path / "s.npz"
    cfg = make_cfg(syn["root"], tmp_path / "empty", out_path, start_index=20, end_index=24, M=1)
    expect_clear_error(cfg, out_path, types=(FileNotFoundError, AssertionError, ValueError))


def test_checkpoint_keys_mismatch_errors(syn, tmp_path):
    """가중치 키가 정책과 안 맞으면(다른 구조의 체크포인트) 분명한 에러.
    strict=False 경고만 내고 넘어가면 무작위 초기화 정책으로 채점한 값이 저장된다."""
    p = build_policy(syn["root"], seed=0)
    ck = tmp_path / "ck"
    ck.mkdir()
    save_file({f"other.{k}": v.detach().clone().contiguous() for k, v in p.state_dict().items()},
              str(ck / "model.safetensors"))
    out_path = tmp_path / "s.npz"
    cfg = make_cfg(syn["root"], ck, out_path, start_index=20, end_index=24, M=1)
    expect_clear_error(cfg, out_path, types=(AssertionError, ValueError, RuntimeError, KeyError))


def test_missing_checkpoint_arg_errors_before_scoring(syn, monkeypatch, tmp_path):
    """+checkpoint 가 없으면 생성 전에 에러."""
    fake = FakeSampler(monkeypatch, identity)
    cfg = make_cfg(syn["root"], None, tmp_path / "s.npz", start_index=20, end_index=24, M=1)
    with pytest.raises(Exception):
        ss.main.__wrapped__(cfg)
    assert fake.calls == 0


def test_missing_out_arg_errors_before_scoring(syn, monkeypatch, tmp_path):
    """+out 이 없으면 채점을 다 한 뒤가 아니라 생성 전에 에러 — 경계(명세엔 '언제' 가 없다):
    전체 채점은 수천 프레임이라 끝에서 실패하면 계산을 통째로 버린다."""
    fake = FakeSampler(monkeypatch, identity)
    cfg = make_cfg(syn["root"], syn["ckpt"], None, start_index=20, end_index=24, M=1)
    with pytest.raises(Exception):
        ss.main.__wrapped__(cfg)
    assert fake.calls == 0


def test_out_dir_missing_created_or_early_error(syn, monkeypatch, tmp_path):
    """경계: 출력 폴더가 없으면 만들어 저장하거나, 생성 전에 에러 (끝에서 실패해 계산을 버리지 않는다)."""
    fake = FakeSampler(monkeypatch, identity)
    out_path = tmp_path / "no" / "such" / "s.npz"
    cfg = make_cfg(syn["root"], syn["ckpt"], out_path, start_index=20, end_index=24, M=1)
    try:
        ss.main.__wrapped__(cfg)
    except Exception:
        assert fake.calls == 0
        return
    assert out_path.exists()


# ── 출력 형식 ────────────────────────────────────────────────────────────

def test_output_keys_dtypes_order(syn, base):
    """필수 키 · dtype · 모양 · 순서 · 메타가 명세대로이고 값이 유한·비음수."""
    assert REQUIRED_KEYS <= set(base)
    n = N - START
    np.testing.assert_array_equal(base["index"], np.arange(START, N))
    assert base["index"].dtype == np.int64
    ep_raw = np.repeat(np.arange(len(EPISODES)), EPISODES)
    np.testing.assert_array_equal(base["episode_index"], ep_raw[base["index"]])
    assert np.issubdtype(base["episode_index"].dtype, np.integer)
    assert base["l1"].shape == (n, 2) and base["l1"].dtype == np.float32
    for k in ("l1_mean", "l1_min"):
        assert base[k].shape == (n,) and base[k].dtype == np.float32
    assert base["n_valid_slots"].shape == (n,)
    assert np.issubdtype(base["n_valid_slots"].dtype, np.integer)
    np.testing.assert_allclose(base["l1_mean"], base["l1"].mean(1), rtol=1e-6)
    np.testing.assert_array_equal(base["l1_min"], base["l1"].min(1))
    assert np.isfinite(base["l1"]).all() and (base["l1"] >= 0).all()
    assert int(base["M"]) == 2 and int(base["seed"]) == 0
    assert int(base["num_inference_steps"]) == 10           # policy=diffusion 의 DDIM 10
    assert str(base["checkpoint"]) == str(syn["ckpt"])
    assert str(base["dataset_root"]) == str(syn["root"])
    assert str(base["judge"]) == JUDGE


def test_cli_entrypoint_matches_in_process(syn, base, tmp_path):
    """hydra CLI 진입점(+인자)으로 돌려도 같은 값 — 실제 사용 경로."""
    out_path = tmp_path / "cli.npz"
    cmd = [sys.executable, "-m", "manibot.scripts.score_samples",
           "task=piper_cube_stack", "policy=diffusion", "device=cpu",
           f"task.dataset_root={syn['root']}", "task.dataset_repo_id=null", f"task.fps={FPS}",
           "resize_shape=null", f"crop_shape=[{CROP},{CROP}]", "policy.unet.down_dims=[64,128]",
           "train.num_workers=0", f"hydra.run.dir={tmp_path}",
           f"+checkpoint={syn['ckpt']}", "+start_index=36", "+end_index=42",
           "+M=2", "+batch_size=4", "+noise_seed=0", f"+out={out_path}"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    with np.load(out_path) as d:
        np.testing.assert_array_equal(d["index"], np.arange(36, 42))
        np.testing.assert_allclose(d["l1"], base["l1"][36 - START:42 - START], rtol=1e-5, atol=1e-6)


# ── 실데이터 (square r1 + base 50K) — 작은 범위만 ───────────────────────────

def real_cfg(out, overrides=(), **plus):
    ov = ["task=square_ph50_r1", "policy=diffusion", "device=cpu", "train.num_workers=0",
          f"task.dataset_root={REAL_DS}", f"+checkpoint={REAL_CKPT}", f"+out={out}", *overrides]
    ov += [f"+{k}={v}" for k, v in plus.items()]
    with initialize_config_dir(config_dir=CFG_DIR, version_base="1.3"):
        return compose("default_policy", overrides=ov)


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    if not (REAL_DS.exists() and REAL_CKPT.exists()):
        pytest.skip("실데이터·체크포인트가 이 머신에 없다")
    tmp = tmp_path_factory.mktemp("real")
    torch.manual_seed(1)
    a, b = REAL_RANGE
    return run(real_cfg(tmp / "r.npz", start_index=a, end_index=b, M=2, batch_size=3)), tmp


def test_real_n_valid_and_episode(real):
    """실데이터: 에피소드 경계에서 n_valid_slots 가 공식대로 (끝 4 프레임 4·3·2·1, 다음 에피소드 첫 4 프레임 15)."""
    import zarr
    out, _ = real
    ep = np.asarray(zarr.open(str(REAL_DS), "r")["data"]["episode_index"]).ravel()
    np.testing.assert_array_equal(out["index"], np.arange(*REAL_RANGE))
    np.testing.assert_array_equal(out["episode_index"], ep[out["index"]])
    want = []
    for i in out["index"]:
        s = np.flatnonzero(ep == ep[i])
        win = np.arange(i, i + H - 1)                     # 판정 칸 = 프레임 i .. i+14
        want.append(int(((win >= s[0]) & (win <= s[-1])).sum()))
    np.testing.assert_array_equal(out["n_valid_slots"], want)
    np.testing.assert_array_equal(out["n_valid_slots"], [4, 3, 2, 1, 15, 15, 15, 15])
    assert np.isfinite(out["l1"]).all() and (out["l1"] > 0).all()


def test_real_batch_size_and_global_rng_invariant(real):
    """실정책(랜덤 크롭 설정 76/84): 배치 크기·전역 RNG 를 바꿔도 같은 값 — eval 모드·고정 노이즈."""
    out, tmp = real
    torch.manual_seed(999)
    a, b = REAL_RANGE
    again = run(real_cfg(tmp / "r8.npz", start_index=a, end_index=b, M=2, batch_size=8))
    np.testing.assert_allclose(again["l1"], out["l1"], rtol=1e-5, atol=1e-6)


def test_real_matches_previous_full_run(real):
    """실데이터: 앞서 전체 범위(시드 0, M=2)로 채점한 산출물의 같은 행과 일치 — 범위·배치·기기와 무관."""
    if not REAL_FULL.exists():
        pytest.skip("이전 전체 채점 산출물이 없다")
    out, _ = real
    with np.load(REAL_FULL) as f:
        if "judge" not in f.files or str(f["judge"]) != JUDGE:
            pytest.skip("이전 전체 채점이 지금 판정 칸(t..t+14, 패딩 제외)으로 만든 것이 아니다 — L1 을 비교할 수 없다")
        assert int(f["seed"]) == 0 and int(f["M"]) == 2
        rows = np.searchsorted(f["index"], out["index"])
        np.testing.assert_array_equal(f["index"][rows], out["index"])
        np.testing.assert_allclose(out["l1"], f["l1"][rows], rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(out["n_valid_slots"], f["n_valid_slots"][rows])


def test_real_default_start_is_first_rollout_frame(real):
    """실데이터: +n_demo_episodes=50 이면 시작 = 시연 프레임 수(7468)."""
    import zarr
    _, tmp = real
    ep = np.asarray(zarr.open(str(REAL_DS), "r")["data"]["episode_index"]).ravel()
    n_demo_frames = int((ep < 50).sum())
    out = run(real_cfg(tmp / "rd.npz", n_demo_episodes=50, end_index=n_demo_frames + 2, M=1))
    np.testing.assert_array_equal(out["index"], [n_demo_frames, n_demo_frames + 1])
    assert (out["episode_index"] == 50).all()
