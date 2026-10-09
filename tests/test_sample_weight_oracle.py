"""독립 기준 구현(oracle) 대조 — sample_weight 경로의 PI 판정 칸 · drop · L1 유효 칸.

정의 [사용자 결정 2026-10-09] — 기대값은 이 정의만으로 아래 oracle_* 가 순수 파이썬으로 계산한다.
저장소 함수(frame_labels · relabel_preintv · action_windows · judge_slots)는 기대값에 쓰지 않는다.

  프레임 라벨  앞 n_demo 에피소드 = demo (action_mode 와 무관). 배포 에피소드는 action_mode 0 robot · 1 intv.
               개입 시작 프레임(앞 프레임이 robot 인 intv 프레임, 또는 에피소드 첫 프레임의 intv) 직전의
               robot 프레임을 최대 15 개까지 거슬러 가며 preintv 로 바꾼다. 앞선 intv 프레임이나 에피소드
               시작을 만나면 멈춘다.
  판정 칸      샘플 t 의 판정 칸 = 프레임 t..t+14 중 같은 에피소드 안 (최대 15 칸). 행동 창(t-1..t+14)의
               t-1 칸과 에피소드 밖을 경계 프레임으로 채운 패딩 칸은 쓰지 않는다.
  drop(t)      t 가 배포 에피소드 프레임 AND 판정 칸의 preintv 수 >= drop_preintv_min. 시연은 drop 없음.
  L1 유효 칸   판정 칸과 같은 칸 (창 1..15 번 칸 중 패딩 아닌 칸). n_valid_slots = 그 수.
  loss=sirius  지금 동작 그대로 — 창 16 칸(t-1..t+14, 에피소드 밖은 경계 프레임 복사) 전부를 센다.

대조 대상 (프레임 단위, 하나라도 다르면 실패):
  (a) build_sample_weights 의 drop · (w == 0)      (b) build 의 시연 w == 1
  (c) train.py sample_weight 학습의 sampler 에서 빠지는 샘플 집합
      (가중 파일 = oracle 이 쓴 파일 · build 가 만든 파일 두 가지)
  (d) score_samples 의 n_valid_slots 와 L1 이 평균한 칸 (가짜 생성기: 생성 = 라벨 + 칸별 오프셋)
  (e) loss=sirius 의 sampler 제외 집합 (16 칸 셈 그대로인지 — 정의의 마지막 줄)

경우: 배포 에피소드 길이 1..40 x 개입 시작(없음 · 0..L-1, 개입 뒤 끝까지 intv) 전부 +
      개입 두 번(robot a · intv b · robot g · intv c · robot d, 간격 g = 0..20) 여러 패턴.
      drop_preintv_min = 1 · 4 · 15.
산출물은 pytest tmp_path 아래에만 만든다.
실행 (manibot 루트, CPU):
    CUDA_VISIBLE_DEVICES="" ~/miniconda3/envs/manibot/bin/python -m pytest -p no:cacheprovider \
        tests/test_sample_weight_oracle.py
"""
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel

from manibot.datasets.replay_buffer import ReplayBuffer
from manibot.policies.factory import make_policy
from manibot.scripts import build_sample_weights as bsw
from manibot.scripts import score_samples as ss
from manibot.scripts.train import PolicyTrainer
from manibot.utils.dataset_utils import create_dataloader, create_dataset, create_dataset_stats
from manibot.utils.task_utils import derive_task_meta

REPO = Path(__file__).resolve().parents[1]
CFG_DIR = str(REPO / "src" / "manibot" / "configs")
S_DIM, A_DIM, HW, FPS = 7, 7, 32, 20
H = 16                       # 행동 창 칸 수 (delta -1..14)
K_PRE = 15                   # 개입 직전 preintv 최대 길이
JUDGE_LEN = 15               # 판정 칸 최대 수 (t..t+14)
K_VALUES = (1, 4, 15)
JUDGE = "t..t+14,no_pad"     # 점수·가중 파일의 판정 규약 이름 (oracle 이 쓰는 파일에만 쓴다)


# ── 경우 목록 ──────────────────────────────────────────────────────────────

def demo_episodes():
    # 시연의 action_mode 는 보지 않아야 한다 — 개입처럼 생긴 모드를 하나 넣어 둔다
    return [("demo L=20", [0] * 20), ("demo L=20 mode 0^10 1^10", [0] * 10 + [1] * 10),
            ("demo L=3", [0] * 3)]


def single_episodes():
    out = []
    for L in range(1, 41):
        out.append((f"single L={L} no-intv", [0] * L))
        for s in range(L):
            out.append((f"single L={L} intv@{s}", [0] * s + [1] * (L - s)))
    return out


def double_episodes():
    out = []
    for a in (0, 1, 3, 15, 16, 20):
        for b in (1, 5):
            for g in range(21):
                for c in (1, 16):
                    for d in (0, 10):
                        m = [0] * a + [1] * b + [0] * g + [1] * c + [0] * d
                        out.append((f"double a={a} b={b} g={g} c={c} d={d}", m))
    return out


# ── 기준 구현 (순수 파이썬) ─────────────────────────────────────────────────

def oracle_labels(mode, is_demo):
    """에피소드 하나의 프레임 라벨 리스트 ('demo' | 'robot' | 'intv' | 'preintv')."""
    if is_demo:
        return ["demo"] * len(mode)
    lab = ["intv" if x == 1 else "robot" for x in mode]
    for o in range(len(mode)):
        starts = mode[o] == 1 and (o == 0 or mode[o - 1] == 0)
        if not starts:
            continue
        j, n = o - 1, 0
        while j >= 0 and n < K_PRE and mode[j] == 0:
            lab[j] = "preintv"
            n += 1
            j -= 1
    return lab


def oracle_table(episodes, n_demo):
    """전역 프레임 순서의 행 리스트. 행 = dict(ep, t, L, demo, n_judge, n_pre_judge, n_pre16, slots).
    slots = 판정 칸의 창 칸 번호 (창 칸 j 는 프레임 t-1+j)."""
    rows = []
    for e, (_, mode) in enumerate(episodes):
        L = len(mode)
        lab = oracle_labels(mode, e < n_demo)
        for t in range(L):
            judge_frames = [f for f in range(t, t + JUDGE_LEN) if f < L]
            slots = [f - (t - 1) for f in judge_frames]
            window16 = [min(max(t + d, 0), L - 1) for d in range(-1, H - 1)]
            rows.append({
                "ep": e, "t": t, "L": L, "demo": e < n_demo,
                "n_judge": len(judge_frames),
                "n_pre_judge": sum(lab[f] == "preintv" for f in judge_frames),
                "n_pre16": sum(lab[f] == "preintv" for f in window16),
                "slots": slots,
            })
    return rows


def oracle_drop(rows, k):
    return np.array([(not r["demo"]) and r["n_pre_judge"] >= k for r in rows], dtype=bool)


def mismatch_report(name, got, want, rows, episodes, extra=None, limit=12):
    """다른 프레임을 (경우, t, oracle 값, 저장소 값) 으로 적는다. 같으면 빈 문자열."""
    got, want = np.asarray(got), np.asarray(want)
    if got.shape != want.shape:
        return f"{name}: 길이 다름 저장소 {got.shape} != oracle {want.shape}"
    bad = np.flatnonzero(got != want)
    if len(bad) == 0:
        return ""
    lines = [f"{name}: {len(bad)} 프레임 다름 (전체 {len(want)}). 처음 {min(limit, len(bad))} 개:"]
    for i in bad[:limit]:
        r = rows[i]
        info = f"n_judge={r['n_judge']} n_pre_judge={r['n_pre_judge']} n_pre16={r['n_pre16']}"
        if extra is not None:
            info += f" {extra(i)}"
        lines.append(f"  frame {i} [{episodes[r['ep']][0]}] t={r['t']} | {info} | "
                     f"oracle={want[i]} 저장소={got[i]}")
    # 어떤 경우들에서 갈리는지 한눈에
    kinds = sorted({episodes[rows[i]["ep"]][0].split(" ")[0] for i in bad})
    lines.append(f"  다른 경우의 종류: {kinds} · 다른 에피소드 수 {len({rows[i]['ep'] for i in bad})}")
    return "\n".join(lines)


# ── 기준 구현 자체의 손계산 확인 ───────────────────────────────────────────

def test_oracle_hand_counted():
    """oracle 이 틀리면 대조가 무의미하다 — 손으로 센 값과 맞춘다."""
    # L=20, 개입 17 에서 시작: preintv = 프레임 2..16 (15 개), 프레임 0·1 은 robot
    lab = oracle_labels([0] * 17 + [1] * 3, False)
    assert [i for i, x in enumerate(lab) if x == "preintv"] == list(range(2, 17))
    assert lab[:2] == ["robot", "robot"] and lab[17:] == ["intv"] * 3
    # 첫 프레임이 intv: preintv 없음
    assert "preintv" not in oracle_labels([1, 1, 0, 0], False)
    # 두 번째 개입 앞 간격 3: 앞선 intv 를 만나 멈춘다 -> 3 개만
    lab = oracle_labels([0] * 2 + [1] * 2 + [0] * 3 + [1], False)
    assert lab == ["preintv"] * 2 + ["intv"] * 2 + ["preintv"] * 3 + ["intv"]
    # 시연은 모드와 무관하게 demo
    assert oracle_labels([0, 1, 1], True) == ["demo"] * 3

    eps = [("demo", [0] * 5), ("x", [0] * 17 + [1] * 3)]
    rows = oracle_table(eps, n_demo=1)
    r = rows[5:]                                  # 배포 에피소드
    # t=0: 판정 프레임 0..14 -> preintv 2..14 = 13 개. 16 칸 창(-1->0 고정, 0..14) -> 0 고정 칸은 robot -> 13
    assert (r[0]["n_judge"], r[0]["n_pre_judge"], r[0]["n_pre16"]) == (15, 13, 13)
    # t=3: 판정 3..17 -> preintv 3..16 = 14. 16 칸 창 2..17 -> preintv 2..16 = 15
    assert (r[3]["n_pre_judge"], r[3]["n_pre16"]) == (14, 15)
    # t=19 (끝 프레임): 판정 칸 1 개(19, intv). 16 칸 창 = 18 + 19 x 15 -> preintv 0
    assert (r[19]["n_judge"], r[19]["n_pre_judge"], r[19]["n_pre16"]) == (1, 0, 0)
    # t=10: 판정 10..19 (10 칸) -> preintv 10..16 = 7. 16 칸 창 9..19 + 19 x 5 -> 9..16 = 8
    assert (r[10]["n_judge"], r[10]["n_pre_judge"], r[10]["n_pre16"]) == (10, 7, 8)
    assert r[10]["slots"] == list(range(1, 11))
    # 길이 1 에피소드: 판정 칸 1 개 (창 1 번 칸)
    assert oracle_table([("one", [0])], 0)[0]["slots"] == [1]
    # drop: 시연은 없음, k=13 이면 t=0 은 drop, k=14 면 아님
    d13, d14 = oracle_drop(rows, 13), oracle_drop(rows, 14)
    assert not d13[:5].any() and d13[5] and not d14[5]


# ── 합성 데이터 · 설정 ──────────────────────────────────────────────────────

def build_zarr(root, episodes):
    rng = np.random.default_rng(0)
    buf = ReplayBuffer.create_from_path(str(root), mode="a")
    acts = []
    for i, (_, m) in enumerate(episodes):
        T = len(m)
        a = rng.standard_normal((T, A_DIM)).astype(np.float32)
        acts.append(a)
        buf.add_episode({
            "observation.state": rng.standard_normal((T, S_DIM)).astype(np.float32),
            "action": a,
            "observation.images.main": np.zeros((T, HW, HW, 3), dtype=np.uint8),
            "observation.images.wrist": np.zeros((T, HW, HW, 3), dtype=np.uint8),
            "episode_index": np.full(T, i, dtype=np.int64),
            "timestamp": np.arange(T, dtype=np.float32) / FPS,
            "action_mode": np.asarray(m, dtype=np.int64),
        })
    a = np.concatenate(acts)
    st_a = {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
            "min": a.min(0).tolist(), "max": a.max(0).tolist()}
    st_s = {"mean": [0.0] * S_DIM, "std": [1.0] * S_DIM, "min": [-3.0] * S_DIM, "max": [3.0] * S_DIM}
    imgs = ["observation.images.main", "observation.images.wrist"]
    json.dump({
        "repo_id": None, "stats": {"observation.state": st_s, "action": st_a},
        "num_frames": int(buf.n_steps), "num_episodes": len(episodes),
        "features": {"observation.state": {"dtype": "float32", "shape": [S_DIM]},
                     "action": {"dtype": "float32", "shape": [A_DIM]},
                     **{k: {"dtype": "image", "shape": [HW, HW, 3]} for k in imgs}},
        "camera_keys": imgs, "video_keys": [], "image_keys": imgs,
        "fps": FPS, "tasks": {0: "fake"},
    }, open(Path(root) / "config.json", "w"))


def overrides(root, out, extra=None):
    ov = {
        "task": "piper_cube_stack", "policy": "diffusion", "device": "cpu",
        "task.dataset_root": str(root), "task.dataset_repo_id": "null", "task.fps": FPS,
        "resize_shape": f"[{HW},{HW}]", "crop_shape": f"[{HW - 4},{HW - 4}]",
        "policy.unet.down_dims": "[32,64]",
        "train.batch_size": 4, "train.num_workers": 0, "train.use_amp": "false",
        "wandb.enable": "false", "base_dir": str(out), "output_dir": str(Path(out) / "run"),
    }
    ov.update(extra or {})
    return [f"{k}={v}" for k, v in ov.items() if v is not None]


def compose_cfg(ov):
    with initialize_config_dir(config_dir=CFG_DIR, version_base="1.3"):
        return compose(config_name="default_policy", overrides=ov)


def policy_dataset(cfg):
    meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, meta)
    torch.manual_seed(0)
    policy, pre, _ = make_policy(cfg, meta, stats)
    return policy, pre, create_dataset(policy, cfg)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("oracle")
    demos = demo_episodes()
    episodes = demos + single_episodes() + double_episodes()
    n_demo = len(demos)
    root = tmp / "ds"
    build_zarr(root, episodes)
    rows = oracle_table(episodes, n_demo)
    ep = np.array([r["ep"] for r in rows])
    return {"tmp": tmp, "root": root, "episodes": episodes, "n_demo": n_demo, "rows": rows, "ep": ep,
            "builds": {}}


def test_world_covers_requested_cases(world):
    """경우 목록이 요청 범위를 다 담았는지 — 길이 1..40 x 시작 위치(없음 · 0..L-1) · 개입 두 번 간격 0..20."""
    names = {n for n, _ in world["episodes"]}
    for L in range(1, 41):
        assert f"single L={L} no-intv" in names
        assert all(f"single L={L} intv@{s}" in names for s in range(L))
    gaps = {int(n.split("g=")[1].split()[0]) for n in names if n.startswith("double")}
    assert gaps == set(range(21))
    assert len(world["rows"]) == sum(len(m) for _, m in world["episodes"])


def run_build(world, k):
    """build_sample_weights 를 k 마다 한 번 (모듈 안에서 재사용). 점수 파일은 합성 — 새 프레임 전부에 양수 L1."""
    if k in world["builds"]:
        return world["builds"][k]
    tmp, ep, n_demo = world["tmp"], world["ep"], world["n_demo"]
    sc = tmp / "scores_syn.npz"
    if not sc.exists():
        idx = np.flatnonzero(ep >= n_demo)
        l1 = np.random.default_rng(1).gamma(2.0, 0.05, len(idx)).astype(np.float32) + 1e-3
        np.savez(sc, index=idx, episode_index=ep[idx], l1=l1[:, None], l1_mean=l1, l1_min=l1,
                 n_valid_slots=np.array([world["rows"][i]["n_judge"] for i in idx]), judge=JUDGE)
    out = tmp / f"weights_k{k}.npz"
    cfg = compose_cfg(overrides(world["root"], tmp, {
        "+scores": sc, "+out": out, "+n_demo_episodes": n_demo, "+drop_preintv_min": k}))
    bsw.main.__wrapped__(cfg)
    with np.load(out) as d:
        world["builds"][k] = {key: d[key] for key in d.files}
    return world["builds"][k]


# ── (a) · (b) build_sample_weights ─────────────────────────────────────────

@pytest.mark.parametrize("k", K_VALUES)
def test_a_build_drop_and_zero_weight_match_oracle(world, k):
    W = run_build(world, k)
    rows, eps = world["rows"], world["episodes"]
    want = oracle_drop(rows, k)
    assert np.array_equal(W["episode_index"], world["ep"]), "가중 파일의 에피소드 번호가 데이터와 다르다"
    msgs = [mismatch_report(f"(a) k={k} drop", W["drop"].astype(bool), want, rows, eps),
            mismatch_report(f"(a) k={k} w==0", W["w"] == 0, want, rows, eps,
                            extra=lambda i: f"w={W['w'][i]:.4f}")]
    msgs = [m for m in msgs if m]
    assert not msgs, "\n".join(msgs)
    assert int(W["drop_preintv_min"]) == k and str(W["judge"]) == JUDGE


@pytest.mark.parametrize("k", K_VALUES)
def test_b_build_demo_weight_is_one(world, k):
    W = run_build(world, k)
    demo = world["ep"] < world["n_demo"]
    bad = np.flatnonzero(demo & (W["w"] != 1))
    assert len(bad) == 0, f"(b) k={k} 시연 프레임 {len(bad)} 개의 w != 1 (예: 프레임 {bad[:5].tolist()} " \
                          f"w={W['w'][bad[:5]].tolist()})"
    assert not W["drop"][demo].any()


# ── (c) train.py sample_weight 의 sampler 제외 집합 ────────────────────────

def write_oracle_weights(path, world, k):
    """oracle 의 drop 으로 가중 파일을 쓴다 — build 와 독립적으로 train.py 의 재계산만 본다."""
    ep, n_demo = world["ep"], world["n_demo"]
    drop = oracle_drop(world["rows"], k)
    w = np.ones(len(ep), dtype=np.float32)
    ok = (ep >= n_demo) & ~drop
    r = np.random.default_rng(2).uniform(0.4, 3.0, ok.sum()).astype(np.float32)
    w[ok] = r / r.mean()
    w[drop] = 0.0
    np.savez(path, w=w, drop=drop, episode_index=ep, n_prev=0, n_new=int((ep >= n_demo).sum()),
             n_new_pi=int(drop.sum()), w_new_quantiles=np.zeros(7), clip_lo_frac=0.0, clip_hi_frac=0.0,
             lo=0.4, hi=3.0, drop_preintv_min=k, n_demo_episodes=n_demo, judge=JUDGE,
             scores="", prev="", dataset_root=str(world["root"]))
    return path


def trainer_kept(world, extra, tmp):
    cfg = compose_cfg(overrides(world["root"], tmp, {
        "finetune.enabled": "true", "finetune.balanced": "null",
        "finetune.n_demo_episodes": world["n_demo"], "+policy.drop_n_last_frames": 0, **extra}))
    policy, pre, ds = policy_dataset(cfg)
    dl = create_dataloader(ds, cfg, is_training=True)
    before = list(dl.sampler.indices)
    tr = PolicyTrainer(cfg, policy, device="cpu", train_dataloader=dl, preprocessor=pre)
    return np.asarray(before), np.asarray(tr.train_dataloader.sampler.indices), len(ds)


@pytest.mark.parametrize("source", ["oracle_file", "build_file"])
@pytest.mark.parametrize("k", K_VALUES)
def test_c_train_sampler_excluded_set_matches_oracle(world, k, source, tmp_path):
    if source == "oracle_file":
        wpath = write_oracle_weights(tmp_path / "w.npz", world, k)
    else:
        run_build(world, k)
        wpath = world["tmp"] / f"weights_k{k}.npz"
    before, kept, N = trainer_kept(world, {"finetune.loss": "sample_weight",
                                           "finetune.sample_weights": wpath,
                                           "finetune.sirius_drop_preintv_min": k}, tmp_path)
    assert np.array_equal(np.sort(before), np.arange(N)), "drop_n_last_frames=0 인데 sampler 가 전 프레임이 아니다"
    excluded = np.ones(N, dtype=bool)
    excluded[kept] = False
    msg = mismatch_report(f"(c) k={k} {source} sampler 제외", excluded, oracle_drop(world["rows"], k),
                          world["rows"], world["episodes"])
    assert not msg, msg
    assert len(kept) == len(set(kept.tolist())), "sampler 에 같은 인덱스가 여러 번 있다"


# ── (d) score_samples 의 유효 칸 ────────────────────────────────────────────

SLOT_OFFSET = 0.01 * (np.arange(H) + 1.0) ** 2     # 창 칸 j 의 오프셋 — 칸 집합이 다르면 평균이 달라진다


@pytest.fixture(scope="module")
def scores(world):
    tmp = world["tmp"]
    cfg0 = compose_cfg(overrides(world["root"], tmp))
    policy, _, _ = policy_dataset(cfg0)
    ckpt = tmp / "ckpt"
    ckpt.mkdir()
    policy.save_pretrained(ckpt)
    torch.save({"ema": {"shadow_params": [p.detach().clone() for p in policy.parameters()]}, "step": 0},
               ckpt / "training_state.pt")

    state = {}
    offs = torch.tensor(SLOT_OFFSET, dtype=torch.float32)

    def fake_prep(policy, batch):
        # 이미지 인코더를 건너뛴다 — 유효 칸은 조건과 무관하다. x0 = 정규화된 기록 행동 (원래와 같은 키)
        x0 = batch["action"]
        state["x0"] = x0
        return torch.zeros(x0.shape[0], 1, dtype=x0.dtype), x0

    def fake_cs(model, batch_size, global_cond=None, generator=None, noise=None):
        x0 = state["x0"]
        M = batch_size // x0.shape[0]
        return x0.repeat_interleave(M, 0) + offs.to(x0)[None, :, None]

    mp = pytest.MonkeyPatch()
    mp.setattr(ss, "prepare_cond", fake_prep)
    mp.setattr(DiffusionModel, "conditional_sample", fake_cs)
    try:
        out = tmp / "scores.npz"
        cfg = compose_cfg(overrides(world["root"], tmp, {
            "+checkpoint": ckpt, "+out": out, "+start_index": 0, "+M": 1, "+batch_size": 512,
            "+noise_seed": 0}))
        logging.disable(logging.INFO)
        ss.main.__wrapped__(cfg)
    finally:
        logging.disable(logging.NOTSET)
        mp.undo()
    with np.load(out) as d:
        return {key: d[key] for key in d.files}


def test_d_score_n_valid_slots_match_oracle(world, scores):
    rows, eps = world["rows"], world["episodes"]
    assert np.array_equal(scores["index"], np.arange(len(rows))), "채점 범위가 전 프레임이 아니다"
    want = np.array([r["n_judge"] for r in rows])
    msg = mismatch_report("(d) n_valid_slots", scores["n_valid_slots"], want, rows, eps)
    assert not msg, msg


def test_d_score_l1_averages_exactly_the_judge_slots(world, scores):
    """생성 = 라벨 + 칸 j 의 오프셋 c_j (증가 수열) -> L1 = 판정 칸의 c_j 평균. t-1 칸(0 번)이나 패딩 칸이
    섞이거나 칸이 하나 밀리면 평균이 달라진다."""
    rows, eps = world["rows"], world["episodes"]
    want = np.array([SLOT_OFFSET[r["slots"]].mean() for r in rows])
    got = scores["l1_mean"].astype(np.float64)
    close = np.isclose(got, want, rtol=0, atol=1e-5)
    msg = mismatch_report("(d) L1 의 칸 집합", ~close, np.zeros(len(rows), dtype=bool), rows, eps,
                          extra=lambda i: f"L1 저장소={got[i]:.6f} oracle={want[i]:.6f}")
    assert not msg, msg


# ── (e) loss=sirius 는 16 칸 셈 그대로 ─────────────────────────────────────

@pytest.mark.parametrize("k", K_VALUES)
def test_e_sirius_sampler_still_counts_16_slots(world, k, tmp_path):
    before, kept, N = trainer_kept(world, {"finetune.loss": "sirius",
                                           "finetune.sirius_drop_preintv_min": k}, tmp_path)
    assert np.array_equal(np.sort(before), np.arange(N))
    excluded = np.ones(N, dtype=bool)
    excluded[kept] = False
    want = np.array([r["n_pre16"] >= k for r in world["rows"]], dtype=bool)
    msg = mismatch_report(f"(e) k={k} sirius sampler 제외(16 칸)", excluded, want,
                          world["rows"], world["episodes"])
    assert not msg, msg
