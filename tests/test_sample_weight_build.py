"""관점 A — 데이터 경계와 가중 규칙 (build_sample_weights · frame_labels · action_windows).

기대값은 '의도'에서 온다 — 저장소 코드가 아니라 아래 ref_* (numpy 만으로 다시 쓴 규칙)와
손으로 센 값에 맞춘다. 저장소 코드를 기대값으로 쓰면 같은 실수를 같이 통과한다.

  의도 (1) 시연 w=1 · (2) 새 비-PI 평균 w=1, L1 이 클수록 w 가 크거나 같다
       (4) prev 범위는 그대로 고정 · (8) 잘못된 입력은 분명한 에러 (ValueError · AssertionError)

PI 판정 정의 (ref_pi) [사용자 결정 2026-10-09]: 판정 칸 = 프레임 i .. i+14 중 같은 에피소드 안인 칸
(행동 창 i-1 .. i+14 에서 t-1 칸과 에피소드 밖을 경계 프레임으로 채운 패딩 칸을 뺀 것, 최대 15칸) 중
preintv 가 drop_preintv_min 칸 이상. preintv = 개입 시작(앞 프레임이 개입 아님, 프레임 0 제외)
직전 15 프레임 중 정책(0) 프레임. (행동 창 자체는 16칸 그대로 — action_windows 테스트)

실행 (CPU):
    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_sample_weight_build.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from hydra import compose, initialize_config_dir

import manibot.scripts.build_sample_weights as bsw
from manibot.datasets.replay_buffer import ReplayBuffer
from manibot.losses.sirius import action_windows, frame_labels

REPO = Path(__file__).resolve().parents[1]
CFG_DIR = str(REPO / "src" / "manibot" / "configs")
S, A, HW, FPS = 7, 7, 32, 20
K_PRE, H = 15, 16
JUDGE = "t..t+14,no_pad"     # 점수·가중 파일에 남는 판정 규약
DEMO, ROBOT, INTV, PRE = -1, 0, 1, -10
# 분명한 에러 = 코드가 의도해서 낸 검증 에러. IndexError·TypeError 같은 우연한 예외는 아니다
CLEAR = (ValueError, AssertionError)

REAL_R1 = REPO / "data" / "square_ph50_r1_zarr"
REAL_R2 = REPO / "data" / "square_ph50_r2_zarr"


# ── 합성 데이터 ────────────────────────────────────────────────────────────

def seg(*parts):
    """seg((0, 3), (1, 5)) -> [0,0,0,1,1,1,1,1]."""
    return np.concatenate([np.full(n, v, dtype=np.int64) for v, n in parts])


def make_zarr(root, modes):
    """convert 와 같은 모양(zarr + config.json)의 작은 데이터셋. 에피소드 내용은 번호로 시드해
    같은 번호면 같은 내용 — 라운드 데이터셋(r1 ⊂ r2 ⊂ r3)의 앞부분이 실제로 같게."""
    root = Path(root)
    shutil.rmtree(root, ignore_errors=True)
    buf = ReplayBuffer.create_from_path(str(root), mode="a")
    for i, m in enumerate(modes):
        m = np.asarray(m, dtype=np.int64)
        T = len(m)
        rng = np.random.default_rng(1000 + i)
        buf.add_episode({
            "observation.state": rng.standard_normal((T, S)).astype(np.float32),
            "action": rng.standard_normal((T, A)).astype(np.float32),
            "observation.images.main": np.zeros((T, HW, HW, 3), dtype=np.uint8),
            "observation.images.wrist": np.zeros((T, HW, HW, 3), dtype=np.uint8),
            "episode_index": np.full(T, i, dtype=np.int64),
            "timestamp": np.arange(T, dtype=np.float32) / FPS,
            "action_mode": m,
        })
    st = {"mean": [0.0] * S, "std": [1.0] * S, "min": [-1.0] * S, "max": [1.0] * S}
    imgs = ["observation.images.main", "observation.images.wrist"]
    json.dump({
        "repo_id": None, "stats": {"observation.state": st, "action": st},
        "num_frames": int(buf.n_steps), "num_episodes": len(modes),
        "features": {"observation.state": {"dtype": "float32", "shape": [S]},
                     "action": {"dtype": "float32", "shape": [A]},
                     **{k: {"dtype": "image", "shape": [HW, HW, 3]} for k in imgs}},
        "camera_keys": imgs, "video_keys": [], "image_keys": imgs,
        "fps": FPS, "tasks": {0: "fake"},
    }, open(root / "config.json", "w"))
    ep = np.concatenate([np.full(len(m), i, dtype=np.int64) for i, m in enumerate(modes)])
    return ep, np.concatenate([np.asarray(m, dtype=np.int64) for m in modes])


def save_scores(path, ep, index, l1_mean, dataset_root="synthetic", M=2):
    """score_samples 와 같은 필드의 점수 파일."""
    index = np.asarray(index, dtype=np.int64)
    l1_mean = np.asarray(l1_mean, dtype=np.float32)
    l1 = np.repeat(l1_mean[:, None], M, 1)
    np.savez(path, index=index, episode_index=np.asarray(ep)[np.clip(index, 0, len(ep) - 1)],
             l1=l1, l1_mean=l1_mean, l1_min=l1_mean, n_valid_slots=np.full(len(index), H),
             checkpoint="synthetic", M=M, seed=0, num_inference_steps=10,
             dataset_root=str(dataset_root), judge=JUDGE)
    return path


def lognormal_l1(n, seed):
    # 평균 대비 비가 lo(0.4) 아래·hi(3.0) 위로 둘 다 나오게 퍼진 분포
    return (0.05 * np.exp(1.2 * np.random.default_rng(seed).standard_normal(n))).astype(np.float32)


def compose_cfg(root, extra=(), task="piper_cube_stack"):
    ov = [f"task={task}", "policy=diffusion", "device=cpu", f"task.dataset_root={root}",
          "task.dataset_repo_id=null", "policy.unet.down_dims=[64,128]"]
    if task == "piper_cube_stack":
        ov += [f"task.fps={FPS}", f"resize_shape=[{HW},{HW}]", f"crop_shape=[{HW - 4},{HW - 4}]"]
    with initialize_config_dir(config_dir=CFG_DIR, version_base="1.3"):
        return compose(config_name="default_policy", overrides=ov + list(extra))


def run_build(root, scores, out, n_demo, prev=None, task="piper_cube_stack", **kw):
    """build_sample_weights.main 을 진입점 그대로(Hydra cfg) 부른다. kw -> +lo= · +hi= · +drop_preintv_min=."""
    extra = [f"+scores={scores}", f"+out={out}"]
    if n_demo is not None:
        extra.append(f"+n_demo_episodes={n_demo}")
    if prev is not None:
        extra.append(f"+prev={prev}")
    extra += [f"+{k}={v}" for k, v in kw.items()]
    bsw.main(compose_cfg(root, extra, task))
    with np.load(out) as d:
        return {k: d[k] for k in d.files}


def build_or_clear_error(*a, **kw):
    """명세가 정하지 않은 경계용 — 분명한 에러면 None, 아니면 결과. 다른 예외는 그대로 올라가 실패."""
    try:
        return run_build(*a, **kw)
    except CLEAR:
        return None


def expect_clear_error(out, *a, **kw):
    with pytest.raises(CLEAR):
        run_build(*a, out=out, **kw)
    # 에러인데 결과 파일이 남으면 다음 단계가 그걸 집어 쓴다
    assert not Path(out).exists(), "에러가 났는데 가중 파일이 저장됐다"


def make_dataset(root):
    from manibot.policies.factory import make_policy
    from manibot.utils.dataset_utils import create_dataset, create_dataset_stats
    from manibot.utils.task_utils import derive_task_meta
    cfg = compose_cfg(root)
    meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, meta)
    policy, _, _ = make_policy(cfg, meta, stats)
    return create_dataset(policy, cfg)


# ── 기대값: 규칙을 numpy 로 다시 쓴 것 ─────────────────────────────────────

def ref_labels(ep, mode, n_demo):
    fl = np.asarray(mode, dtype=np.int64).copy()
    for e in np.unique(ep):
        s = np.flatnonzero(ep == e)
        if e < n_demo:
            fl[s] = DEMO
            continue
        m = mode[s]
        for o in range(1, len(m)):
            if m[o] == INTV and m[o - 1] != INTV:
                j = np.arange(max(0, o - K_PRE), o)
                fl[s[j[m[j] == ROBOT]]] = PRE
    return fl


def ref_windows(ep):
    f = np.arange(len(ep))
    first, last = np.empty_like(f), np.empty_like(f)
    for e in np.unique(ep):
        s = np.flatnonzero(ep == e)
        first[s], last[s] = s[0], s[-1]
    return np.clip(f[:, None] + np.arange(-1, H - 1)[None], first[:, None], last[:, None])


def ref_judge(ep):
    """(N, H) 판정 칸: 창 칸 j 는 프레임 i-1+j. 프레임 i 이후(j >= 1)이고 에피소드 끝 안인 칸."""
    f = np.arange(len(ep))
    last = np.empty_like(f)
    for e in np.unique(ep):
        s = np.flatnonzero(ep == e)
        last[s] = s[-1]
    raw = f[:, None] + np.arange(-1, H - 1)[None]
    return (raw >= f[:, None]) & (raw <= last[:, None])


def ref_pi(ep, mode, n_demo, k=4):
    return ((ref_labels(ep, mode, n_demo)[ref_windows(ep)] == PRE) & ref_judge(ep)).sum(1) >= k


def ref_weights(ep, mode, n_demo, l1, k=4, lo=0.4, hi=3.0, prev=None):
    """l1: (N,) 프레임별 l1_mean (없는 칸은 NaN)."""
    N = len(ep)
    pi = ref_pi(ep, mode, n_demo, k)
    w, drop = np.ones(N), np.zeros(N, dtype=bool)
    n_prev = 0
    if prev is not None:
        n_prev = len(prev["w"])
        w[:n_prev], drop[:n_prev] = prev["w"], prev["drop"]
    new = (ep >= n_demo) & (np.arange(N) >= n_prev)
    w[new & pi], drop[new & pi] = 0.0, True
    ok = new & ~pi
    r_raw = l1[ok] / l1[ok].mean()
    r = np.clip(r_raw, lo, hi)
    w[ok] = r / r.mean()
    return SimpleNamespace(w=w, drop=drop, pi=pi, new=new, ok=ok, n_prev=n_prev,
                           clip_lo=float((r_raw < lo).mean()), clip_hi=float((r_raw > hi).mean()))


# ── 기본 데이터셋: 경계를 한 데 모은 시연 2 + 배포 10 ──────────────────────

N_DEMO = 2
DEMO_MODES = [seg((0, 20)), seg((0, 10), (1, 8))]   # 둘째 시연은 action_mode 에 1 이 있어도 시연
DEPLOY_MODES = {
    "a": seg((0, 3), (1, 5), (0, 22)),     # 시연 바로 뒤 · 개입 시작 3 (preintv 창이 잘림)
    "b": seg((0, 40)),                     # 개입 없는 배포
    "c": seg((1, 25)),                     # 개입만 (첫 프레임부터)
    "d": seg((0, 20), (1, 8), (0, 6), (1, 6), (0, 10)),   # 개입 두 번, 사이 정책 6 프레임
    "e": seg((0, 7), (1, 3)),              # 길이 10 < 16
    "f": seg((0, 1)),                      # 길이 1 (정책)
    "g": seg((1, 1)),                      # 길이 1 (개입)
    "h": seg((0, 23), (1, 1)),             # 마지막 프레임에 개입 시작
    "i": seg((0, 14), (1, 3)),             # 개입 시작 14
    "j": seg((1, 5), (0, 15)),             # 첫 프레임 개입 후 정책
}
EP_OF = {name: N_DEMO + n for n, name in enumerate(DEPLOY_MODES)}


@pytest.fixture(scope="module")
def basic(tmp_path_factory):
    d = tmp_path_factory.mktemp("basic")
    root = d / "ds"
    ep, mode = make_zarr(root, DEMO_MODES + list(DEPLOY_MODES.values()))
    N = len(ep)
    first_new = int((ep < N_DEMO).sum())
    idx = np.arange(first_new, N)
    l1 = np.full(N, np.nan)
    l1[idx] = lognormal_l1(len(idx), seed=0)
    scores = save_scores(d / "scores.npz", ep, idx, l1[idx], dataset_root=root)
    res = run_build(root, scores, d / "w.npz", N_DEMO)
    ref = ref_weights(ep, mode, N_DEMO, l1)
    # 전제: 이 데이터셋이 clip 양쪽 · PI · 비-PI 를 모두 지나간다
    assert ref.clip_lo > 0 and ref.clip_hi > 0 and ref.pi.any() and ref.ok.sum() > 50
    return SimpleNamespace(dir=d, root=root, ep=ep, mode=mode, N=N, l1=l1, idx=idx,
                           first_new=first_new, scores=scores, res=res, ref=ref)


def local(b, name):
    return np.flatnonzero(b.ep == EP_OF[name])


# ── frame_labels ───────────────────────────────────────────────────────────

def test_frame_labels_match_rule(basic):
    """모든 프레임 라벨이 규칙(ref_labels)과 같고, 에피소드 번호는 zarr 그대로."""
    fl, ep = frame_labels(basic.root, N_DEMO)
    assert np.array_equal(ep, basic.ep)
    assert np.array_equal(fl, ref_labels(basic.ep, basic.mode, N_DEMO))


def test_frame_labels_hand_counted(basic):
    """손으로 센 라벨 — ref_labels 자체가 틀렸을 때를 대비한 고정 기대값."""
    fl, _ = frame_labels(basic.root, N_DEMO)
    expect = {
        "a": [PRE] * 3 + [INTV] * 5 + [ROBOT] * 22,
        "b": [ROBOT] * 40,
        "c": [INTV] * 25,
        "d": [ROBOT] * 5 + [PRE] * 15 + [INTV] * 8 + [PRE] * 6 + [INTV] * 6 + [ROBOT] * 10,
        "e": [PRE] * 7 + [INTV] * 3,
        "f": [ROBOT], "g": [INTV],
        "h": [ROBOT] * 8 + [PRE] * 15 + [INTV],
        "i": [PRE] * 14 + [INTV] * 3,
        "j": [INTV] * 5 + [ROBOT] * 15,
    }
    for name, lab in expect.items():
        assert fl[local(basic, name)].tolist() == lab, name
    # 시연은 action_mode 에 개입(1)이 있어도 전부 demo
    assert (fl[basic.ep < N_DEMO] == DEMO).all()


@pytest.mark.parametrize("onset", [0, 1, 2, 3, 7, 14, 15, 16, 29])
def test_frame_labels_onset_position(tmp_path, onset):
    """개입 시작이 0~14 면 preintv 가 [0, onset) 로 잘리고, 15 이상이면 정확히 15 프레임.
    onset 0 (첫 프레임부터 개입) 은 preintv 없음, 29 는 마지막 프레임."""
    mode = seg((0, onset), (1, 30 - onset)) if onset else seg((1, 30))
    make_zarr(tmp_path / "ds", [seg((0, 5)), mode])
    fl, _ = frame_labels(tmp_path / "ds", 1)
    n_pre = min(onset, K_PRE)
    expect = [DEMO] * 5 + [ROBOT] * (onset - n_pre) + [PRE] * n_pre + [INTV] * (30 - onset)
    assert fl.tolist() == expect


def test_frame_labels_do_not_cross_episodes(tmp_path):
    """다음 에피소드가 개입으로 시작해도 앞 에피소드(시연·배포) 끝이 preintv 로 바뀌지 않는다."""
    make_zarr(tmp_path / "ds", [seg((0, 20)), seg((1, 10), (0, 10)), seg((0, 20)), seg((1, 5), (0, 5))])
    fl, _ = frame_labels(tmp_path / "ds", 1)
    assert fl[:20].tolist() == [DEMO] * 20
    assert fl[20:40].tolist() == [INTV] * 10 + [ROBOT] * 10
    assert fl[40:60].tolist() == [ROBOT] * 20
    assert fl[60:70].tolist() == [INTV] * 5 + [ROBOT] * 5


def test_frame_labels_without_preintv(basic):
    """use_preintv=False 면 배포 프레임은 action_mode 그대로, 시연은 demo."""
    fl, _ = frame_labels(basic.root, N_DEMO, use_preintv=False)
    expect = np.where(basic.ep < N_DEMO, DEMO, basic.mode)
    assert np.array_equal(fl, expect)


def test_frame_labels_n_demo_covers_all(basic):
    """n_demo 가 에피소드 수 이상이면 전부 demo (정의된 값)."""
    fl, _ = frame_labels(basic.root, 100)
    assert (fl == DEMO).all()


def test_frame_labels_reject_unknown_action_mode(tmp_path):
    """action_mode 는 0(정책)·1(개입)만 — 다른 값(2)은 라벨 체계(LABELS)에 없는 클래스가 되어
    조용히 '새 비-PI' 로 가중된다. 분명한 에러를 기대한다."""
    make_zarr(tmp_path / "ds", [seg((0, 5)), seg((0, 10), (2, 3), (0, 5))])
    with pytest.raises(CLEAR):
        frame_labels(tmp_path / "ds", 1)


# ── action_windows ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def basic_windows(basic):
    ds = make_dataset(basic.root)
    return ds, action_windows(ds, basic.ep)


def test_action_windows_match_rule(basic, basic_windows):
    """(N, 16), 샘플 i 의 창 = i-1 .. i+14 를 자기 에피소드 안으로 고정."""
    ds, win = basic_windows
    assert win.shape == (basic.N, H) and len(ds) == basic.N
    assert np.array_equal(win, ref_windows(basic.ep))


def test_action_windows_never_cross_episodes(basic, basic_windows):
    _, win = basic_windows
    assert (basic.ep[win] == basic.ep[:, None]).all()


def test_action_windows_short_episodes(basic, basic_windows):
    """길이 1 이면 16칸이 전부 그 프레임, 길이 10 이면 앞은 첫 프레임·뒤는 끝 프레임으로 채운다."""
    _, win = basic_windows
    (f,) = local(basic, "f")
    assert win[f].tolist() == [f] * H
    s = local(basic, "e")[0]
    assert win[s].tolist() == [s] * 2 + list(range(s + 1, s + 10)) + [s + 9] * 5
    assert win[s + 9].tolist() == [s + 8] + [s + 9] * 15


# ── 기본 데이터셋의 가중 ───────────────────────────────────────────────────

def test_demo_weight_is_one(basic):
    """의도 (1): 시연 프레임은 w=1 · drop 아님 (action_mode 에 1 이 있는 시연 포함)."""
    demo = basic.ep < N_DEMO
    assert (basic.res["w"][demo] == 1).all()
    assert not basic.res["drop"][demo].any()


def test_new_pi_dropped_with_zero_weight(basic):
    """새 PI 샘플은 w=0 · drop, 그 밖의 샘플은 drop 아님."""
    w, drop = basic.res["w"], basic.res["drop"]
    pi = basic.ref.new & basic.ref.pi
    assert np.array_equal(drop, pi)
    assert (w[pi] == 0).all()
    assert (w[~pi] > 0).all()


def test_pi_hand_counted(basic):
    """손으로 센 PI 샘플 (k=4, 에피소드 안 위치).  판정 칸 = 프레임 t..t+14 중 에피소드 안.
    'a' 의 프레임 0 은 판정 칸(0..14)에 preintv 가 0·1·2 의 3칸뿐이라 PI 아님 — t-1 칸(고정된 복사 칸)은 세지 않는다.
    'e'(preintv 0..6, 길이 10)는 7-t >= 4 인 t=0..3 · 'h'(preintv 8..22, 끝 23)는 t<=8 이면 t+7 칸, 그 뒤 23-t 칸
    -> t=0..19 · 'i'(preintv 0..13)는 14-t >= 4 인 t=0..10."""
    drop = basic.res["drop"]
    expect = {"a": [], "b": [], "c": [], "e": [0, 1, 2, 3], "f": [], "g": [],
              "h": list(range(20)), "i": list(range(11)), "j": []}
    for name, pis in expect.items():
        assert np.flatnonzero(drop[local(basic, name)]).tolist() == pis, name


def test_drop_equals_pi_over_whole_dataset(basic):
    """drop 은 데이터셋 전체에서 PI 판정과 같다 — 학습(train.py)이 같은 판정으로 배치에서 빼므로."""
    assert np.array_equal(basic.res["drop"], basic.ref.pi)


def test_new_nonpi_mean_is_one(basic):
    """의도 (2): 새 비-PI 샘플의 w 평균 = 1."""
    w = basic.res["w"][basic.ref.ok].astype(np.float64)
    assert abs(w.mean() - 1) < 1e-5


def test_weight_monotonic_in_l1(basic):
    """의도 (2): L1 이 클수록 w 가 크거나 같다 (clip 구간에선 같다)."""
    ok = basic.ref.ok
    order = np.argsort(basic.l1[ok], kind="stable")
    assert (np.diff(basic.res["w"][ok][order].astype(np.float64)) >= 0).all()


def test_weight_rule_clip_then_normalize(basic):
    """w = clip(L1/평균, 0.4, 3.0) / 그 평균. clip 안쪽 샘플끼리는 w 비 = L1 비."""
    ok = basic.ref.ok
    w = basic.res["w"].astype(np.float64)
    assert np.allclose(w, basic.ref.w, rtol=1e-6, atol=0)
    r = basic.l1[ok] / basic.l1[ok].mean()
    inner = (r > 0.4) & (r < 3.0)
    ratio = w[ok][inner] / basic.l1[ok][inner]
    assert np.allclose(ratio, ratio[0], rtol=1e-5)
    # clip 바깥은 양 끝값 하나씩으로 모인다
    assert np.allclose(w[ok][r <= 0.4], w[ok][r <= 0.4].min(), rtol=1e-6)
    assert np.allclose(w[ok][r >= 3.0], w[ok][r >= 3.0].max(), rtol=1e-6)


def test_output_fields(basic):
    """저장 필드: w(float32, N) · drop(bool) · episode_index(zarr 그대로) · 요약 · 입력 경로."""
    r = basic.res
    assert r["w"].dtype == np.float32 and r["w"].shape == (basic.N,)
    assert r["drop"].dtype == bool
    assert np.array_equal(r["episode_index"], basic.ep)
    assert np.isfinite(r["w"]).all()
    assert int(r["n_prev"]) == 0
    assert int(r["n_new"]) == int(basic.ref.new.sum())
    assert int(r["n_new_pi"]) == int((basic.ref.new & basic.ref.pi).sum())
    assert float(r["lo"]) == 0.4 and float(r["hi"]) == 3.0
    assert int(r["drop_preintv_min"]) == 4 and int(r["n_demo_episodes"]) == N_DEMO
    assert str(r["judge"]) == JUDGE
    assert np.isclose(float(r["clip_lo_frac"]), basic.ref.clip_lo)
    assert np.isclose(float(r["clip_hi_frac"]), basic.ref.clip_hi)
    assert str(r["scores"]) == str(basic.scores)
    assert str(r["dataset_root"]) == str(basic.root)


def test_cli_entrypoint_matches(basic, tmp_path):
    """사용 예시대로 CLI(+scores · +n_demo_episodes · +out)로 돌려도 같은 결과."""
    out = tmp_path / "w_cli.npz"
    cmd = [sys.executable, "-m", "manibot.scripts.build_sample_weights",
           "task=piper_cube_stack", "policy=diffusion", "device=cpu",
           f"task.dataset_root={basic.root}", "task.dataset_repo_id=null", f"task.fps={FPS}",
           f"resize_shape=[{HW},{HW}]", f"crop_shape=[{HW - 4},{HW - 4}]",
           "policy.unet.down_dims=[64,128]", f"+scores={basic.scores}",
           f"+n_demo_episodes={N_DEMO}", f"+out={out}", f"hydra.run.dir={tmp_path}"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    d = np.load(out)
    assert np.array_equal(d["w"], basic.res["w"]) and np.array_equal(d["drop"], basic.res["drop"])


# ── 가중 경계: L1 값 ───────────────────────────────────────────────────────

def _scores_with(b, tmp_path, l1_mean, index=None, name="s.npz"):
    index = b.idx if index is None else index
    return save_scores(tmp_path / name, b.ep, index, l1_mean, dataset_root=b.root)


def test_l1_all_equal_gives_one(basic, tmp_path):
    """L1 이 모두 같으면 새 비-PI 전부 w=1."""
    s = _scores_with(basic, tmp_path, np.full(len(basic.idx), 0.07))
    r = run_build(basic.root, s, tmp_path / "w.npz", N_DEMO)
    assert np.allclose(r["w"][basic.ref.ok], 1.0, rtol=1e-6)


def test_l1_all_zero(basic, tmp_path):
    """L1 이 모두 0 — 명세에 없는 경계. 기대: 분명한 에러(ValueError) — 0/0 이라 '평균 대비 비'가
    정의되지 않고, 전부 0 은 채점이 망가졌다는 신호다. 최소 기대: 저장된 w 에 NaN 이 없다
    (에러 대신 '전부 같다'로 보고 w=1 로 정의해도 통과)."""
    s = _scores_with(basic, tmp_path, np.zeros(len(basic.idx)))
    r = build_or_clear_error(basic.root, s, tmp_path / "w.npz", N_DEMO)
    if r is not None:
        assert np.isfinite(r["w"]).all(), "L1 전부 0 에서 w 에 NaN 이 조용히 저장됐다"
        assert np.allclose(r["w"][basic.ref.ok], 1.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_l1_nonfinite_on_nonpi_rejected(basic, tmp_path, bad):
    """의도 (8): 새 비-PI 프레임의 L1 이 NaN·inf 면 분명한 에러, 파일 없음."""
    l1 = basic.l1[basic.idx].copy()
    l1[np.flatnonzero(basic.ref.ok[basic.idx])[3]] = bad
    expect_clear_error(tmp_path / "w.npz", basic.root, _scores_with(basic, tmp_path, l1), n_demo=N_DEMO)


def test_l1_negative_rejected(basic, tmp_path):
    """의도 (8): L1(절댓값 평균)은 음수일 수 없다 — 음수는 clip 하한으로 조용히 올라가 w=lo 가 된다.
    분명한 에러를 기대한다."""
    l1 = basic.l1[basic.idx].copy()
    l1[np.flatnonzero(basic.ref.ok[basic.idx])[3]] = -0.01
    expect_clear_error(tmp_path / "w.npz", basic.root, _scores_with(basic, tmp_path, l1), n_demo=N_DEMO)


def test_l1_nan_on_pi_frame_is_ignored(basic, tmp_path):
    """PI 프레임은 가중을 안 쓰므로 그 점수가 NaN 이어도 결과가 같다 (정의된 값)."""
    l1 = basic.l1[basic.idx].copy()
    l1[np.flatnonzero((basic.ref.new & basic.ref.pi)[basic.idx])[0]] = np.nan
    r = run_build(basic.root, _scores_with(basic, tmp_path, l1), tmp_path / "w.npz", N_DEMO)
    assert np.array_equal(r["w"], basic.res["w"]) and np.array_equal(r["drop"], basic.res["drop"])


def test_single_new_nonpi_gets_one(tmp_path):
    """새 비-PI 가 1 개면 그 w = 1. (k=1: 정책 15 + 개입 1 에피소드는 앞 15 프레임이 PI, 판정 칸이 자기 하나인
    마지막 개입 프레임만 남는다)"""
    ep, mode = make_zarr(tmp_path / "ds", DEMO_MODES + [seg((0, 15), (1, 1))])
    idx = np.arange(int((ep < N_DEMO).sum()), len(ep))
    l1 = np.full(len(ep), np.nan)
    l1[idx] = lognormal_l1(len(idx), 1)
    ref = ref_weights(ep, mode, N_DEMO, l1, k=1)
    assert ref.ok.sum() == 1
    s = save_scores(tmp_path / "s.npz", ep, idx, l1[idx])
    r = run_build(tmp_path / "ds", s, tmp_path / "w.npz", N_DEMO, drop_preintv_min=1)
    assert np.isclose(r["w"][ref.ok][0], 1.0)
    assert np.array_equal(r["drop"], ref.pi)


def test_all_new_pi(basic, tmp_path):
    """새 데이터가 전부 PI 인 라운드는 판정 칸 정의에서 생기지 않는다: 배포 에피소드의 마지막 프레임은 판정 칸이
    자기 하나이고, preintv 뒤에는 항상 같은 에피소드의 개입 프레임이 있어 마지막 프레임은 preintv 가 아니다.
    가장 느슨한 k=1 에서도 모든 배포 에피소드의 마지막 프레임은 drop 이 아니다."""
    last = np.flatnonzero(np.r_[basic.ep[1:] != basic.ep[:-1], True])
    last = last[basic.ep[last] >= N_DEMO]
    assert not ref_pi(basic.ep, basic.mode, N_DEMO, k=1)[last].any()
    r = run_build(basic.root, basic.scores, tmp_path / "w.npz", N_DEMO, drop_preintv_min=1)
    assert not r["drop"][last].any() and (r["w"][last] > 0).all()
    assert np.array_equal(r["drop"], ref_pi(basic.ep, basic.mode, N_DEMO, k=1))


def test_no_new_frames_when_n_demo_covers_all(basic, tmp_path):
    """n_demo 가 에피소드 수 이상이라 새 프레임이 없다 — 명세에 없는 경계. 기대: 정의된 값
    (전부 w=1 · drop 없음). 최소 기대: 우연한 예외·NaN 없이 끝나거나 분명한 에러."""
    r = build_or_clear_error(basic.root, basic.scores, tmp_path / "w.npz", 100)
    if r is not None:
        assert np.isfinite(r["w"]).all()
        assert (r["w"] == 1).all() and not r["drop"].any()


# ── 점수 파일 ──────────────────────────────────────────────────────────────

def test_scores_missing_for_some_nonpi_rejected(basic, tmp_path):
    """의도 (8): 새 비-PI 프레임 일부에 점수가 없으면 분명한 에러 (끝 5 프레임 = 'j' 의 정책 프레임)."""
    assert basic.ref.ok[basic.idx[-5:]].all()
    s = _scores_with(basic, tmp_path, basic.l1[basic.idx[:-5]], index=basic.idx[:-5])
    expect_clear_error(tmp_path / "w.npz", basic.root, s, n_demo=N_DEMO)


def test_scores_only_on_nonpi_frames(basic, tmp_path):
    """점수가 새 비-PI 프레임에만 (띄엄띄엄) 있어도 된다 — PI 프레임 점수는 쓰지 않는다."""
    idx = np.flatnonzero(basic.ref.ok)
    r = run_build(basic.root, _scores_with(basic, tmp_path, basic.l1[idx], index=idx),
                  tmp_path / "w.npz", N_DEMO)
    assert np.array_equal(r["w"], basic.res["w"]) and np.array_equal(r["drop"], basic.res["drop"])


def test_scores_on_demo_frames_do_not_matter(basic, tmp_path):
    """시연까지 채점된 파일(start_index=0)이어도 시연 w=1, 새 비-PI 정규화에 시연 점수가 안 섞인다."""
    idx = np.arange(basic.N)
    l1 = np.where(basic.ep < N_DEMO, 100.0, basic.l1)
    r = run_build(basic.root, _scores_with(basic, tmp_path, l1, index=idx), tmp_path / "w.npz", N_DEMO)
    assert np.array_equal(r["w"], basic.res["w"])


def test_scores_from_larger_dataset_rejected(basic, tmp_path):
    """의도 (8): 다른(더 큰) 데이터셋의 점수 — 인덱스가 이 데이터셋 밖이면 분명한 에러
    (예: r2 점수를 r1 에 쓴 경우). IndexError 는 우연한 예외라 '분명한 에러'가 아니다."""
    idx = np.arange(basic.first_new, basic.N + 20)
    l1 = lognormal_l1(len(idx), 3)
    expect_clear_error(tmp_path / "w.npz", basic.root, _scores_with(basic, tmp_path, l1, index=idx),
                       n_demo=N_DEMO)


def test_scores_from_other_layout_rejected(basic, tmp_path):
    """의도 (8): 에피소드 구성이 다른 데이터셋의 점수 — episode_index 가 안 맞으면 분명한 에러."""
    s = tmp_path / "s.npz"
    save_scores(s, basic.ep, basic.idx, basic.l1[basic.idx])
    d = dict(np.load(s))
    d["episode_index"] = np.roll(d["episode_index"], 1)   # 경계가 한 프레임 밀린 다른 데이터셋
    np.savez(s, **d)
    expect_clear_error(tmp_path / "w.npz", basic.root, s, n_demo=N_DEMO)


@pytest.mark.parametrize("judge", [None, "t-1..t+14,clamp"], ids=["none", "other"])
def test_scores_with_other_judge_rule_rejected(basic, tmp_path, judge):
    """의도 (8): 판정 규약이 없거나(2026-10-09 이전 16칸 L1) 다른 점수 파일은 L1 의 유효 칸이 달라 분명한 에러."""
    s = _scores_with(basic, tmp_path, basic.l1[basic.idx])
    d = dict(np.load(s))
    if judge is None:
        del d["judge"]
    else:
        d["judge"] = judge
    np.savez(s, **d)
    expect_clear_error(tmp_path / "w.npz", basic.root, s, n_demo=N_DEMO)


def test_cross_policy_scores_same_dataset(basic, tmp_path):
    """교차 비교: 다른 정책이 채점한 점수로 같은 데이터셋을 가중해도 규칙은 그대로 —
    시연·PI·drop 은 같고, 새 비-PI 는 각자의 L1 로 평균 1·단조, 어떤 점수를 썼는지 파일에 남는다."""
    l1_b = np.full(basic.N, np.nan)
    l1_b[basic.idx] = lognormal_l1(len(basic.idx), seed=11)
    s_b = _scores_with(basic, tmp_path, l1_b[basic.idx], name="s_b.npz")
    r_b = run_build(basic.root, s_b, tmp_path / "w_b.npz", N_DEMO)
    ref_b = ref_weights(basic.ep, basic.mode, N_DEMO, l1_b)
    assert np.array_equal(r_b["drop"], basic.res["drop"])
    demo_or_pi = (basic.ep < N_DEMO) | basic.ref.pi
    assert np.array_equal(r_b["w"][demo_or_pi], basic.res["w"][demo_or_pi])
    assert np.allclose(r_b["w"], ref_b.w, rtol=1e-6)
    assert not np.allclose(r_b["w"][basic.ref.ok], basic.res["w"][basic.ref.ok])
    assert str(r_b["scores"]) == str(s_b) != str(basic.res["scores"])


# ── 잘못된 인자 ────────────────────────────────────────────────────────────

def test_lo_greater_than_hi_rejected(basic, tmp_path):
    """의도 (8): lo > hi 는 np.clip 이 전부 hi 로 만들어 조용히 w=1 이 된다. 분명한 에러를 기대."""
    expect_clear_error(tmp_path / "w.npz", basic.root, basic.scores, n_demo=N_DEMO, lo=3.0, hi=0.4)


def test_drop_preintv_min_zero_rejected(basic, tmp_path):
    """의도 (8): k=0 이면 모든 샘플이 PI 가 된다 — 학습(train.py)은 k=0 을 '제외 안 함'으로 읽어
    둘의 판정이 갈린다. 분명한 에러를 기대."""
    expect_clear_error(tmp_path / "w.npz", basic.root, basic.scores, n_demo=N_DEMO, drop_preintv_min=0)


def test_missing_n_demo_rejected(basic, tmp_path):
    """의도 (8): n_demo_episodes 를 안 주면(인자·finetune 둘 다 null) 분명한 에러.
    int(None) 의 TypeError 는 우연한 예외다."""
    expect_clear_error(tmp_path / "w.npz", basic.root, basic.scores, n_demo=None)


# ── prev: 라운드 연쇄 ──────────────────────────────────────────────────────

R1_DEPLOY = [DEPLOY_MODES[k] for k in ("a", "b", "d", "e", "f")]
R2_DEPLOY = [DEPLOY_MODES[k] for k in ("h", "i", "j")] + [seg((0, 30), (1, 6), (0, 4))]
R3_DEPLOY = [DEPLOY_MODES[k] for k in ("c", "g")] + [seg((0, 12), (1, 10), (0, 18))]


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """r1 ⊂ r2 ⊂ r3 (앞부분이 같은 누적 데이터셋)과 각 라운드 가중. 점수는 시연 다음부터 끝까지
    — r2·r3 점수는 이전 라운드 범위에도 (다른 값으로) 있다: prev 범위가 고정되는지 보려고."""
    d = tmp_path_factory.mktemp("chain")
    out = SimpleNamespace(dir=d)
    modes = list(DEMO_MODES)
    prev = None
    for r, dep in enumerate([R1_DEPLOY, R2_DEPLOY, R3_DEPLOY], start=1):
        modes = modes + dep
        root = d / f"r{r}"
        ep, mode = make_zarr(root, modes)
        idx = np.arange(int((ep < N_DEMO).sum()), len(ep))
        l1 = np.full(len(ep), np.nan)
        l1[idx] = lognormal_l1(len(idx), seed=100 + r)
        s = save_scores(d / f"s{r}.npz", ep, idx, l1[idx], dataset_root=root)
        w = d / f"w{r}.npz"
        res = run_build(root, s, w, N_DEMO, prev=prev)
        ref = ref_weights(ep, mode, N_DEMO, l1, prev=None if prev is None else dict(np.load(prev)))
        setattr(out, f"r{r}", SimpleNamespace(root=root, ep=ep, mode=mode, N=len(ep), l1=l1, idx=idx,
                                              scores=s, w_path=w, res=res, ref=ref))
        prev = w
    return out


def test_chain_prev_range_frozen(chain):
    """의도 (4): r2 의 앞 n1 프레임 w·drop 은 r1 가중 그대로 (r2 점수가 그 범위를 다르게 매겨도)."""
    n1 = chain.r1.N
    assert int(chain.r2.res["n_prev"]) == n1
    assert np.array_equal(chain.r2.res["w"][:n1], chain.r1.res["w"])
    assert np.array_equal(chain.r2.res["drop"][:n1], chain.r1.res["drop"])


def test_chain_new_range_rule(chain):
    """r2 의 새 범위만 규칙대로 — 평균 1 은 이번 라운드 새 비-PI 끼리 (prev 범위를 섞지 않는다)."""
    r2 = chain.r2
    assert np.allclose(r2.res["w"], r2.ref.w, rtol=1e-6)
    assert abs(r2.res["w"][r2.ref.ok].astype(np.float64).mean() - 1) < 1e-5
    assert int(r2.res["n_new"]) == int(r2.ref.new.sum())


def test_chain_three_rounds(chain):
    """r1 -> r2 -> r3: r3 = [r1 범위 = w1] + [r2 범위 = w2] + [새 범위 = 규칙], 시연은 끝까지 1."""
    n1, n2 = chain.r1.N, chain.r2.N
    w3, d3 = chain.r3.res["w"], chain.r3.res["drop"]
    assert np.array_equal(w3[:n1], chain.r1.res["w"])
    assert np.array_equal(w3[n1:n2], chain.r2.res["w"][n1:])
    assert np.array_equal(d3[:n2], chain.r2.res["drop"])
    assert np.allclose(w3, chain.r3.ref.w, rtol=1e-6)
    assert (w3[chain.r3.ep < N_DEMO] == 1).all()
    assert abs(w3[chain.r3.ref.ok].astype(np.float64).mean() - 1) < 1e-5


def test_chain_drop_equals_pi_over_whole_dataset(chain):
    """r3 의 drop 이 데이터셋 전체 PI 판정과 같다 — 이전 라운드 범위까지 학습 sampler 와 일치."""
    r3 = chain.r3
    assert np.array_equal(r3.res["drop"], ref_pi(r3.ep, r3.mode, N_DEMO))


def test_prev_not_prefix_rejected(chain, tmp_path):
    """의도 (8): prev 데이터셋의 에피소드 구성이 새 데이터셋 앞부분과 다르면 분명한 에러."""
    other = DEMO_MODES + [DEPLOY_MODES["a"], DEPLOY_MODES["d"], DEPLOY_MODES["b"]] + R1_DEPLOY[3:]
    ep, _ = make_zarr(tmp_path / "other", other)
    idx = np.arange(int((ep < N_DEMO).sum()), len(ep))
    s = save_scores(tmp_path / "s.npz", ep, idx, lognormal_l1(len(idx), 5))
    run_build(tmp_path / "other", s, tmp_path / "w_other.npz", N_DEMO)
    expect_clear_error(tmp_path / "w.npz", chain.r2.root, chain.r2.scores, n_demo=N_DEMO,
                       prev=tmp_path / "w_other.npz")


def test_prev_longer_than_dataset_rejected(chain, tmp_path):
    """의도 (8): prev(r2 가중)가 데이터셋(r1)보다 길면 분명한 에러."""
    expect_clear_error(tmp_path / "w.npz", chain.r1.root, chain.r1.scores, n_demo=N_DEMO,
                       prev=chain.r2.w_path)


def test_prev_ending_mid_episode_rejected(chain, tmp_path):
    """의도 (8): prev 가 에피소드 중간에서 끝나면(잘린 파일) 분명한 에러."""
    d = dict(np.load(chain.r1.w_path))
    for k in ("w", "drop", "episode_index"):
        d[k] = d[k][:-3]
    np.savez(tmp_path / "prev_cut.npz", **d)
    expect_clear_error(tmp_path / "w.npz", chain.r2.root, chain.r2.scores, n_demo=N_DEMO,
                       prev=tmp_path / "prev_cut.npz")


def test_prev_k_mismatch_rejected(chain, tmp_path):
    """의도 (8): prev 의 drop_preintv_min(4)과 지금 k(5)가 다르면 분명한 에러."""
    expect_clear_error(tmp_path / "w.npz", chain.r2.root, chain.r2.scores, n_demo=N_DEMO,
                       prev=chain.r1.w_path, drop_preintv_min=5)


def test_prev_n_demo_mismatch_rejected(chain, tmp_path):
    """의도 (8): prev 를 만든 n_demo_episodes(2)와 지금(1)이 다르면 분명한 에러 — 시연/배포 경계가
    라운드마다 달라지면 학습의 frame_labels(finetune.n_demo_episodes)와 가중 파일이 갈린다."""
    expect_clear_error(tmp_path / "w.npz", chain.r2.root, chain.r2.scores, n_demo=1,
                       prev=chain.r1.w_path)


def test_prev_nonfinite_weight_rejected(chain, tmp_path):
    """의도 (8): 손상된 prev(w 에 NaN)를 그대로 복사하면 학습 손실이 NaN 이 된다. 분명한 에러를 기대."""
    d = dict(np.load(chain.r1.w_path))
    d["w"] = d["w"].copy()
    d["w"][-1] = np.nan
    np.savez(tmp_path / "prev_nan.npz", **d)
    expect_clear_error(tmp_path / "w.npz", chain.r2.root, chain.r2.scores, n_demo=N_DEMO,
                       prev=tmp_path / "prev_nan.npz")


def test_prev_without_judge_rule_rejected(chain, tmp_path):
    """의도 (8): 판정 규약이 없는 prev(2026-10-09 이전 16칸 판정) 의 drop 을 옮기면 학습이 다시 센 drop 과 갈린다.
    분명한 에러를 기대."""
    d = dict(np.load(chain.r1.w_path))
    del d["judge"]
    np.savez(tmp_path / "prev_old.npz", **d)
    expect_clear_error(tmp_path / "w.npz", chain.r2.root, chain.r2.scores, n_demo=N_DEMO,
                       prev=tmp_path / "prev_old.npz")


def test_prev_covering_whole_dataset(chain, tmp_path):
    """prev 가 데이터셋 전체(새 프레임 0 개) — 명세에 없는 경계. 기대: 정의된 값 — w·drop 이
    prev 그대로. 최소 기대: 우연한 예외(IndexError 등)·NaN 없이 끝나거나 분명한 에러."""
    r = build_or_clear_error(chain.r1.root, chain.r1.scores, tmp_path / "w.npz", N_DEMO,
                             prev=chain.r1.w_path)
    if r is not None:
        assert np.array_equal(r["w"], chain.r1.res["w"])
        assert np.array_equal(r["drop"], chain.r1.res["drop"])


def test_previous_round_scores_for_next_round_rejected(chain, tmp_path):
    """의도 (8): r1 점수 파일로 r2 를 가중하면 r2 새 프레임에 점수가 없다 — 분명한 에러."""
    expect_clear_error(tmp_path / "w.npz", chain.r2.root, chain.r1.scores, n_demo=N_DEMO,
                       prev=chain.r1.w_path)


# ── 실데이터 (읽기만, 점수는 합성) ─────────────────────────────────────────

needs_real = pytest.mark.skipif(not (REAL_R1.exists() and REAL_R2.exists()),
                                reason="실데이터 data/square_ph50_r{1,2}_zarr 없음")


def _real_arrays(root):
    z = zarr.open(str(root), "r")["data"]
    return np.asarray(z["episode_index"]).ravel(), np.asarray(z["action_mode"]).ravel()


@needs_real
def test_real_r1_is_prefix_of_r2():
    """prev 규칙의 전제: r2 의 앞부분이 r1 과 프레임 단위로 같다 (에피소드 번호·action_mode·행동·상태).
    build 는 에피소드 번호만 대조하므로 내용이 같은지는 여기서 본다."""
    z1, z2 = zarr.open(str(REAL_R1), "r")["data"], zarr.open(str(REAL_R2), "r")["data"]
    n1 = z1["episode_index"].shape[0]
    for k in ("episode_index", "action_mode", "action", "observation.state"):
        assert np.array_equal(np.asarray(z1[k]), np.asarray(z2[k][:n1])), k


@needs_real
def test_real_r1_then_r2_chain(tmp_path):
    """실제 에피소드 구조에서 규칙: r1(시연 50) 가중이 규칙과 같고, r2(prev=r1)의 앞부분은 r1 그대로·
    새 범위는 규칙대로. L1 은 실제 점수 대신 합성 (규칙 검증이 목적)."""
    res = {}
    prev, prev_d = None, None
    for r, root in ((1, REAL_R1), (2, REAL_R2)):
        ep, mode = _real_arrays(root)
        idx = np.arange(int((ep < 50).sum()), len(ep))
        l1 = np.full(len(ep), np.nan)
        l1[idx] = lognormal_l1(len(idx), seed=200 + r)
        s = save_scores(tmp_path / f"s{r}.npz", ep, idx, l1[idx], dataset_root=root)
        out = tmp_path / f"w{r}.npz"
        got = run_build(root, s, out, 50, prev=prev, task=f"square_ph50_r{r}")
        ref = ref_weights(ep, mode, 50, l1, prev=prev_d)
        assert np.array_equal(got["drop"], ref.drop), f"r{r} drop"
        assert np.allclose(got["w"], ref.w, rtol=1e-6), f"r{r} w"
        assert (got["w"][ep < 50] == 1).all()
        assert abs(got["w"][ref.ok].astype(np.float64).mean() - 1) < 1e-5
        res[r] = got
        prev, prev_d = out, got
    n1 = len(res[1]["w"])
    assert np.array_equal(res[2]["w"][:n1], res[1]["w"])
