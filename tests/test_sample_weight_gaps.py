"""빈칸 G1~G17 — 비평가가 찾은, 기존 3개 테스트 파일(build · score · train)이 다루지 않은 경로.

테스트 종류 (함수 이름의 gN = 비평가 목록의 N 번째):
  [수정]  의도 (1)~(8)에 어긋나 소스를 고친 경로. 기대값 = 의도.
  [재료]  결과 해석에 쓸 수치·집합 차이를 고정한다. 해석은 붙이지 않는다.
  [회귀]  기존 경로(sirius · bc · wbc · apo)가 이번 변경 전(HEAD)과 같게 도는지.
  [결정]  기대값을 사용자가 정해야 하는 경로. xfail(strict=True) 에 비평가가 낸 '후보 기대값' 을 적는다.
          지금 동작은 xfail 로 드러나고, 누가 구현하면 XPASS -> strict 실패가 되어 표시를 지우라는 신호가 된다.
          후보 기대값은 결정이 아니다.

기존 테스트 파일의 합성 데이터·헬퍼를 그대로 쓴다 (같은 데이터 모양 · 같은 '분명한 에러' 기준).
합성 데이터·출력은 pytest tmp_path 아래에만 만든다. 실데이터·실체크포인트·이전 산출물은 읽기만, 없으면 건너뛴다.
실행 (CPU, manibot 루트에서):
    CUDA_VISIBLE_DEVICES="" ~/miniconda3/envs/manibot/bin/python -m pytest tests/test_sample_weight_gaps.py
"""
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr
from hydra import compose, initialize_config_dir
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from safetensors.torch import save_file

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import test_sample_weight_build as BD  # noqa: E402
import test_sample_weight_score as SC  # noqa: E402
import test_sample_weight_train as TR  # noqa: E402

from manibot.policies.diffusion_ops import prepare_cond  # noqa: E402
from manibot.policies.factory import make_policy  # noqa: E402
from manibot.scripts.train import _build_finetune_loss  # noqa: E402
from manibot.utils.dataset_utils import create_dataset, create_dataset_stats  # noqa: E402
from manibot.utils.task_utils import derive_task_meta  # noqa: E402

REPO = TESTS.parent
CFG_DIR = str(REPO / "src" / "manibot" / "configs")
PREV = Path("/tmp/claude-1000/-home-jungwook-workspace-ljw-workspace/"
            "3c0919a6-1ea8-45d3-b605-feec3e75212e/scratchpad/l1review")
W_R1_REAL = PREV / "w_r1_real.npz"            # r1 새 프레임 전부를 base 로 채점(M=2)해 만든 가중
SCORES_R1_FULL = PREV / "scores_r1_full.npz"

need_art = pytest.mark.skipif(not (SCORES_R1_FULL.exists() and W_R1_REAL.exists()),
                              reason="이전 채점·가중 산출물 없음")
need_r2 = pytest.mark.skipif(not (TR.R1.exists() and TR.R2.exists()), reason="r1·r2 zarr 없음")
need_r1 = pytest.mark.skipif(not TR.R1.exists(), reason="r1 zarr 없음")


def decision(what):
    return pytest.mark.xfail(strict=True, reason=f"[결정] {what} — 사용자 결정. 후보 기대값(비평가 제안)")


# ── 공통: subprocess · 작은 정책 설정 ─────────────────────────────────────────

def run_py(args, env=None, cwd=REPO):
    e = {**os.environ, "CUDA_VISIBLE_DEVICES": "", **(env or {})}
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                       cwd=str(cwd), env=e)
    return r.returncode, r.stdout + r.stderr


def run_ok(mod, overrides):
    code, out = run_py(["-m", mod, *overrides])
    assert code == 0, f"{mod} 실패 (code {code}):\n{out[-4000:]}"
    return out


def small_ov(root, out, **over):
    """32 이미지 · down_dims [32,64] — 학습·채점·가중이 같은 정책 설정을 쓰게 한 곳에서 만든다."""
    ov = {"task": "piper_cube_stack", "policy": "diffusion", "device": "cpu",
          "task.dataset_root": root, "task.dataset_repo_id": "null", "task.fps": BD.FPS,
          "resize_shape": f"[{BD.HW},{BD.HW}]", "crop_shape": f"[{BD.HW - 4},{BD.HW - 4}]",
          "policy.unet.down_dims": "[32,64]",
          "train.batch_size": 4, "train.num_workers": 0, "train.use_amp": "false", "train.log_freq": 1,
          "wandb.enable": "false", "base_dir": out, "output_dir": Path(out) / "run",
          "hydra.run.dir": out}
    ov.update(over)
    return [f"{k}={v}" for k, v in ov.items() if v is not None]


def sw_train(n_demo, k=4, **over):
    """sample_weight 학습 설정 (train.py 가 요구하는 조합)."""
    return {"finetune.enabled": "true", "finetune.loss": "sample_weight", "finetune.balanced": "null",
            "finetune.n_demo_episodes": n_demo, "finetune.sirius_drop_preintv_min": k,
            "+policy.drop_n_last_frames": 0, **over}


# ── 픽스처 ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    """score 테스트의 합성 데이터(96 이미지 -> 크롭 84) + 같은 설정으로 저장한 체크포인트."""
    tmp = tmp_path_factory.mktemp("gaps_syn")
    root = tmp / "ds"
    actions, stats = SC.build_dataset(root)
    ckpt = SC.save_ckpt(tmp / "ckpt", SC.build_policy(root, seed=0))
    return SimpleNamespace(tmp=tmp, root=root, ckpt=ckpt, actions=actions, stats=stats)


def score(syn, ckpt, out, overrides=(), root=None, **plus):
    plus = {"start_index": 20, "end_index": 28, "M": 1, "batch_size": 4, **plus}
    return SC.run(SC.make_cfg(root or syn.root, ckpt, out, list(overrides), **plus))


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """build 테스트의 규칙 데이터(시연 2 + 배포 4, action_mode 있음) + 새 프레임 전부의 L1."""
    d = tmp_path_factory.mktemp("gaps_small")
    modes = BD.DEMO_MODES + [BD.DEPLOY_MODES[k] for k in ("a", "b", "d", "i")]
    ep, mode = BD.make_zarr(d / "ds", modes)
    idx = np.arange(int((ep < BD.N_DEMO).sum()), len(ep))
    return SimpleNamespace(dir=d, root=d / "ds", modes=modes, ep=ep, mode=mode, idx=idx,
                           l1=BD.lognormal_l1(len(idx), seed=7))


def small_scores(small, path, index=None, l1=None, **meta):
    index = small.idx if index is None else np.asarray(index)
    l1 = small.l1 if l1 is None else np.asarray(l1)
    BD.save_scores(path, small.ep, index, l1, dataset_root=small.root)
    if meta:
        d = dict(np.load(path))
        d.update(meta)
        np.savez(path, **d)
    return path


@pytest.fixture(scope="module")
def rounds(tmp_path_factory):
    """r1 ⊂ r2 ⊂ r3 (build 테스트의 라운드 구성) + 각 라운드 점수 + w1 · w2(prev=w1)."""
    d = tmp_path_factory.mktemp("gaps_rounds")
    out = SimpleNamespace(dir=d)
    modes = list(BD.DEMO_MODES)
    prev = None
    for r, dep in enumerate([BD.R1_DEPLOY, BD.R2_DEPLOY, BD.R3_DEPLOY], start=1):
        modes = modes + dep
        root = d / f"r{r}"
        ep, _ = BD.make_zarr(root, modes)
        idx = np.arange(int((ep < BD.N_DEMO).sum()), len(ep))
        s = BD.save_scores(d / f"s{r}.npz", ep, idx, BD.lognormal_l1(len(idx), 300 + r), dataset_root=root)
        w = d / f"w{r}.npz"
        if r < 3:
            BD.run_build(root, s, w, BD.N_DEMO, prev=prev)
            prev = w
        setattr(out, f"r{r}", SimpleNamespace(root=root, ep=ep, idx=idx, scores=s, w=w, N=len(ep)))
    return out


@pytest.fixture(scope="module")
def trsyn(tmp_path_factory):
    """train 테스트의 합성 데이터 (시연 2 + 배포 3, action_mode 있음)."""
    return TR.build_dataset(tmp_path_factory.mktemp("gaps_tr") / "ds", TR.synth_modes())


# ── G1 채점 설정이 체크포인트 학습 설정과 다름 ──────────────────────────────────

@pytest.mark.parametrize("override", [
    "crop_shape=[76,76]", "policy.pred_horizon=32", "policy.noise_scheduler.type=DDPM",
    "policy.noise_scheduler.num_inference_steps=100",
], ids=["crop76", "horizon32", "ddpm", "steps100"])
def test_g1_policy_config_differs_from_checkpoint_errors(syn, tmp_path, override):
    """[수정] 키 이름·모양은 같고 크롭·horizon·스케줄러·추론 스텝이 다른 정책으로 채점하면 분명한 에러, 파일 없음.
    (크롭 84 -> 76 은 둘 다 3x3 특징맵이라 strict 로드가 통과한다 — 키 대조로는 못 잡는다.)"""
    out = tmp_path / "s.npz"
    cfg = SC.make_cfg(syn.root, syn.ckpt, out, [override], start_index=20, end_index=24, M=1)
    SC.expect_clear_error(cfg, out)


def test_g1_fields_outside_generation_do_not_error(syn, tmp_path):
    """[수정] 생성에 영향 없는 필드(기기 · AMP · 옵티마이저 · 샘플러의 drop_n_last_frames)만 다르면 그대로 채점한다.
    실제 base 체크포인트는 device=cuda:0 으로 저장돼 있고 CPU·다른 GPU 에서 채점한다."""
    ck = tmp_path / "ck"
    shutil.copytree(syn.ckpt, ck)
    j = json.loads((ck / "config.json").read_text())
    j.update(device="cuda:0", use_amp=True, drop_n_last_frames=0, optimizer_lr=0.5)
    (ck / "config.json").write_text(json.dumps(j))
    got = score(syn, ck, tmp_path / "a.npz")
    ref = score(syn, syn.ckpt, tmp_path / "b.npz")
    np.testing.assert_array_equal(got["l1"], ref["l1"])
    assert int(got["num_inference_steps"]) == j["num_inference_steps"]


# ── G2 정규화 stats ──────────────────────────────────────────────────────────

def stats_copy(syn, root2, pad=0.5):
    shutil.copytree(syn.root, root2)
    j = json.loads((root2 / "config.json").read_text())
    st = j["stats"]["action"]
    st["min"] = [v - pad for v in st["min"]]
    st["max"] = [v + pad for v in st["max"]]
    (root2 / "config.json").write_text(json.dumps(j))
    return root2


def test_g2_stats_change_l1_without_any_signal(syn, tmp_path):
    """[재료] 프레임은 같고 config.json 의 action stats 만 다른 사본을 같은 체크포인트로 채점하면 L1 이 바뀌고
    에러가 없다 — 체크포인트에 stats 가 없어 대조할 기준이 없다 (G1 의 설정 대조로도 안 잡힌다)."""
    root2 = stats_copy(syn, tmp_path / "ds2")
    a = score(syn, syn.ckpt, tmp_path / "a.npz")
    b = score(syn, syn.ckpt, tmp_path / "b.npz", root=root2)
    np.testing.assert_array_equal(a["index"], b["index"])
    assert not np.allclose(a["l1"], b["l1"], rtol=1e-3)


@decision("점수 파일에 stats 흔적을 남기고 build·train 에서 대조할지")
def test_g2_candidate_scores_carry_stats_fingerprint(syn, tmp_path):
    out = score(syn, syn.ckpt, tmp_path / "a.npz")
    assert any("stat" in k for k in out)


# ── G3 점수 파일의 출처 ──────────────────────────────────────────────────────

@decision("EMA 가 아닌 정책(ema=False)의 점수를 build 가 거절할지")
def test_g3_candidate_build_rejects_raw_policy_scores(small, tmp_path):
    s = small_scores(small, tmp_path / "s.npz", ema=False)
    BD.expect_clear_error(tmp_path / "w.npz", small.root, s, n_demo=BD.N_DEMO)


@decision("가중 파일에 점수 출처(체크포인트 · EMA · 시드)를 복사할지")
def test_g3_candidate_weights_carry_score_provenance(small, tmp_path):
    s = small_scores(small, tmp_path / "s.npz", ema=True)
    res = BD.run_build(small.root, s, tmp_path / "w.npz", BD.N_DEMO)
    for k in ("checkpoint", "ema", "seed"):
        assert any(k in key for key in res), k


# ── G4 라운드 사이에 가중 파일이 바뀜 ─────────────────────────────────────────

def test_g4_rebuilt_previous_round_is_not_detected(rounds, tmp_path):
    """[재료] w2(prev=w1) 를 만든 뒤 w1 을 다른 점수로 다시 만들면 새 w1 과 w2[:n1] 이 갈라지지만,
    w3(prev=w2) 는 w2 만 보므로 에러 없이 만들어진다 — 어느 run 이 어떤 w1 으로 학습했는지 파일로 확인할 수 없다."""
    r1, r2, r3 = rounds.r1, rounds.r2, rounds.r3
    s1b = BD.save_scores(tmp_path / "s1b.npz", r1.ep, r1.idx, BD.lognormal_l1(len(r1.idx), 999),
                         dataset_root=r1.root)
    w1b = BD.run_build(r1.root, s1b, tmp_path / "w1b.npz", BD.N_DEMO)
    w2 = dict(np.load(r2.w))
    assert not np.array_equal(w1b["w"], w2["w"][:r1.N])
    w3 = BD.run_build(r3.root, r3.scores, tmp_path / "w3.npz", BD.N_DEMO, prev=r2.w)
    assert np.array_equal(w3["w"][:r2.N], w2["w"])


@decision("lo·hi 를 라운드 사이에 고정할지 (prev 와 다르면 에러)")
def test_g4_candidate_lo_hi_differs_from_prev_rejected(rounds, tmp_path):
    BD.expect_clear_error(tmp_path / "w.npz", rounds.r2.root, rounds.r2.scores, n_demo=BD.N_DEMO,
                          prev=rounds.r1.w, lo=0.2, hi=5.0)


@decision("가중 파일 해시를 체크포인트에 남겨 resume 때 대조할지")
def test_g4_candidate_resume_with_replaced_weights_rejected(trsyn, tmp_path):
    w = tmp_path / "w.npz"
    TR.write_weights(w, TR.synth_modes(), seed=0)
    ov = TR.synth_overrides(trsyn, tmp_path, {"finetune.sample_weights": w, "train.steps": 1,
                                              "train.save_freq": 1}) + [f"hydra.run.dir={tmp_path}"]
    run_ok("manibot.scripts.train", ov)
    TR.write_weights(w, TR.synth_modes(), seed=1)          # 다른 유효 파일로 교체
    ov[ov.index("train.steps=1")] = "train.steps=2"
    code, out = run_py(["-m", "manibot.scripts.train", *ov, "resume=true"])
    assert "Resumed from checkpoint at step 1" in out
    assert code != 0


# ── G5 범위를 나눈 채점 · 합치기 · 중단 ───────────────────────────────────────

@pytest.mark.parametrize("same_values", [False, True], ids=["different", "same"])
def test_g5_duplicate_score_index_rejected(small, tmp_path, same_values):
    """[수정] 손으로 합친 점수 파일에 같은 프레임이 두 번 있으면 분명한 에러, 가중 파일 없음.
    값이 다르면 어느 쪽을 쓸지 정할 근거가 없고(마지막 값이 조용히 이긴다), 같아도 합치기가 잘못됐다는 신호다."""
    dup = small.idx[:5]
    l1 = np.r_[small.l1, small.l1[:5] * (1 if same_values else 3)]
    s = small_scores(small, tmp_path / "s.npz", np.r_[small.idx, dup], l1)
    BD.expect_clear_error(tmp_path / "w.npz", small.root, s, n_demo=BD.N_DEMO)


def test_g5_unsorted_unique_index_gives_same_weights(small, tmp_path):
    """[재료] 조각을 순서 없이 이어 붙여도(index 가 중복 없이 섞임) 가중은 정렬된 파일과 같다."""
    perm = np.random.default_rng(0).permutation(len(small.idx))
    a = BD.run_build(small.root, small_scores(small, tmp_path / "a.npz"), tmp_path / "wa.npz", BD.N_DEMO)
    s = small_scores(small, tmp_path / "b.npz", small.idx[perm], small.l1[perm])
    b = BD.run_build(small.root, s, tmp_path / "wb.npz", BD.N_DEMO)
    np.testing.assert_array_equal(a["w"], b["w"])
    np.testing.assert_array_equal(a["drop"], b["drop"])


def test_g5_range_pieces_concatenated_equal_one_run(syn, tmp_path):
    """[재료] [20,40) 과 [40,58) 로 나눠 채점해 이어 붙이면 [20,58) 한 번과 행 단위로 같다 —
    중단되면 끝난 범위는 두고 start_index 로 이어 돌릴 수 있다 (합치는 도구는 없다: 사용자 결정)."""
    a = score(syn, syn.ckpt, tmp_path / "a.npz", start_index=20, end_index=40)
    b = score(syn, syn.ckpt, tmp_path / "b.npz", start_index=40, end_index=SC.N)
    full = score(syn, syn.ckpt, tmp_path / "f.npz", start_index=20, end_index=SC.N, batch_size=9)
    np.testing.assert_array_equal(np.r_[a["index"], b["index"]], full["index"])
    np.testing.assert_allclose(np.r_[a["l1"], b["l1"]], full["l1"], rtol=1e-5, atol=1e-6)


def test_g5_failure_mid_scoring_leaves_no_partial_file(syn, monkeypatch, tmp_path):
    """[재료] 둘째 배치에서 생성이 실패(OOM 흉내)하면 파일이 남지 않는다 — 끝에서 한 번만 저장하므로
    그때까지의 계산은 잃는다. 부분 저장을 할지는 사용자 결정."""
    calls = {"n": 0}
    orig = DiffusionModel.conditional_sample

    def cs(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("CUDA out of memory (가짜)")
        return orig(self, *a, **k)
    monkeypatch.setattr(DiffusionModel, "conditional_sample", cs)
    out = tmp_path / "s.npz"
    with pytest.raises(RuntimeError, match="out of memory"):
        score(syn, syn.ckpt, out, start_index=20, end_index=36)
    assert calls["n"] == 2 and not out.exists()


# ── G6 교차 비교 두 팔의 학습 샘플 집합 (실데이터 r1) ──────────────────────────

@need_r1
def test_g6_real_r1_sample_sets_of_the_two_arms(tmp_path):
    """[재료] r1 에서 ours(sample_weight · drop_n_last_frames=0 · PI k=4, 판정 칸 t..t+14) 와 원문 SIRIUS 팔(sirius drop9 ·
    첫 칸 라벨 · 16칸 판정 · drop_n_last_frames 기본 7)이 실제로 뽑는 샘플 집합. 가중은 합성 L1 로 이 자리에서 만든다
    (뽑는 집합은 w 와 무관하다). 값은 2026-10-09 판정 칸 결정 뒤 이 코드로 잰 것 — 비교 run 전의 기준."""
    from manibot.losses.sirius import LABELS
    ep_r, mode_r = BD._real_arrays(TR.R1)
    idx = np.arange(int((ep_r < 50).sum()), len(ep_r))
    s = BD.save_scores(tmp_path / "s.npz", ep_r, idx, BD.lognormal_l1(len(idx), 0), dataset_root=TR.R1)
    BD.run_build(TR.R1, s, tmp_path / "w.npz", 50, task="square_ph50_r1")
    cfg = TR.compose_cfg(TR.real_overrides(TR.R1, tmp_path, sw_train(50, **{
        "finetune.sample_weights": tmp_path / "w.npz"})))
    policy, pre, ds = TR.build_policy_dataset(cfg)
    ours = TR.make_trainer(cfg, policy, pre, ds)
    cfg2 = TR.compose_cfg(TR.real_overrides(TR.R1, tmp_path, {
        "finetune.loss": "sirius", "finetune.sirius_drop_preintv_min": 9}))
    sirius = TR.make_trainer(cfg2, policy, pre, ds)
    A, B = set(ours.train_dataloader.sampler.indices), set(sirius.train_dataloader.sampler.indices)
    ep = np.asarray(ds.replay_buffer["episode_index"]).ravel()
    is_pre = ours.loss_fn.lab == LABELS["preintv"]
    n_pre, n_judge = is_pre.sum(1), (is_pre & ours.loss_fn.judge).sum(1)    # 16칸 · 판정 칸
    # ours 의 집합 = 독립 구현(build 테스트의 ref_pi)이 PI 로 보지 않는 샘플
    assert A == set(np.flatnonzero(~BD.ref_pi(ep_r, mode_r, 50)))
    last7 = np.zeros(len(ep), dtype=bool)
    for e in np.unique(ep):
        last7[np.flatnonzero(ep == e)[-7:]] = True
    ab, ba = np.array(sorted(A - B)), np.array(sorted(B - A))
    print(f"\nours {len(A)} · sirius {len(B)} · 공통 {len(A & B)} · ours만 {len(ab)} · sirius만 {len(ba)}")
    assert (len(A), len(B), len(A & B)) == (14350, 13845, 13692)
    # ours 에만: 에피소드마다 끝 7 프레임 (시연 50x7 + 배포 44x7), 모두 preintv 0 칸
    assert len(ab) == 658 and last7[ab].all() and (n_pre[ab] == 0).all()
    assert ((ep[ab] < 50).sum(), (ep[ab] >= 50).sum()) == (350, 308)
    # sirius 에만: 배포, 끝 7 프레임 밖, 16칸 preintv 4~8 칸 = 판정 칸 preintv 4~8 칸
    assert len(ba) == 153 and not last7[ba].any() and (ep[ba] >= 50).all()
    assert np.bincount(n_pre[ba], minlength=9).tolist() == [0, 0, 0, 0, 17, 34, 34, 34, 34]
    assert np.bincount(n_judge[ba], minlength=9).tolist() == [0, 0, 0, 0, 34, 34, 34, 34, 17]


# ── G7 유효 칸 수별 L1 · w (실제 산출물) ──────────────────────────────────────

@need_art
def test_g7_real_l1_and_weight_by_valid_slots():
    """[재료] base 로 채점한 r1 새 프레임(M=2): 유효 칸 수별 l1_mean 평균과 비-PI 의 w 평균. 해석은 붙이지 않는다.
    (L1 이 '유효 칸 평균' 이라는 공식 자체는 score 테스트 test_slot_offsets_averaged_over_valid_slots_only 가 고정한다.)
    이 두 산출물은 2026-10-09 판정 칸 결정 전의 것이다 (L1·PI 모두 16칸, judge 키 없음) — 지금 코드가 내는 값이 아니다."""
    s, w = np.load(SCORES_R1_FULL), np.load(W_R1_REAL)
    nv, l1 = s["n_valid_slots"], s["l1_mean"]
    table = {int(k): (int((nv == k).sum()), round(float(l1[nv == k].mean()), 4)) for k in np.unique(nv)}
    print("\n유효 칸 -> (샘플 수, l1_mean 평균):", table)
    assert table[2] == (44, 0.1495) and table[3] == (44, 0.1379) and table[5] == (44, 0.1109)
    assert table[15] == (88, 0.0911) and table[16] == (6613, 0.0907)
    ww, ok = w["w"][s["index"]], ~w["drop"][s["index"]]
    short, full = ok & (nv <= 8), ok & (nv == 16)
    print(f"비-PI w 평균: 유효 칸 <=8 {short.sum()} 개 {ww[short].mean():.3f} · 16칸 {full.sum()} 개 {ww[full].mean():.3f}")
    assert (int(short.sum()), round(float(ww[short].mean()), 3)) == (308, 1.163)
    assert (int(full.sum()), round(float(ww[full].mean()), 3)) == (6205, 0.983)


# ── G8 use_ema=false ─────────────────────────────────────────────────────────

def test_g8_use_ema_false_scores_raw_weights(syn, tmp_path):
    """[수정] use_ema=false 면 EMA 가 있어도 raw 가중치로 채점하고 ema=False 를 남긴다 (eval.py 와 같은 뜻).
    raw=정책1 · EMA=정책2 체크포인트를 use_ema=false 로 채점한 값 == 정책1 로 채점한 값."""
    p1, p2 = SC.build_policy(syn.root, seed=1), SC.build_policy(syn.root, seed=2)
    mix = SC.save_ckpt(tmp_path / "raw1_ema2", p1, ema=p2)
    only1 = SC.save_ckpt(tmp_path / "raw1_ema1", p1, ema=p1)
    a = score(syn, mix, tmp_path / "a.npz", ["use_ema=false"])
    b = score(syn, only1, tmp_path / "b.npz")
    np.testing.assert_array_equal(a["l1"], b["l1"])
    assert not bool(a["ema"]) and bool(b["ema"])


def test_g8_use_ema_false_without_training_state(syn, tmp_path):
    """[수정] EMA 가 없는 체크포인트도 use_ema=false 를 명시하면 채점하고 ema=False 를 남긴다."""
    ck = SC.save_ckpt(tmp_path / "ck", SC.build_policy(syn.root, seed=0), ema=None)
    out = score(syn, ck, tmp_path / "a.npz", ["use_ema=false"])
    assert not bool(out["ema"]) and np.isfinite(out["l1"]).all()


# ── G9 LeRobot 비공개 API · 기본값 (Docker 등 다른 버전 환경에서 돌릴 것) ─────────

def test_g9_lerobot_private_api_contract(syn):
    """[재료] 새 코드가 기대는 LeRobot 비공개 API 와 기본값. 버전이 다른 환경에서 의미가 바뀌면 여기서 갈린다.
    버전은 출력으로 남긴다 (pyproject 상한을 둘지는 사용자 결정)."""
    import diffusers
    import lerobot
    print(f"\nlerobot {lerobot.__version__} · diffusers {diffusers.__version__} · torch {torch.__version__}"
          f" · zarr {zarr.__version__}")
    assert "noise" in inspect.signature(DiffusionModel.conditional_sample).parameters
    assert callable(getattr(DiffusionModel, "_prepare_global_conditioning", None))
    norm = DiffusionConfig.__dataclass_fields__["normalization_mapping"].default_factory()
    assert norm["ACTION"].value == "MIN_MAX" and norm["STATE"].value == "MIN_MAX"
    # eval 모드 = 센터 크롭(결정적) · train 모드 = 랜덤 크롭 — score_samples 의 재현성이 여기에 기댄다
    cfg = SC.make_cfg(syn.root)
    meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, meta)
    torch.manual_seed(0)
    policy, pre, _ = make_policy(cfg, meta, stats)
    ds = create_dataset(policy, cfg)
    items = [ds[i] for i in range(20, 24)]
    b = pre({k: torch.stack([it[k] for it in items]) for k in items[0] if isinstance(items[0][k], torch.Tensor)})
    conds = {}
    with torch.no_grad():
        for mode in ("eval", "train"):
            getattr(policy, mode)()
            for s in (1, 2):
                torch.manual_seed(s)
                conds[mode, s] = prepare_cond(policy, b)[0]
    assert torch.equal(conds["eval", 1], conds["eval", 2])
    assert not torch.equal(conds["train", 1], conds["train", 2])


# ── G10 라운드 학습의 초기화 · 설정 누락 ───────────────────────────────────────

@decision("sample_weight 학습에 theta_init 이 없으면 명시 옵션 없이는 에러로 할지 (지금은 무작위 초기화로 진행)")
def test_g10_candidate_missing_init_rejected(trsyn, tmp_path):
    w = tmp_path / "w.npz"
    TR.write_weights(w, TR.synth_modes())
    ov = TR.synth_overrides(trsyn, tmp_path, {"finetune.sample_weights": w, "train.steps": 1,
                                              "train.save_freq": 100})
    code, out = run_py(["-m", "manibot.scripts.train", *ov, f"hydra.run.dir={tmp_path}"])
    assert code != 0, out[-2000:]


@decision("theta_init 의 키가 정책과 다르면 에러로 할지 (지금은 strict=False 경고 후 무작위 정책에서 출발; 기존 경로에도 같은 블록)")
def test_g10_candidate_theta_init_key_mismatch_rejected(trsyn, tmp_path):
    w = tmp_path / "w.npz"
    TR.write_weights(w, TR.synth_modes())
    cfg, policy, _, _ = TR.setup_synth(trsyn, tmp_path, {"finetune.sample_weights": w})
    ck = tmp_path / "ck"
    ck.mkdir()
    save_file({f"other.{k}": v.detach().clone().contiguous() for k, v in policy.state_dict().items()},
              str(ck / "model.safetensors"))
    ov = TR.synth_overrides(trsyn, tmp_path, {"finetune.sample_weights": w, "finetune.theta_init": ck,
                                              "train.steps": 1, "train.save_freq": 100})
    code, out = run_py(["-m", "manibot.scripts.train", *ov, f"hydra.run.dir={tmp_path}"])
    assert code != 0, out[-2000:]


@decision("finetune.enabled=false 인데 loss=sample_weight·sample_weights 가 있으면 에러로 할지 (지금은 가중 없는 BC)")
def test_g10_candidate_disabled_finetune_with_weights_rejected(trsyn, tmp_path):
    w = tmp_path / "w.npz"
    TR.write_weights(w, TR.synth_modes())
    cfg, policy, pre, ds = TR.setup_synth(trsyn, tmp_path, {"finetune.sample_weights": w,
                                                            "finetune.enabled": "false",
                                                            "finetune.sirius_drop_preintv_min": "null"})
    TR.raises_clear(lambda: TR.make_trainer(cfg, policy, pre, ds))


# ── G11 라운드 연쇄 전체 (실제 진입점, 각 단계 산출물을 다음 입력으로) ─────────────

def test_g11_round_chain_end_to_end(tmp_path):
    """[재료/검증] base 학습 -> score(base) -> build -> train(r1, theta_init=base) -> score(r1 체크포인트, 새 범위만)
    -> build(prev) -> train(r2). 확인: train 이 저장한 체크포인트를 score 가 EMA 로 싣고(설정 대조 통과),
    r2 가중의 앞부분이 r1 가중 그대로이고, 마지막 학습이 그 drop 을 PI 제외로 그대로 받는다."""
    n_demo = BD.N_DEMO
    m0 = list(BD.DEMO_MODES)
    m1 = m0 + BD.R1_DEPLOY
    m2 = m1 + BD.R2_DEPLOY
    for name, m in (("r0", m0), ("r1", m1), ("r2", m2)):
        BD.make_zarr(tmp_path / name, m)
    ep1 = np.asarray(zarr.open(str(tmp_path / "r1"), "r")["data"]["episode_index"]).ravel()
    ep2 = np.asarray(zarr.open(str(tmp_path / "r2"), "r")["data"]["episode_index"]).ravel()
    n1, N2 = len(ep1), len(ep2)
    ck = "run/checkpoints/step_0000000002"
    steps = {"train.steps": 2, "train.save_freq": 2}

    run_ok("manibot.scripts.train", small_ov(tmp_path / "r0", tmp_path / "base", **steps))
    base = tmp_path / "base" / ck
    run_ok("manibot.scripts.score_samples", small_ov(tmp_path / "r1", tmp_path / "sc1", **{
        "+checkpoint": base, "+n_demo_episodes": n_demo, "+M": 2, "+batch_size": 16,
        "+out": tmp_path / "s1.npz"}))
    run_ok("manibot.scripts.build_sample_weights", small_ov(tmp_path / "r1", tmp_path / "b1", **{
        "+scores": tmp_path / "s1.npz", "+n_demo_episodes": n_demo, "+out": tmp_path / "w1.npz"}))
    run_ok("manibot.scripts.train", small_ov(tmp_path / "r1", tmp_path / "t1", **sw_train(n_demo, **{
        "finetune.sample_weights": tmp_path / "w1.npz", "finetune.theta_init": base}), **steps))
    ck1 = tmp_path / "t1" / ck
    run_ok("manibot.scripts.score_samples", small_ov(tmp_path / "r2", tmp_path / "sc2", **{
        "+checkpoint": ck1, "+start_index": n1, "+M": 2, "+batch_size": 16, "+out": tmp_path / "s2.npz"}))
    run_ok("manibot.scripts.build_sample_weights", small_ov(tmp_path / "r2", tmp_path / "b2", **{
        "+scores": tmp_path / "s2.npz", "+prev": tmp_path / "w1.npz", "+n_demo_episodes": n_demo,
        "+out": tmp_path / "w2.npz"}))
    out = run_ok("manibot.scripts.train", small_ov(tmp_path / "r2", tmp_path / "t2", **sw_train(n_demo, **{
        "finetune.sample_weights": tmp_path / "w2.npz", "finetune.theta_init": ck1}), **steps))

    s1, s2 = np.load(tmp_path / "s1.npz"), np.load(tmp_path / "s2.npz")
    w1, w2 = np.load(tmp_path / "w1.npz"), np.load(tmp_path / "w2.npz")
    assert bool(s1["ema"]) and bool(s2["ema"]), "train 이 저장한 EMA 를 score 가 싣지 못했다"
    assert str(s2["checkpoint"]) == str(ck1)
    np.testing.assert_array_equal(s1["index"], np.arange(int((ep1 < n_demo).sum()), n1))
    np.testing.assert_array_equal(s2["index"], np.arange(n1, N2))
    assert np.array_equal(w2["w"][:n1], w1["w"]) and np.array_equal(w2["drop"][:n1], w1["drop"])
    assert int(w2["n_prev"]) == n1 and len(w2["w"]) == N2
    n_drop = int(w2["drop"].sum())
    assert f"pi_theta 초기화 <- {ck1} (EMA 적용)" in out
    assert f"PI 제외 {n_drop}" in out
    assert f"preintv 4칸 이상 샘플 제외: {N2} -> {N2 - n_drop}" in out
    losses = TR.logged_losses(out)
    assert len(losses) == 2 and all(np.isfinite(losses)), losses
    assert (tmp_path / "t2" / ck / "training_state.pt").exists()


# ── G12 python -O 에서도 검증이 남는가 ───────────────────────────────────────

O_TRAINER = r'''
import json, sys
sys.path.insert(0, {tests!r})
import test_sample_weight_train as TR
a = json.loads(sys.argv[1])
cfg = TR.compose_cfg(TR.synth_overrides(a["root"], a["out"], a["over"]))
policy, pre, ds = TR.build_policy_dataset(cfg)
try:
    TR.make_trainer(cfg, policy, pre, ds)
except ValueError as e:
    print("VALUEERROR:", e)
    sys.exit(3)
print("NO ERROR")
'''

O_SCORE = r'''
import json, sys
sys.path.insert(0, {tests!r})
import test_sample_weight_score as SC
a = json.loads(sys.argv[1])
cfg = SC.make_cfg(a["root"], a["ckpt"], a["out"], M=1)
try:
    SC.ss.main.__wrapped__(cfg)
except ValueError as e:
    print("VALUEERROR:", e)
    sys.exit(3)
print("NO ERROR")
'''


@pytest.mark.parametrize("case", ["drop_last_7", "no_k", "no_n_demo", "balanced", "drop_mismatch",
                                  "bc_with_k"])
def test_g12_validation_survives_python_O(trsyn, tmp_path, case):
    """[수정] PYTHONOPTIMIZE=1 (assert 가 사라지는 실행)에서도 잘못된 sample_weight 설정은 ValueError 로 멈춘다.
    assert 로만 막으면 drop_n_last_frames=7 이 조용히 통과해 끝 프레임이 빠진 다른 학습이 된다."""
    modes = TR.synth_modes()
    w, drop = TR.write_weights(tmp_path / "w.npz", modes)
    over = {"finetune.sample_weights": str(tmp_path / "w.npz")}
    if case == "drop_last_7":
        over["+policy.drop_n_last_frames"] = 7
    elif case == "no_k":
        over["finetune.sirius_drop_preintv_min"] = "null"
    elif case == "no_n_demo":
        over["finetune.n_demo_episodes"] = "null"
    elif case == "balanced":
        over["finetune.balanced"] = "[0.5,0.25,0.25]"
    elif case == "drop_mismatch":
        flip = drop.copy()
        flip[np.flatnonzero(~drop)[-1]] = True             # PI 아닌 샘플 하나를 drop 으로
        TR.write_weights(tmp_path / "w.npz", modes, w=w, drop=flip)
    elif case == "bc_with_k":
        over.update({"finetune.loss": "bc", "+policy.drop_n_last_frames": None})
    arg = json.dumps({"root": str(trsyn), "out": str(tmp_path), "over": over})
    code, out = run_py(["-c", O_TRAINER.format(tests=str(TESTS)), arg], env={"PYTHONOPTIMIZE": "1"})
    assert code == 3, f"ValueError 가 아니다 (code {code}):\n{out[-3000:]}"


def test_g12_score_validation_survives_python_O(syn, tmp_path):
    """[수정] PYTHONOPTIMIZE=1 에서 +start_index·+n_demo_episodes 가 둘 다 없으면 ValueError (int(None) TypeError 가 아니라)."""
    arg = json.dumps({"root": str(syn.root), "ckpt": str(syn.ckpt), "out": str(tmp_path / "s.npz")})
    code, out = run_py(["-c", O_SCORE.format(tests=str(TESTS)), arg], env={"PYTHONOPTIMIZE": "1"})
    assert code == 3, f"ValueError 가 아니다 (code {code}):\n{out[-3000:]}"
    assert not (tmp_path / "s.npz").exists()


# ── G13 클래스별 실제 손실 몫 로그 ───────────────────────────────────────────

@decision("클래스별 실제 손실 몫(share/{c} = 클래스 sum(w) / 배치 sum(w))을 로그로 남길지")
def test_g13_candidate_loss_logs_class_share(trsyn, tmp_path):
    w, drop = TR.write_weights(tmp_path / "w.npz", TR.synth_modes())
    cfg, policy, pre, ds = TR.setup_synth(trsyn, tmp_path, {"finetune.sample_weights": tmp_path / "w.npz"})
    loss_fn = _build_finetune_loss(cfg, policy, ds)
    ok = TR.new_ok_indices(w, drop, TR.N_DEMO_FRAMES)
    b = TR.batch_of(ds, pre, [3, 40, ok[0], ok[20], ok[-1]])
    with torch.no_grad():
        _, out = loss_fn(policy, b)
    share = {k: v for k, v in out.items() if k.startswith("share/")}
    assert share and abs(sum(share.values()) - 1) < 1e-6


# ── G14 라운드 2 기본 채점 범위 (실데이터 프레임 수) ───────────────────────────

@need_r2
def test_g14_real_r2_default_range_includes_frozen_r1_range():
    """[재료] r2 를 기본 범위(첫 비시연 프레임~끝)로 채점하면 build 가 prev 로 고정해 쓰지 않는 r1 배포 범위도 채점한다.
    필요한 범위는 [len(w1), N2) 이고, 그것만 채점해도 build(prev) 가 받는다 — G11 연쇄가 실제로 그렇게 돈다."""
    ep1 = np.asarray(zarr.open(str(TR.R1), "r")["data"]["episode_index"]).ravel()
    ep2 = np.asarray(zarr.open(str(TR.R2), "r")["data"]["episode_index"]).ravel()
    n_demo_frames, n1, N2 = int((ep2 < 50).sum()), len(ep1), len(ep2)
    default, needed = N2 - n_demo_frames, N2 - n1
    print(f"\nr2 기본 범위 {default} 프레임 = 필요한 새 범위 {needed} + 고정된 r1 배포 {n1 - n_demo_frames}")
    assert (n_demo_frames, n1, N2) == (7468, 14741, 25058)
    assert (default, n1 - n_demo_frames) == (17590, 7273)


# ── G15 기존 apo · wbc · sirius · bc 경로 회귀 (HEAD 대 작업 트리) ───────────────

REGRESSION = r'''
import json, sys
from pathlib import Path
import numpy as np
import torch
import manibot
a = json.loads(Path(sys.argv[1]).read_text())
assert str(Path(manibot.__file__).resolve()).startswith(a["src"]), manibot.__file__
from hydra import compose, initialize_config_dir
from manibot.policies.factory import make_policy
from manibot.scripts.train import PolicyTrainer
from manibot.utils.dataset_utils import create_dataloader, create_dataset, create_dataset_stats
from manibot.utils.task_utils import derive_task_meta
with initialize_config_dir(config_dir=str(Path(manibot.__file__).parent / "configs"), version_base="1.3"):
    cfg = compose("default_policy", overrides=a["overrides"])
meta, stats = create_dataset_stats(cfg)
derive_task_meta(cfg.task, meta)
torch.manual_seed(0)
policy, pre, _ = make_policy(cfg, meta, stats)
ds = create_dataset(policy, cfg)
dl = create_dataloader(ds, cfg, is_training=True)
tr = PolicyTrainer(cfg, policy, device="cpu", train_dataloader=dl, preprocessor=pre)
res = {"loss_fn": type(tr.loss_fn).__name__}
bs = dl.batch_sampler
if type(bs).__name__ == "BalancedBatchSampler":
    it = iter(bs)
    res.update(sampler="balanced", per=[int(n) for n in bs.per], n_batches=len(bs),
               pools=[len(p) for p in bs.pools], batches=[[int(i) for i in next(it)] for _ in range(3)])
else:
    s = dl.sampler
    res.update(sampler=type(s).__name__,
               indices=sorted(int(i) for i in s.indices) if hasattr(s, "indices") else len(s))
if type(tr.loss_fn).__name__ == "SiriusLoss":
    res["n"] = tr.loss_fn.n
items = [ds[i] for i in a["batch"]]
b = {k: torch.stack([it[k] for it in items]) for k in items[0] if isinstance(items[0][k], torch.Tensor)}
idx = b["dataset_index"]
b = pre(b)
b["dataset_index"] = idx
policy.train()
torch.manual_seed(5)
loss, out = tr.loss_fn(policy, b) if tr.loss_fn is not None else policy.forward(b)
res["loss"] = float(loss)
res["out"] = {k: float(v) for k, v in sorted((out or {}).items())
              if isinstance(v, (int, float)) or (torch.is_tensor(v) and v.numel() == 1)}
Path(a["result"]).write_text(json.dumps(res))
'''


@pytest.fixture(scope="module")
def regress(tmp_path_factory, trsyn):
    """HEAD 의 src 사본 + apo·wbc 입력(labels · t_stats · ref 체크포인트)."""
    d = tmp_path_factory.mktemp("gaps_regress")
    arch = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD", "src"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(d)], input=arch.stdout, check=True)
    modes = TR.synth_modes()
    fl = TR.ref_frame_labels(modes, TR.N_DEMO)
    ep = TR.ref_episode_index(modes)
    has = np.isin(ep, [e for e, m in enumerate(modes) if e >= TR.N_DEMO and (np.asarray(m) == 1).any()])
    S = np.where(fl == TR.LABELS["intv"], 1.0, np.where(fl == TR.LABELS["preintv"], -1.0, 0.0))
    np.savez(d / "labels.npz", S=S, has_intv=has)
    np.save(d / "t_stats.npy", np.linspace(0.1, 1.0, 100).astype(np.float32))
    cfg = TR.compose_cfg(TR.synth_overrides(trsyn, d))
    meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, meta)
    torch.manual_seed(1)
    ref, _, _ = make_policy(cfg, meta, stats)
    SC.save_ckpt(d / "ref", ref)
    return SimpleNamespace(dir=d, head_src=d / "src", work_src=REPO / "src")


REGRESS_CASES = {
    "sirius_drop9": {"finetune.loss": "sirius", "finetune.sirius_drop_preintv_min": 9,
                     "finetune.n_demo_episodes": TR.N_DEMO, "finetune.balanced": "null"},
    "bc": {"finetune.loss": "bc", "finetune.balanced": "null"},
    "wbc_balanced3": {"finetune.loss": "wbc", "finetune.labels": "{d}/labels.npz",
                      "finetune.t_stats": "{d}/t_stats.npy", "finetune.balanced": "[0.5,0.25,0.25]"},
    "apo_balanced4": {"finetune.loss": "apo", "finetune.labels": "{d}/labels.npz",
                      "finetune.t_stats": "{d}/t_stats.npy", "finetune.ref_checkpoint": "{d}/ref",
                      "finetune.n_demo_episodes": TR.N_DEMO, "finetune.balanced": "[0.25,0.25,0.25,0.25]"},
}


@pytest.mark.parametrize("case", list(REGRESS_CASES))
def test_g15_existing_paths_match_head(regress, trsyn, tmp_path, case):
    """[회귀] 같은 합성 입력으로 HEAD(이번 변경 전)와 작업 트리에서 학습기를 만들어 비교한다: 손실 종류 · 샘플러 종류 ·
    풀 크기·배치 몫·처음 3 배치(balanced) 또는 샘플 목록 · 클래스 수 · 고정 배치의 손실과 로그 값."""
    over = {"train.batch_size": 8, "finetune.enabled": "true", "finetune.sirius_drop_preintv_min": "null",
            "finetune.n_demo_episodes": "null", "+policy.drop_n_last_frames": None}
    over.update({k: v.format(d=regress.dir) if isinstance(v, str) else v for k, v in REGRESS_CASES[case].items()})
    ov = TR.synth_overrides(trsyn, tmp_path, over)
    results = {}
    for name, src in (("head", regress.head_src), ("work", regress.work_src)):
        args = tmp_path / f"{name}.json"
        args.write_text(json.dumps({"src": str(src.resolve()), "overrides": ov, "batch": [3, 31, 70, 100, 120, 140],
                                    "result": str(tmp_path / f"{name}_res.json")}))
        script = tmp_path / "regress.py"
        script.write_text(REGRESSION)
        code, out = run_py([script, args], env={"PYTHONPATH": str(src)}, cwd=tmp_path)
        assert code == 0, f"{name} 실패:\n{out[-3000:]}"
        results[name] = json.loads((tmp_path / f"{name}_res.json").read_text())
    h, w = results["head"], results["work"]
    print(f"\n{case}: {json.dumps({k: v for k, v in w.items() if k not in ('indices', 'batches')})[:400]}")
    assert h["loss_fn"] == w["loss_fn"] and h["sampler"] == w["sampler"]
    for k in ("per", "n_batches", "pools", "batches", "indices", "n"):
        assert h.get(k) == w.get(k), k
    assert w["loss"] == pytest.approx(h["loss"], rel=1e-6, abs=1e-9)
    assert h["out"].keys() == w["out"].keys()
    for k in h["out"]:                                       # apo 의 u_cap 처럼 양쪽 다 NaN 인 값은 같은 것으로
        assert w["out"][k] == pytest.approx(h["out"][k], rel=1e-6, abs=1e-9, nan_ok=True), k


# ── G16 같은 초에 시작한 seed 병렬 학습의 출력 폴더 ─────────────────────────────

@decision("seed 를 출력 폴더 이름(session)에 넣을지 또는 겹치면 에러로 할지 — 기존 코드")
def test_g16_candidate_parallel_seeds_get_distinct_output_dirs(tmp_path):
    for _ in range(5):
        with initialize_config_dir(config_dir=CFG_DIR, version_base="1.3"):
            a = compose("default_policy", overrides=[f"base_dir={tmp_path}", "seed=0"]).output_dir
            b = compose("default_policy", overrides=[f"base_dir={tmp_path}", "seed=1"]).output_dir
        if a[-15:] == b[-15:]:                              # 같은 초 (YYYYmmdd_HHMMSS)
            break
        time.sleep(0.2)
    else:
        pytest.skip("같은 초에 두 설정을 만들지 못했다")
    assert a != b, f"seed 0 과 1 이 같은 폴더를 쓴다: {a}"


# ── G17 구성은 같고 내용이 다른 데이터셋 ──────────────────────────────────────

@decision("점수·가중에 데이터 내용 해시를 남겨 다른 사본과 섞이면 에러로 할지")
def test_g17_candidate_same_layout_other_content_rejected(small, tmp_path):
    other = tmp_path / "other"
    shutil.copytree(small.root, other)
    a = zarr.open(str(other), "r+")["data"]["action"]
    a[:] = np.asarray(a) + 1.0
    s = small_scores(small, tmp_path / "s.npz")             # 원본(small.root)으로 만든 점수
    BD.expect_clear_error(tmp_path / "w.npz", other, s, n_demo=BD.N_DEMO)
