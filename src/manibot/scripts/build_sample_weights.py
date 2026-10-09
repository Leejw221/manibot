"""라운드 학습의 샘플 가중 w 를 만든다 (라운드당 1회, score_samples 다음).

규칙:
  시연 프레임                 w = 1
  이전 라운드 프레임 (+prev)  이전 라운드 파일의 w · drop 을 그대로 — 한 번 정한 가중은 고정한다
  새 프레임 중 PI 샘플        w = 0 · drop. 판정 칸(프레임 t..t+14 중 에피소드 안, judge_slots)에 개입 직전 칸
                              (relabel_preintv k=15)이 drop_preintv_min 개 이상. 학습은 배치에 넣지 않는다
                              (sirius_drop_preintv_min). t-1 칸·패딩 칸은 세지 않는다 [사용자 결정 2026-10-09]
  새 프레임 중 나머지         r = L1 / (새 비-PI 샘플의 평균 L1) -> clip(lo, hi) -> 평균이 1 이 되게 나눈다
                              L1 = score_samples 의 l1_mean (직전 라운드 정책이 그 청크를 얼마나 재현하나)

산출: <out>.npz — w (N,) · drop (N,) · episode_index (N,) · judge(판정 규약) · 요약 · 입력 경로
사용:
    python -m manibot.scripts.build_sample_weights task=square_ph50_r1 \
        +scores=.../scores_r1.npz +n_demo_episodes=50 +out=.../weights_r1.npz
    python -m manibot.scripts.build_sample_weights task=square_ph50_r2 \
        +scores=.../scores_r2.npz +prev=.../weights_r1.npz +n_demo_episodes=50 +out=.../weights_r2.npz
"""

import hydra
import numpy as np

from manibot.losses.sample_weight import judge_slots
from manibot.losses.sirius import LABELS, action_windows, frame_labels
from manibot.policies.factory import make_policy
from manibot.utils.dataset_utils import create_dataset, create_dataset_stats
from manibot.utils.task_utils import derive_task_meta


@hydra.main(version_base="1.3", config_path="../configs", config_name="default_policy")
def main(cfg):
    n_demo = cfg.get("n_demo_episodes") or cfg.finetune.get("n_demo_episodes")
    if not n_demo:
        raise ValueError("+n_demo_episodes (또는 finetune.n_demo_episodes) 가 필요하다")
    n_demo = int(n_demo)
    lo, hi = float(cfg.get("lo", 0.4)), float(cfg.get("hi", 3.0))
    k_drop = int(cfg.get("drop_preintv_min", 4))
    # k=0 이면 모든 샘플이 PI 가 되는데 학습(train.py)은 k=0 을 '제외 안 함' 으로 읽는다
    if k_drop < 1 or lo > hi:
        raise ValueError(f"drop_preintv_min={k_drop} 은 1 이상, lo={lo} <= hi={hi} 여야 한다")
    prev = cfg.get("prev")

    meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, meta)
    policy, _, _ = make_policy(cfg, meta, stats)
    ds = create_dataset(policy, cfg)
    fl, ep = frame_labels(cfg.task.dataset_root, n_demo)
    judge, judge_name = judge_slots(ds, ep)
    pi = ((fl[action_windows(ds, ep)] == LABELS["preintv"]) & judge).sum(1) >= k_drop
    N = len(ds)
    w = np.ones(N, dtype=np.float32)
    drop = np.zeros(N, dtype=bool)

    # 이전 데이터셋이 새 데이터셋의 앞부분이어야 프레임 번호로 가중을 옮길 수 있다
    n_prev = 0
    if prev:
        p = np.load(prev)
        n_prev = len(p["w"])
        if not (n_prev <= N and np.array_equal(p["episode_index"], ep[:n_prev])
                and (n_prev == N or ep[n_prev] != ep[n_prev - 1])):
            raise ValueError(f"이전 라운드 가중의 에피소드 구성이 이 데이터셋의 앞부분과 다르다: {prev}")
        if int(p["drop_preintv_min"]) != k_drop:
            raise ValueError(f"drop_preintv_min 이 이전 라운드({int(p['drop_preintv_min'])})와 다르다")
        # 시연/배포 경계가 라운드마다 달라지면 학습의 frame_labels 와 앞부분의 drop 이 갈린다
        if int(p["n_demo_episodes"]) != n_demo:
            raise ValueError(f"n_demo_episodes 가 이전 라운드({int(p['n_demo_episodes'])})와 다르다")
        # 규약이 다른 앞부분의 drop 을 그대로 옮기면 학습이 다시 센 drop 과 갈린다
        prev_judge = str(p["judge"]) if "judge" in p.files else "(없음: 16칸 판정)"
        if prev_judge != judge_name:
            raise ValueError(f"이전 라운드 가중의 판정 규약 {prev_judge} != {judge_name}: {prev}")
        if not (np.isfinite(p["w"]).all() and (p["w"] >= 0).all()):
            raise ValueError(f"이전 라운드 가중의 w 에 NaN·inf·음수가 있다: {prev}")
        w[:n_prev], drop[:n_prev] = p["w"], p["drop"]

    new = (ep >= n_demo) & (np.arange(N) >= n_prev)
    new_pi, new_ok = new & pi, new & ~pi
    w[new_pi], drop[new_pi] = 0.0, True
    # 평균 1 로 맞출 대상이 없다 = n_demo_episodes · prev · drop_preintv_min 중 하나가 잘못됐을 가능성이 크다
    if not new_ok.any():
        raise ValueError(f"가중할 새 비-PI 샘플이 없다 (새 {int(new.sum())} · PI {int(new_pi.sum())}) — "
                         "n_demo_episodes · prev · drop_preintv_min 확인")

    sc = np.load(cfg.scores)
    # 규약이 없는 점수 = 2026-10-09 이전의 L1 (t-1 칸·고정된 복사 칸 포함 16칸 평균)
    sc_judge = str(sc["judge"]) if "judge" in sc.files else "(없음: 16칸 L1)"
    if sc_judge != judge_name:
        raise ValueError(f"점수 파일의 판정 규약 {sc_judge} != {judge_name}: {cfg.scores}")
    si = sc["index"]
    if not (((si >= 0) & (si < N)).all() and np.array_equal(sc["episode_index"], ep[si])):
        raise ValueError(f"점수 파일의 프레임 번호·에피소드 번호가 이 데이터셋과 다르다: {cfg.scores}")
    # 조각을 손으로 합친 파일에 같은 프레임이 두 번 있으면 아래 대입에서 마지막 값이 조용히 이긴다
    if len(np.unique(si)) != len(si):
        raise ValueError(f"점수 파일에 같은 프레임 번호가 여러 번 있다 ({len(si) - len(np.unique(si))} 개): {cfg.scores}")
    l1 = np.full(N, np.nan)
    l1[si] = sc["l1_mean"]
    miss = new_ok & ~np.isfinite(l1)
    if miss.any():
        raise ValueError(f"새 비-PI 프레임 {int(miss.sum())} 개에 점수가 없다 (첫 프레임 {int(np.flatnonzero(miss)[0])})")
    # L1 은 절댓값 평균이라 음수일 수 없고, 전부 0 이면 평균 대비 비가 0/0 이다 — 채점이 망가진 신호
    if (l1[new_ok] < 0).any() or l1[new_ok].mean() <= 0:
        raise ValueError(f"새 비-PI 프레임의 L1 에 음수가 있거나 전부 0 이다: {cfg.scores}")
    r = l1[new_ok] / l1[new_ok].mean()
    clip_lo, clip_hi = float((r < lo).mean()), float((r > hi).mean())
    r = np.clip(r, lo, hi)
    w[new_ok] = r / r.mean()

    q = np.percentile(w[new_ok], [0, 5, 25, 50, 75, 95, 100])
    np.savez(cfg.out, w=w, drop=drop, episode_index=ep, judge=judge_name,
             n_prev=n_prev, n_new=int(new.sum()), n_new_pi=int(new_pi.sum()),
             w_new_quantiles=q, clip_lo_frac=clip_lo, clip_hi_frac=clip_hi,
             lo=lo, hi=hi, drop_preintv_min=k_drop, n_demo_episodes=n_demo,
             scores=str(cfg.scores), prev=str(prev or ""), dataset_root=str(cfg.task.dataset_root))
    print(f"저장: {cfg.out}   샘플 {N} = 시연 {int((ep < n_demo).sum())} + 이전 라운드 "
          f"{n_prev - int((ep[:n_prev] < n_demo).sum())} + 새 {int(new.sum())}")
    print(f"  새: PI 제외 {int(new_pi.sum())} · 가중 {int(new_ok.sum())}  "
          f"clip 하한 {clip_lo:.1%} · 상한 {clip_hi:.1%}")
    print("  새 w 분위 (0/5/25/50/75/95/100%): " + " ".join(f"{v:.3f}" for v in q))


if __name__ == "__main__":
    main()
