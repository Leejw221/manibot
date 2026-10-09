"""정책이 각 샘플의 행동 청크를 얼마나 재현하나 — 샘플별 L1 을 미리 잰다 (라운드당 1회).

왜: 라운드 학습의 샘플 가중을 클래스가 아니라 "직전 라운드 정책이 이미 하는 행동인가" 로 정한다.
샘플의 관측을 조건으로 청크를 M 번 생성해 기록된 청크와 비교한다 (build_sample_weights 가 쓴다).

재현성: DDIM(eta=0)은 초기 노이즈가 같으면 같은 청크를 낸다. 초기 노이즈를 (noise_seed, 프레임
인덱스) 로 시드한 CPU 생성기로 만들어, 배치 크기·처리 범위가 달라도 같은 샘플은 같은 노이즈를 받는다.
정책은 eval — 이미지 랜덤 크롭 대신 센터 크롭 (LeRobot DiffusionRgbEncoder 가 self.training 으로 가른다).

L1 = |생성 - 기록| 을 판정 칸(프레임 t..t+14 중 에피소드 안, judge_slots) x 행동 차원 평균 (정규화된 행동 공간).
t-1 칸은 추론 때 실행되지 않고 패딩 칸은 경계 프레임의 복사라 뺀다 [사용자 결정 2026-10-09].
산출: <out>.npz — index · episode_index · l1 (N, M) · l1_mean · l1_min · n_valid_slots · judge · 메타
사용 (기본 범위 = 첫 비시연 프레임부터 끝까지):
    python -m manibot.scripts.score_samples task=square_ph50_r1 policy=diffusion \
        +checkpoint=outputs/.../step_0000050000 +n_demo_episodes=50 +M=8 +out=.../scores_r1.npz
"""

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from safetensors import safe_open

from manibot.losses.sample_weight import judge_slots
from manibot.policies.diffusion_ops import prepare_cond
from manibot.policies.factory import make_policy
from manibot.utils.checkpoints import load_ema_weights, load_model_weights
from manibot.utils.dataset_utils import create_dataset, create_dataset_stats
from manibot.utils.task_utils import derive_task_meta

# 체크포인트 config.json 과 대조하지 않는 필드 — 기기·허브·컴파일과 학습에만 쓰는 값(옵티마이저·스케줄러·
# 샘플러·손실 마스크·랜덤 크롭 여부). 채점(eval 모드 생성)의 결과를 바꾸지 않는다.
CONFIG_IGNORE = {
    "device", "use_amp", "push_to_hub", "repo_id", "private", "tags", "license",
    "pretrained_path", "pretrained_revision", "compile_model", "compile_mode", "gradient_checkpointing",
    "optimizer_lr", "optimizer_betas", "optimizer_eps", "optimizer_weight_decay",
    "scheduler_name", "scheduler_warmup_steps", "drop_n_last_frames", "do_mask_loss_for_padding",
    "crop_is_random",
}


@hydra.main(version_base="1.3", config_path="../configs", config_name="default_policy")
def main(cfg):
    ckpt = cfg.checkpoint
    out = Path(cfg.out)              # 채점(수천 프레임)을 다 한 뒤가 아니라 시작 전에 없으면 실패하게
    M = int(cfg.get("M", 8))
    bs = int(cfg.get("batch_size", 32))
    seed = int(cfg.get("noise_seed", 0))
    device = cfg.device
    if M < 1:
        raise ValueError(f"+M={M} 은 1 이상이어야 한다")

    meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, meta)
    policy, pre, _ = make_policy(cfg, meta, stats)
    policy = policy.to(device).eval()
    load_model_weights(policy, ckpt, device)
    # load_model_weights 는 strict=False — 키가 안 맞으면 무작위 초기화 정책으로 채점하게 된다
    st = Path(ckpt) / "model.safetensors"
    if st.exists():
        with safe_open(str(st), "pt") as f:
            diff = set(f.keys()) ^ set(policy.state_dict())
        if diff:
            raise ValueError(f"체크포인트 키가 정책과 다르다 ({len(diff)} 개, 예: {sorted(diff)[:3]}): {ckpt}")
    # 키 이름·모양이 같아도 크롭·horizon·스케줄러가 다르면 다른 정책이다 — 학습 때 설정과 대조한다
    cj = Path(ckpt) / "config.json"
    if cj.exists():
        import draccus
        from lerobot.configs.policies import PreTrainedConfig
        was = json.loads(cj.read_text())
        now = json.loads(json.dumps(draccus.encode(policy.config, PreTrainedConfig)))
        bad = sorted(k for k in (was.keys() & now.keys()) - CONFIG_IGNORE if was[k] != now[k])
        if bad:
            raise ValueError("체크포인트 config.json 과 지금 정책 설정이 다르다 (task·policy override 확인): "
                             + "; ".join(f"{k} {was[k]} -> {now[k]}" for k in bad[:5]) + f"  ({ckpt})")
    else:
        print(f"  config.json 없음 — 정책 설정 대조를 건너뛴다: {ckpt}")
    # use_ema=false 면 EMA 가 있어도 raw 로 채점한다 (eval.py 와 같은 뜻)
    used = bool(cfg.use_ema) and load_ema_weights(policy, ckpt, device)
    # 가중의 기준은 직전 정책의 EMA — 못 실으면 raw 로 조용히 채점하지 않는다 (raw 는 use_ema=false 로 명시)
    if cfg.use_ema and not used:
        raise ValueError(f"EMA 가중치를 싣지 못했다 (training_state.pt 없음 또는 파라미터 수 불일치): {ckpt}")
    print(f"채점 정책 = {ckpt}  (EMA {'적용' if used else '없음'})")

    ds = create_dataset(policy, cfg)
    ep_all = np.asarray(ds.replay_buffer["episode_index"]).ravel()
    start = cfg.get("start_index")
    if start is None:
        n_demo = cfg.get("n_demo_episodes") or cfg.finetune.get("n_demo_episodes")
        if not n_demo:
            raise ValueError("+start_index 나 +n_demo_episodes 가 필요하다")
        start = int((ep_all < int(n_demo)).sum())   # 시연 에피소드가 앞에 있다
    start = int(start)
    end = cfg.get("end_index")
    end = len(ds) if end is None else int(end)      # 0 을 '없음' 으로 읽지 않는다
    if not 0 <= start < end <= len(ds):
        raise ValueError(f"채점 범위가 비었거나 데이터셋 밖이다: start_index={start} end_index={end} "
                         f"(프레임 {len(ds)} 개 · n_demo_episodes 확인)")
    judge, judge_name = judge_slots(ds, ep_all)
    out.parent.mkdir(parents=True, exist_ok=True)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(int(start), end)),
                                         batch_size=bs, shuffle=False,
                                         num_workers=cfg.train.num_workers)

    H, D = policy.config.horizon, policy.config.action_feature.shape[0]
    steps = policy.diffusion.num_inference_steps
    idx_l, l1_l, nv_l = [], [], []
    with torch.no_grad():
        for b in loader:
            idx = b["dataset_index"]
            valid = torch.from_numpy(judge[idx.numpy()]).float()      # (B, H) 판정 칸
            b = pre({k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()})
            cond, x0 = prepare_cond(policy, b)
            noise = torch.cat([torch.randn((M, H, D), generator=torch.Generator().manual_seed(seed * 10**8 + int(i)))
                               for i in idx])
            gen = policy.diffusion.conditional_sample(len(idx) * M, global_cond=cond.repeat_interleave(M, 0),
                                                      noise=noise.to(device, x0.dtype))
            err = (gen.view(len(idx), M, H, D) - x0[:, None]).abs().mean(-1).cpu()   # (B, M, H)
            l1_l.append((err * valid[:, None]).sum(-1) / valid.sum(-1, keepdim=True))
            nv_l.append(valid.sum(-1).long())
            idx_l.append(idx)
            print(f"  {int(idx[-1]) + 1 - int(start)}/{end - int(start)}", end="\r", flush=True)

    index = torch.cat(idx_l).numpy()
    # python -O 에서도 남게 assert 가 아니라 raise — 순서가 어긋나면 L1 이 다른 프레임에 붙는다
    if not np.array_equal(index, np.arange(int(start), end)):
        raise RuntimeError("처리 순서가 프레임 인덱스와 다르다")
    l1 = torch.cat(l1_l).numpy().astype(np.float32)
    # NaN 을 저장하면 build_sample_weights 의 평균까지 조용히 NaN 이 된다
    if not np.isfinite(l1).all():
        bad = index[~np.isfinite(l1).all(1)]
        raise ValueError(f"L1 이 유한하지 않은 프레임 {len(bad)} 개 (첫 프레임 {int(bad[0])}) — 저장하지 않는다")
    np.savez(out, index=index, episode_index=ep_all[index], l1=l1,
             l1_mean=l1.mean(1), l1_min=l1.min(1), n_valid_slots=torch.cat(nv_l).numpy(), judge=judge_name,
             checkpoint=str(ckpt), ema=used, M=M, seed=seed, num_inference_steps=steps,
             dataset_root=str(cfg.task.dataset_root))
    print(f"\n저장: {out}   프레임 [{start}, {end}) {len(index)} 개 x M={M}  (DDIM {steps})")
    q = np.percentile(l1.mean(1), [5, 50, 95])
    print(f"  l1_mean  5% {q[0]:.4f}  50% {q[1]:.4f}  95% {q[2]:.4f}")


if __name__ == "__main__":
    main()
