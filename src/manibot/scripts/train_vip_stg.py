"""공유 백본(VIP) DP + STG 헤드 학습 — policies/vip_stg.py · datasets/joint_stg.py.

train.py 의 PolicyTrainer 를 그대로 쓰고 두 가지만 덧붙인다(train.py 는 건드리지 않는다):
  1. 전처리기가 정책이 아는 키만 통과시키므로 bc_mask·stg_mask·stg_label 을 다시 붙인다.
  2. 매 스텝 stg_weight = weight * min(1, step / (warmup_ratio * steps)).
시작 때 VIP 가중치를 싣고, 학습 데이터 전체 프레임의 VIP 특징으로 카메라별 표준화 통계를 정한다.

사용:
    # lambda 측정 (학습 대신 measure_lambda 스텝 동안 BC 만 학습하며 두 손실의 인코더 그래디언트 크기를 잰다)
    python -m manibot.scripts.train_vip_stg task=square_ph50 policy=diffusion_vip_stg \
        seed=0 policy.vip_stg.weight=0 policy.vip_stg.measure_lambda=300 wandb.enable=false val.val_online_freq=0
    # 학습
    python -m manibot.scripts.train_vip_stg task=square_ph50 policy=diffusion_vip_stg seed=0 \
        policy.vip_stg.weight=<lambda> train.steps=50000 val.val_online_freq=0 wandb.project=<project>
    # 16GB GPU 는 policy.vip_stg.grad_checkpoint=true · 중단 뒤 이어서: resume=true session=<같은 이름>
    # 라운드 학습 (R1~): 직전 라운드 STG 로 채점(score_stg)한 뒤 그 점수로 풀·BC 가중을 만든다
    python -m manibot.scripts.train_vip_stg task=square_ph50_vip_r1 policy=diffusion_vip_stg seed=0 \
        +policy.drop_n_last_frames=0 policy.vip_stg.stg_scores=<npz> policy.vip_stg.rate=two \
        policy.vip_stg.weight=<lambda> policy.vip_stg.val_fraction=0 train.steps=<샘플 비례> ...

라운드 학습 기록 (2026-10-22 사용자·희민 결정. 정본 = 노션 계획 페이지
https://app.notion.com/p/400bd012de0081159d75c0a54b2b0f2a 의 결정 목록 — 코드의 "결정 N" 은 그 번호)
- 가중: 시연·개입 = max(1, 진척률), 성공한 무개입 robot = 진척률 그대로(음수는 0, 하한 없음). 사용자: "1.0은 일반
  비가중치처럼 학습하는걸 얘기하는거야 최소한 그정도는 학습하자는거지". 이것을 배치 64 안의 몫으로 옮긴 것이
  batch_shares(policies/vip_stg.py) — 몫 = v/sum(v), 시연·개입은 몫 하한 1/64, 다시 나누지 않는다(사용자 "(가) 방식").
  이전 결정 5(robot 진척률 < 1 이면 버림)를 대신한다 — robot 도 버리지 않고 진척률만큼 배운다.
- 진척률 두 방식을 R1 에서 둘 다 학습해 비교한다(사용자 "할인 방식하고 양끝 방식은 둘이 round 1에서 비교"):
  rate=two 양 끝(결정 4) · rate=disc 행동 칸마다 진척의 0.9 할인 평균(희민의 할인 반환을 청크 안으로).
  base STG 로 잰 시연(정답 1) 진척률 표준편차: two 0.50 · disc 0.59 (~/claude/logs/stg_score/summary.json).
- 채점 STG: 라운드 N 데이터는 라운드 N-1 모델의 STG(EMA) — R1 = base, R2 = R1 모델 (결정 7, 사용자 10-22 재확인).
  학습 중 재학습되는 같은 라운드 STG 로 채점하지 않는다: 자기가 배운 에피소드의 라벨(매 스텝 1씩 감소)에 맞춰져
  느린 구간도 진척률 1 로 읽힌다.
- 스텝 = BC 샘플 수 / 7,118 x 50K (결정 11). R1(시연 7,468 + 개입 2,588 + robot 1,390 = 11,446) -> 80K.
- 우선 seed 0 만 (사용자 10-22 "3개 seed 모두 돌리지말고 seed 0에서만").
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from manibot.datasets.joint_stg import (
    JointBatchSampler,
    JointStgDataset,
    chunk_rates,
    round_pools,
    stg_pool,
    stg_val_episodes,
    steps_to_go,
)
from manibot.policies.vip_stg import make_vip_stg_policy
from manibot.scripts.train import PolicyTrainer
from manibot.utils.checkpoints import get_latest_checkpoint
from manibot.utils.dataset_utils import (
    EpisodeAwareSampler,
    create_dataset,
    create_dataset_stats,
    cycle,
)
from manibot.utils.logger import setup_logging
from manibot.utils.task_utils import derive_task_meta, is_sim_task, make_eval_env

logger = logging.getLogger(__name__)

STG_KEYS = ("bc_mask", "stg_mask", "stg_label", "bc_rate", "bc_floor")


class VipStgTrainer(PolicyTrainer):
    def __init__(self, config, network, *args, **kwargs):
        super().__init__(config, network, *args, **kwargs)
        vs = config.policy.vip_stg
        self.stg_weight_final = float(vs.weight)
        self.stg_warmup_steps = max(1, int(round(float(vs.warmup_ratio) * config.train.steps)))
        pre = self.preprocessor

        def keep_stg_keys(batch):
            kept = {k: batch[k] for k in STG_KEYS if k in batch}
            out = pre(batch)
            out.update({k: v.to(self.device) for k, v in kept.items()})
            return out

        self.preprocessor = keep_stg_keys

    def train_step(self, batch):
        self.network.stg_weight = self.stg_weight_final * min(1.0, self.step_counter / self.stg_warmup_steps)
        out = super().train_step(batch)
        if self.config.train.log_freq > 0 and (self.step_counter + 1) % self.config.train.log_freq == 0:
            logger.info("  STG  " + "  ".join(f"{k}={v:.4g}" for k, v in out.items()))
        return out


@torch.no_grad()
def fit_feature_stats(policy, dataset, preprocessor, cfg, batch_size=128):
    """카메라별 VIP 특징 평균(차원별)과 크기(차원별 표준편차의 RMS 스칼라). 프레임마다 현재 관측(창의 마지막)만 센다."""
    enc = policy.diffusion.rgb_encoder
    was_training = enc.training
    enc.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=cfg.train.num_workers)
    keys = list(cfg.task.image_keys)
    s1 = torch.zeros(len(keys), enc.feat_mean.shape[1], dtype=torch.float64, device=cfg.device)
    s2 = torch.zeros_like(s1)
    n = 0
    for batch in loader:
        batch = preprocessor(batch)
        for c, k in enumerate(keys):
            f = enc.raw_features(batch[k][:, -1].to(cfg.device)).double()
            s1[c] += f.sum(0)
            s2[c] += (f * f).sum(0)
        n += len(batch[keys[0]])
    mean = s1 / n
    var = (s2 / n - mean * mean).clamp_min(0)
    rms_std = var.mean(dim=1, keepdim=True).sqrt()
    enc.feat_mean.copy_(mean.float())
    enc.feat_std.copy_(rms_std.float().expand_as(enc.feat_std))
    enc.train(was_training)
    logger.info(f"VIP 특징 통계: 프레임 {n}개 · 카메라별 크기(RMS std) "
                + ", ".join(f"{k.split('.')[-1]}={float(rms_std[c]):.4f}" for c, k in enumerate(keys)))
    return {"n_frames": n, "rms_std": rms_std.squeeze(1).tolist()}


def round_weights(cfg, base, edi):
    """라운드 학습의 풀·진척률 — 직전 라운드 STG 점수(score_stg)에서 만든다. 분석 재료는 log_dir/round_weights.npz."""
    import zarr
    from lerobot.utils.constants import ACTION

    vs = cfg.policy.vip_stg
    if cfg.policy.get("drop_n_last_frames") != 0:
        raise ValueError("라운드 학습은 에피소드 끝 프레임까지 샘플로 쓴다(결정 16) — +policy.drop_n_last_frames=0")
    # 기본값 0.2 가 남으면 STG 풀에서만 에피소드가 조용히 빠진다 — BC·STG 풀은 같아야 한다(결정 8·9)
    if vs.val_fraction:
        raise ValueError("라운드 학습은 STG 검증 분할이 없다(결정 9) — policy.vip_stg.val_fraction=0")
    sc = np.load(vs.stg_scores)
    if len(sc["d"]) != len(base) or not np.array_equal(sc["episode_ends"], edi["to"]):
        raise ValueError(f"점수 파일의 프레임·에피소드 구성이 데이터셋과 다르다: {vs.stg_scores}")
    n_demo = int(vs.n_demo_episodes)
    keep, floor = round_pools(edi, zarr.open(str(cfg.task.dataset_root), "r")["data"]["action_mode"][:], n_demo)
    rate = chunk_rates(sc["d"], edi, base.delta_indices[ACTION], vs.rate, vs.gamma)
    ep = np.repeat(np.arange(len(edi["to"])), edi["to"] - edi["from"])
    kind = np.where(~keep, -1, np.where(ep < n_demo, 0, np.where(floor, 1, 2)))   # -1 제외 · 0 시연 · 1 개입 · 2 robot
    info = {"stg_scores": str(vs.stg_scores), "scored_by": str(sc["checkpoint"]), "rate": str(vs.rate),
            "gamma": float(vs.gamma), "excluded_frames": int((~keep).sum())}
    for k, name in enumerate(("demo", "intv", "robot")):
        r = rate[kind == k]
        info[name] = {"n": int(len(r)), "rate_mean": float(r.mean()) if len(r) else None,
                      "rate_le0": float((r <= 0).mean()) if len(r) else None}
    np.savez(Path(cfg.log_dir) / "round_weights.npz", rate=rate, keep=keep, floor=floor, kind=kind, d=sc["d"])
    return {"keep": keep, "floor": floor, "rate": rate, "info": info}


def measure_lambda(trainer, steps, every):
    """BC 만 학습하며(stg_weight=0, warmup 시작과 같다) every 스텝마다 두 손실이 인코더(학습 가능한 층)에 주는
    그래디언트 크기를 따로 잰다. lambda = |g_BC| / |g_STG| 의 뒤쪽 절반 중앙값을 제안값으로 낸다."""
    policy = trainer.network
    params = [p for p in policy.diffusion.rgb_encoder.parameters() if p.requires_grad]
    trainer.stg_weight_final = 0.0
    dl_iter = cycle(trainer.train_dataloader)
    rows = []
    for step in range(steps):
        batch = next(dl_iter)
        if step % every == 0:
            b = {k: (v.to(trainer.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            b = trainer.preprocessor(b)
            policy.train()
            with torch.autocast("cuda", dtype=trainer.amp_dtype, enabled=trainer.use_amp):
                sb = policy._stack_images(b)
                gc = policy.diffusion._prepare_global_conditioning(sb)
                bc, stg = sb["bc_mask"].bool(), sb["stg_mask"].bool()
                bc_loss = policy.diffusion.bc_loss(gc[bc], sb["action"][bc], sb["action_is_pad"][bc])
                stg_loss = torch.nn.functional.cross_entropy(
                    policy.diffusion.stg_head(gc[stg]).float(), sb["stg_label"][stg].long())
            g_bc = torch.autograd.grad(bc_loss, params, retain_graph=True, allow_unused=True)
            g_stg = torch.autograd.grad(stg_loss, params, allow_unused=True)
            norm = lambda gs: float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))  # noqa: E731
            nb, ns = norm(g_bc), norm(g_stg)
            rows.append({"step": step, "bc_loss": float(bc_loss), "stg_loss": float(stg_loss),
                         "g_bc": nb, "g_stg": ns, "ratio": nb / max(ns, 1e-12)})
            logger.info(f"[lambda] step {step}: bc={rows[-1]['bc_loss']:.4f} stg={rows[-1]['stg_loss']:.4f} "
                        f"|g_bc|={nb:.4g} |g_stg|={ns:.4g} ratio={rows[-1]['ratio']:.4g}")
        trainer.train_step(batch)
        trainer.step_counter += 1
    tail = [r["ratio"] for r in rows[len(rows) // 2:]]
    return {"rows": rows, "lambda": float(np.median(tail)), "rule": "median |g_bc|/|g_stg| over the later half"}


@hydra.main(config_path="../configs", config_name="default_policy", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.wandb.enable and not cfg.wandb.project:
        raise SystemExit("wandb.project 를 지정해 주세요 (train.py 와 같은 확인)")
    setup_logging(save_dir=cfg.log_dir, debug=cfg.debug)
    vs = cfg.policy.vip_stg
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=Path(cfg.log_dir) / f"train_config_{datetime.now():%Y%m%d%H%M}.yaml")

    meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, meta)
    policy, pre, post = make_vip_stg_policy(cfg, meta, stats)
    policy.diffusion.rgb_encoder.load_vip(vs.weights)
    policy = policy.to(cfg.device)
    n_all = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"VIP+STG 정책: 파라미터 {n_all / 1e6:.1f}M · 학습 {n_train / 1e6:.1f}M")

    base = create_dataset(policy, cfg)
    edi = {k: np.asarray(v) for k, v in base.episode_data_index.items()}
    rnd = round_weights(cfg, base, edi) if vs.get("stg_scores") else None
    # 특징 통계는 학습에 쓰는 프레임으로 — 라운드면 풀 밖(개입 전 robot) 프레임을 뺀다
    feat = fit_feature_stats(
        policy, base if rnd is None else torch.utils.data.Subset(base, np.flatnonzero(rnd["keep"])), pre, cfg)

    # BC 풀 = 기존 base 와 같은 샘플(create_dataloader 의 drop_n_last_frames 규칙)
    drop_last = cfg.policy.get("drop_n_last_frames",
                               cfg.policy.pred_horizon - cfg.policy.action_horizon - cfg.policy.obs_horizon + 1)
    bc_idx = EpisodeAwareSampler(base.episode_data_index, episode_indices_to_use=base.episodes,
                                 drop_n_last_frames=drop_last).indices
    val_eps = stg_val_episodes(len(edi["from"]), vs.val_fraction, vs.split_seed)
    stg_idx = stg_pool(edi, val_eps)
    labels = steps_to_go(edi)
    if rnd is not None:
        bc_idx = np.asarray(bc_idx)                      # EpisodeAwareSampler.indices 는 list
        bc_idx, stg_idx = bc_idx[rnd["keep"][bc_idx]], stg_idx[rnd["keep"][stg_idx]]
    dataset = JointStgDataset(base, stg_idx, labels, vs.num_bins,
                              *((rnd["rate"], rnd["floor"]) if rnd is not None else ()))
    sampler = JointBatchSampler(bc_idx, stg_idx, len(base), vs.bc_batch, vs.stg_batch, seed=cfg.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=cfg.train.num_workers,
                                         pin_memory=True, persistent_workers=cfg.train.num_workers > 0)
    info = {"bc_frames": len(bc_idx), "stg_frames": len(stg_idx), "stg_val_episodes": sorted(val_eps),
            "max_label": int(labels.max()), "clipped_labels": int((labels >= vs.num_bins).sum()),
            "steps_per_epoch": len(sampler), "feature_stats": feat,
            **({"round": rnd["info"]} if rnd is not None else {})}
    logger.info(f"joint STG: {json.dumps({k: v for k, v in info.items() if k != 'feature_stats'})}")
    (Path(cfg.log_dir) / "joint_stg_info.json").write_text(json.dumps(info, indent=1))

    eval_env_factory = None
    if cfg.val.val_online_freq > 0 and is_sim_task(cfg.task):
        eval_env_factory = lambda: make_eval_env(cfg.task)  # noqa: E731

    if vs.measure_lambda:
        trainer = VipStgTrainer(cfg, policy, device=cfg.device, train_dataloader=loader,
                                preprocessor=pre, postprocessor=post)
        res = measure_lambda(trainer, int(vs.measure_lambda), int(vs.measure_every))
        (Path(cfg.log_dir) / "lambda_measure.json").write_text(json.dumps(res, indent=1))
        logger.info(f"lambda 제안값 = {res['lambda']:.4g} ({res['rule']})")
        return

    trainer = VipStgTrainer(cfg, policy, device=cfg.device, train_dataloader=loader,
                            eval_env_factory=eval_env_factory, preprocessor=pre, postprocessor=post)
    start_step = 0
    if cfg.resume:
        # 체크포인트가 가중치·feat 통계(버퍼)·옵티마이저·EMA 를 되돌린다. 배치 순서(epoch)는 처음부터 다시 센다.
        ckpt = Path(cfg.checkpoint_path) if cfg.checkpoint_path else get_latest_checkpoint(cfg.checkpoint_dir)
        if ckpt:
            start_step = trainer.load_checkpoint(ckpt)
            logger.info(f"이어서 학습 <- {ckpt} (step {start_step})")
    trainer.train(num_steps=cfg.train.steps, start_step=start_step)
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
