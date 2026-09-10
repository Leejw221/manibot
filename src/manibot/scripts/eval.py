"""Evaluation entry point: load a checkpoint and roll out in simulation.

Reports rollout success rate — the measure this project judges policies by.
Shares make_predict_fn and eval_policy with the online validation inside
scripts/train.py, so the number printed here and the number logged during
training come from the same code.
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

from manibot.policies.factory import make_policy
from manibot.rollout import make_predict_fn
from manibot.utils.checkpoints import (get_best_checkpoint, get_latest_checkpoint,
                                       load_ema_weights, load_model_weights)
from manibot.utils.dataset_utils import create_dataset_stats
from manibot.utils.eval import _unwrap, eval_policy
from manibot.utils.seeding import seed_sim_env
from manibot.utils.logger import setup_logging
from manibot.utils.task_utils import derive_task_meta, is_sim_task, make_eval_env

logger = logging.getLogger(__name__)


class _DeployCollector:
    """롤아웃을 **배포 데이터**로 같이 저장한다 (`utils/deploy_dataset` 공용 코드).

    개입이 없으므로 모든 프레임이 `action_mode=0`(rollout) 이다. `collect_intervention`
    으로 모은 것과 형식·라벨이 같아 그대로 합칠 수 있다.
    """

    def __init__(self, cfg, raw, cams, low_dim):
        from manibot.utils import deploy_dataset as dd

        self.dd, self.raw, self.cams, self.low_dim = dd, raw, cams, low_dim
        c = cfg.collect
        self.task, self.repo_id, self.root = c.single_task, c.repo_id, c.get("root")
        self.save_every, self.use_videos = c.save_every, c.use_videos
        self.parallel = c.parallel_encoding
        self.fps, self.robot_type = int(cfg.task.fps), str(cfg.task.sim.robots)
        self.features = dd.make_features(raw._get_observations(), cams, low_dim,
                                         raw.action_dim, self.use_videos)
        self.ds = self._open()
        self.ep_success = dd.read_success(self.ds.root)
        self.n = 0
        self.frames = []

    def _open(self):
        return self.dd.open_or_resume(self.repo_id, self.root, self.fps, self.features,
                                      self.robot_type, self.use_videos)

    def start(self, ep):
        self.frames = []

    def record(self, action):
        self.frames.append(self.dd.make_frame(self.raw._get_observations(), self.cams,
                                              self.low_dim, action, 0, self.task))

    def finish(self, ep, success):
        for fr in self.frames:
            self.ds.add_frame(fr)
        self.ds.save_episode(parallel_encoding=bool(self.parallel))
        self.ep_success.append(bool(success))
        self.n += 1
        if self.save_every > 0 and self.n % self.save_every == 0:
            self.ds.finalize()
            self.dd.write_success(self.ds.root, self.ep_success)
            self.ds = self._open()

    def close(self):
        self.ds.finalize()
        self.dd.write_success(self.ds.root, self.ep_success)
        logger.info(f"배포 데이터 {len(self.ep_success)} 에피소드 -> {self.ds.root}")


def _eval_dir(cfg, checkpoint: Path) -> Path:
    """평가 결과를 **평가한 체크포인트 옆에** 둔다.

        <학습 세션>/eval/<step 이름>_<시각>/

    기본 output_dir 을 쓰면 eval 을 돌릴 때마다 체크포인트와 무관한 새 세션 폴더가 생겨서,
    어느 체크포인트의 성적인지 json 을 열어야 알 수 있다. 체크포인트가
    `<세션>/checkpoints/<step>` 모양이 아니면 기본 경로로 떨어진다.
    """
    ck = Path(checkpoint)
    if ck.parent.name == "checkpoints":
        return ck.parent.parent / "eval" / f"{ck.name}_{datetime.now():%Y%m%d_%H%M%S}"
    return Path(cfg.eval_dir)


@hydra.main(config_path="../configs", config_name="default_policy", version_base="1.3")
def evaluate(cfg: DictConfig):
    if not is_sim_task(cfg.task):
        raise ValueError(f"task '{cfg.task.name}' 은 실물이라 시뮬 rollout 평가를 할 수 없다.")

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # 학습과 같은 통계·차원을 쓴다. 체크포인트를 로드하면 그 통계가 다시 덮어쓴다
    # (정규화 버퍼가 state_dict 에 들어 있다) — 학습 때 쓰던 정규화가 그대로 복원된다.
    dataset_meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, dataset_meta)

    policy, preprocessor, postprocessor = make_policy(cfg, dataset_meta, stats)
    policy = policy.to(cfg.device)

    checkpoint = Path(cfg.checkpoint_path) if cfg.checkpoint_path else (
        get_best_checkpoint(cfg.checkpoint_dir) or get_latest_checkpoint(cfg.checkpoint_dir)
    )
    if checkpoint is None:
        raise FileNotFoundError(f"체크포인트를 찾을 수 없다: {cfg.checkpoint_dir}")
    load_model_weights(policy, checkpoint, cfg.device)
    # 학습 중 검증이 EMA 를 쓰므로 여기서도 맞춘다 — 안 맞추면 같은 체크포인트인데
    # 학습 로그의 수치와 여기 수치가 갈린다 (checkpoints.load_ema_weights 참고).
    used_ema = cfg.get("use_ema", True) and load_ema_weights(policy, checkpoint, cfg.device)
    eval_dir = _eval_dir(cfg, checkpoint)
    setup_logging(save_dir=str(eval_dir), log_file="eval.log", debug=cfg.debug)
    logger.info(f"Loaded checkpoint: {checkpoint} (EMA {'적용' if used_ema else '없음'})")
    logger.info(f"결과를 {eval_dir} 에 쓴다")

    policy.eval()
    env = make_eval_env(cfg.task)
    seed_sim_env(_unwrap(env), cfg.seed)
    image_keys = list(cfg.task.image_keys)
    try:
        viewer = None
        if cfg.view:
            from manibot.utils.viewer import SimViewer
            viewer = SimViewer(OmegaConf.to_container(cfg.task.sim.cameras, resolve=True),
                               res=cfg.view_res, fps=cfg.view_fps, title="manibot eval")
            logger.info(f"창을 띄운다 (view_fps={cfg.view_fps} · 0 이면 최고 속도)")
        collector = None
        if cfg.collect.enable:
            raw = env
            for _ in range(4):
                if hasattr(raw, "robots"):
                    break
                raw = raw.env
            cams = OmegaConf.to_container(cfg.task.sim.cameras, resolve=True)
            collector = _DeployCollector(cfg, raw, cams, list(cfg.task.sim.state_from))
        info = eval_policy(
            env,
            make_predict_fn(policy, cfg, cfg.device,
                            preprocessor=preprocessor, postprocessor=postprocessor),
            cfg.val.eval_n_episodes,
            obs_horizon=cfg.policy.obs_horizon,
            action_horizon=cfg.policy.action_horizon,
            max_steps=cfg.task.sim.max_steps,
            fps=cfg.task.fps,
            videos_dir=eval_dir / "videos",
            max_episodes_rendered=cfg.val.num_viz_videos,
            video_key=image_keys[0] if image_keys else None,
            # ⭐ 배포·실물과 같은 조건으로 잰다 (`utils/eval.py:rollout_episode` 참조)
            merger_name=cfg.eval_merger, te_coeff=cfg.eval_te_coeff,
            async_infer=cfg.eval_async_infer,
            collector=collector, viewer=viewer,
        )
    finally:
        if viewer is not None:
            viewer.close()
        if collector is not None:
            collector.close()
        env.close()

    logger.info(f"Eval metrics: {info['aggregated']}")
    # 샘플러 설정을 결과에 남긴다 — 없으면 나중에 "이 수치가 K 몇이었나"를
    # 실행 시각으로 되짚어야 한다 (2026-09-10 에 실제로 그랬다).
    sampler = OmegaConf.to_container(cfg.policy.noise_scheduler, resolve=True)
    out = eval_dir / "eval_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"checkpoint": str(checkpoint),
               "config": OmegaConf.to_container(cfg.task, resolve=True),
               "sampler": sampler, "n_episodes": cfg.val.eval_n_episodes,
               "use_ema": bool(used_ema), "seed": cfg.seed,
               **info}, open(out, "w"), indent=2)
    logger.info(f"Wrote {out}")

    if cfg.wandb.enable and cfg.wandb.project:
        _log_eval_to_wandb(cfg, checkpoint, sampler, info, eval_dir)


def _train_run_name(checkpoint):
    """체크포인트가 나온 학습 run 의 wandb 이름. 평가 run 을 거기에 붙여 부르려는 것."""
    cfgs = sorted((checkpoint.parent.parent / "logs").glob("train_config_*.yaml"))
    if not cfgs:
        return None
    name = OmegaConf.load(cfgs[-1]).get("wandb", {}).get("name")
    # 학습 때 해석 안 된 보간(${task.name}-${session})이 그대로 남은 런이 있다
    return None if (name is None or "${" in str(name)) else str(name)


def _log_eval_to_wandb(cfg, checkpoint, sampler, info, eval_dir):
    """평가 결과를 학습과 같은 프로젝트에 별도 run 으로 올린다.

    학습 run 에 resume 하지 않는 이유: 같은 체크포인트를 K 여러 개로 평가하므로
    한 run 에 섞이면 어느 수치가 어느 샘플러인지 구별이 안 된다.
    """
    import wandb

    base = _train_run_name(checkpoint) or checkpoint.parent.parent.name
    run_name = f"{base}-eval-K{sampler.get('num_inference_steps')}"
    run = wandb.init(
        project=cfg.wandb.project, entity=cfg.wandb.entity, name=run_name,
        job_type="eval", dir=str(eval_dir),
        config={"checkpoint": str(checkpoint), "train_run": base,
                "sampler": sampler, "n_episodes": cfg.val.eval_n_episodes,
                "task": cfg.task.name, "seed": cfg.seed,
                "max_steps": cfg.task.sim.max_steps},
    )
    run.log({f"eval/{k}": v for k, v in info["aggregated"].items()})
    run.summary["success_episodes"] = sum(1 for e in info["per_episode"] if e["success"])
    run.finish()
    logger.info(f"wandb: {run_name}")


def main():
    evaluate()


if __name__ == "__main__":
    main()
