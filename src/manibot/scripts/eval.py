"""Evaluation entry point: load a checkpoint and roll out in simulation.

Reports rollout success rate — the measure this project judges policies by.
Shares make_predict_fn and eval_policy with the online validation inside
scripts/train.py, so the number printed here and the number logged during
training come from the same code.
"""

import json
import logging
import random
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
from manibot.utils.eval import eval_policy
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


@hydra.main(config_path="../configs", config_name="default_policy", version_base="1.3")
def evaluate(cfg: DictConfig):
    setup_logging(save_dir=cfg.log_dir, debug=cfg.debug)
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
    logger.info(f"Loaded checkpoint: {checkpoint} (EMA {'적용' if used_ema else '없음'})")

    policy.eval()
    env = make_eval_env(cfg.task)
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
            videos_dir=Path(cfg.eval_dir) / "videos",
            max_episodes_rendered=cfg.val.num_viz_videos,
            video_key=image_keys[0] if image_keys else None,
            # ⭐ 배포·실물과 같은 조건으로 잰다 (`utils/eval.py:rollout_episode` 참조)
            merger_name=cfg.eval_merger, te_coeff=cfg.eval_te_coeff,
            anchor_offset=cfg.anchor_offset,
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
    out = Path(cfg.eval_dir) / "eval_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"checkpoint": str(checkpoint), "config": OmegaConf.to_container(cfg.task, resolve=True),
               **info}, open(out, "w"), indent=2)
    logger.info(f"Wrote {out}")


def main():
    evaluate()


if __name__ == "__main__":
    main()
