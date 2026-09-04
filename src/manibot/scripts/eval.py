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
from manibot.utils.checkpoints import get_best_checkpoint, get_latest_checkpoint, load_model_weights
from manibot.utils.dataset_utils import create_dataset_stats
from manibot.utils.eval import eval_policy
from manibot.utils.logger import setup_logging
from manibot.utils.task_utils import derive_task_meta, is_sim_task, make_eval_env

logger = logging.getLogger(__name__)


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
    logger.info(f"Loaded checkpoint: {checkpoint}")

    policy.eval()
    env = make_eval_env(cfg.task)
    image_keys = list(cfg.task.image_keys)
    try:
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
        )
    finally:
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
