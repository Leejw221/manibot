"""Training entry point.

The loop lives here rather than in a separate runners/ package: LeRobot keeps
its training loop in scripts/lerobot_train.py, and a single file is one fewer
place for the sim and real paths to drift apart — which is the whole reason
this repository exists.
"""

import os
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from manibot.policies.factory import make_policy
from manibot.utils.checkpoints import get_latest_checkpoint, load_model_weights
from manibot.utils.dataset_utils import (
    create_dataloader,
    create_dataset,
    create_dataset_stats,
    cycle,
    log_dataset_image_resolution,
)
from manibot.rollout import make_predict_fn
from manibot.utils.eval import eval_policy
from manibot.utils.logger import TrainLogger, setup_logging
from manibot.utils.logging_utils import AverageMeter, MetricsTracker
from manibot.utils.task_utils import derive_task_meta, is_sim_task, make_eval_env

logger = logging.getLogger(__name__)



def _build_optimizer(policy, cfg):
    """LeRobot 정책은 get_optim_params() 로 파라미터만 주고, 우리 이전 정책들은
    스스로 옵티마이저를 만든다(ACT 는 백본에 다른 lr 을 쓴다)."""
    if hasattr(policy, "get_optimizer"):
        return policy.get_optimizer()
    return torch.optim.AdamW(
        policy.get_optim_params(),
        lr=cfg.optimizer_lr, betas=tuple(cfg.optimizer_betas),
        eps=cfg.optimizer_eps, weight_decay=cfg.optimizer_weight_decay,
    )


def _build_scheduler(policy, optimizer, num_steps, cfg):
    if hasattr(policy, "get_scheduler"):
        return policy.get_scheduler(optimizer, num_steps)
    from diffusers.optimization import get_scheduler
    return get_scheduler(
        cfg.scheduler_name, optimizer,
        num_warmup_steps=cfg.scheduler_warmup_steps, num_training_steps=num_steps,
    )


def _build_ema(policy, cfg):
    if hasattr(policy, "get_ema"):
        return policy.get_ema()
    if not cfg.get("use_ema", True):
        return None
    from diffusers.training_utils import EMAModel
    return EMAModel(parameters=policy.parameters(), power=cfg.ema_power)


class PolicyTrainer:
    def __init__(
        self,
        config,
        network,
        device,
        train_dataloader=None,
        val_dataloader=None,
        eval_env=None,
        preprocessor=None,
        postprocessor=None,
    ):
        self.config = config
        # LeRobot 은 정규화를 정책 밖(프로세서 파이프라인)에서 한다. 정규화가 정책 안에
        # 있는 정책들에는 항등 함수가 들어와 같은 루프가 둘 다 다룬다.
        self.preprocessor = preprocessor if preprocessor is not None else (lambda b: b)
        self.postprocessor = postprocessor if postprocessor is not None else (lambda a: a)
        self.network = network.to(device)
        self.device = device
        self.step_counter = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.eval_env = eval_env

        num_total_steps = self.config.train.steps

        self.optimizer = _build_optimizer(network, config)
        self.lr_scheduler = _build_scheduler(network, self.optimizer, num_total_steps, config)
        self.ema = _build_ema(network, config)

        # AMP (Automatic Mixed Precision)
        self.use_amp = config.train.get("use_amp", True)
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.amp_dtype == torch.float16))
        if self.use_amp:
            logger.info(f"AMP enabled: {self.amp_dtype}")

        self.train_logger = TrainLogger(config)

        self.train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grad_norm", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("update_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }

        self.best_metrics = {
            "pc_success": -float("inf"),
            "avg_max_reward": -float("inf"),
        }

        dataset = train_dataloader.dataset

        self.train_tracker = MetricsTracker(
            self.config.train.batch_size,
            dataset.num_frames,
            dataset.num_episodes,
            self.train_metrics,
            initial_step=self.step_counter
        )

    def train_step(self, batch):
        """Single training step."""
        import time
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device, non_blocking=True)

        # Forward pass
        start_time = time.perf_counter()
        self.network.train()
        self.optimizer.zero_grad()

        batch = self.preprocessor(batch)
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            loss, output_dict = self.network.forward(batch)

        # Backward pass
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.config.train.grad_clip_norm,
            error_if_nonfinite=False,
        )

        # Optimizer step
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.ema is not None:
            self.ema.step(self.network.parameters())

        # Update metrics
        self.train_tracker.loss = loss.item()
        self.train_tracker.grad_norm = grad_norm.item()
        self.train_tracker.lr = self.optimizer.param_groups[0]["lr"]
        self.train_tracker.update_s = time.perf_counter() - start_time

        return output_dict

    def validate_offline(self):
        # Evaluate the EMA weights (matches validate_online() and what's actually
        # deployed at inference) instead of the raw in-training weights.
        use_ema = self.config.use_ema and self.ema is not None
        if use_ema:
            self.ema.store(self.network.parameters())
            self.ema.copy_to(self.network.parameters())

        self.network.eval()
        val_metrics = defaultdict(float)
        num_samples = 0

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_dataloader):
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(self.device, non_blocking=True)

                    if not hasattr(self.network, "validate"):
                        raise NotImplementedError(
                            "오프라인 검증은 network.validate() 를 쓰는데 이 정책엔 없다. "
                            "val.num_episodes=0 으로 두고 rollout 성공률로 판단한다."
                        )
                    batch = self.preprocessor(batch)
                    output_dict = self.network.validate(batch)

                    batch_size = next(iter(batch.values())).shape[0]
                    for k, v in output_dict.items():
                        val_metrics[k] += v * batch_size
                    num_samples += batch_size

                    if batch_idx == 0 and hasattr(self.network, 'visualize'):
                        viz_results = self.network.visualize(
                            batch,
                            num_samples=self.config.val.num_viz_samples
                        )

                        if viz_results:
                            self.train_logger.log_figures(
                                viz_results,
                                prefix='plot',
                                step=self.step_counter
                            )
        finally:
            if use_ema:
                self.ema.restore(self.network.parameters())

        val_metrics = {k: v / num_samples for k, v in val_metrics.items()}
        return val_metrics

    def validate_online(self):
        model = self.network

        if self.config.use_ema and self.ema is not None:
            self.ema.store(model.parameters())
            self.ema.copy_to(model.parameters())

        model.eval()
        cfg = self.config
        image_keys = list(cfg.task.image_keys)
        eval_info = eval_policy(
            self.eval_env,
            make_predict_fn(model, cfg, self.device,
                            preprocessor=self.preprocessor, postprocessor=self.postprocessor),
            cfg.val.eval_n_episodes,
            obs_horizon=cfg.policy.obs_horizon,
            action_horizon=cfg.policy.action_horizon,
            max_steps=cfg.task.sim.max_steps,
            fps=cfg.task.fps,
            videos_dir=Path(cfg.val_dir) / f"videos_step_{self.step_counter:010d}",
            max_episodes_rendered=cfg.val.num_viz_videos,
            video_key=image_keys[0] if image_keys else None,
        )

        if self.config.use_ema and self.ema is not None:
            self.ema.restore(model.parameters())

        return eval_info

    def train(self, num_steps=None, start_step=0):
        import time

        if num_steps is None:
            num_steps = self.config.train.steps

        self.step_counter = start_step
        dl_iter = cycle(self.train_dataloader)

        logger.info(f"Starting training from step {start_step} to {num_steps}")

        for step in range(start_step, num_steps):
            # Load data
            start_time = time.perf_counter()
            batch = next(dl_iter)
            self.train_tracker.dataloading_s = time.perf_counter() - start_time

            # Train step
            output_dict = self.train_step(batch)

            # Update step counter
            self.step_counter += 1
            self.train_tracker.step()

            # Check intervals
            is_log_step = (
                self.config.train.log_freq > 0 and
                self.step_counter % self.config.train.log_freq == 0
            )
            is_save_step = (
                self.config.train.save_freq > 0 and
                self.step_counter % self.config.train.save_freq == 0
            )
            is_val_offline_step = (
                self.val_dataloader is not None and
                self.config.val.val_offline_freq > 0 and
                self.step_counter % self.config.val.val_offline_freq == 0
            )
            is_val_online_step = (
                self.eval_env and
                self.config.val.val_online_freq > 0 and
                self.step_counter % self.config.val.val_online_freq == 0
            )

            # Log metrics
            if is_log_step:
                logger.info(self.train_tracker)
                wandb_log_dict = self.train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                self.train_logger.log_metrics(wandb_log_dict, prefix='train', step=self.step_counter)
                self.train_tracker.reset_averages()

            if is_val_offline_step:
                logger.info(f"Offline validation at step {self.step_counter}")
                val_metrics = self.validate_offline()

                if val_metrics:
                    logger.info(f"Validation metrics: {val_metrics}")
                    self.train_logger.log_metrics(
                        val_metrics,
                        prefix='val_offline',
                        step=self.step_counter
                    )

            # Online validation
            if is_val_online_step:
                logger.info(f"Online validation at step {self.step_counter}")
                eval_info = self.validate_online()

                # Log aggregated metrics
                logger.info(f"Eval metrics: {eval_info['aggregated']}")
                self.train_logger.log_metrics(
                    eval_info['aggregated'],
                    prefix='val_online',
                    step=self.step_counter
                )

                # Log video
                if eval_info['video_paths']:
                    self.train_logger.log_video_files(
                        {"eval_video": eval_info['video_paths'][0]},
                        prefix='val_online',
                        fps=self.config.task.fps,
                        step=self.step_counter
                    )

                # Save best checkpoints
                if self.config.train.save_best:
                    if eval_info['aggregated']['pc_success'] > self.best_metrics['pc_success']:
                        self.best_metrics['pc_success'] = eval_info['aggregated']['pc_success']
                        self.save_checkpoint(Path(self.config.checkpoint_dir) / "best_pc_success")

                    if eval_info['aggregated']['avg_max_reward'] > self.best_metrics['avg_max_reward']:
                        self.best_metrics['avg_max_reward'] = eval_info['aggregated']['avg_max_reward']
                        self.save_checkpoint(Path(self.config.checkpoint_dir) / "best_avg_max_reward")

            if is_save_step:
                self.save_checkpoint(Path(self.config.checkpoint_dir) / f"step_{self.step_counter:010d}")

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.network.save_pretrained(checkpoint_dir)

        # Save training state
        training_state = {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "ema": self.ema.state_dict() if self.ema else None,
            "step": self.step_counter,
        }
        torch.save(training_state, checkpoint_dir / "training_state.pt")

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)

        training_state_file = checkpoint_dir / "training_state.pt"

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        if not training_state_file.exists():
            raise FileNotFoundError(f"Training state file not found: {training_state_file}")

        # Load network weights into existing instance
        load_model_weights(self.network, checkpoint_dir, self.device)
        self.network.to(self.device)

        # Load training state
        # training_state may include OmegaConf objects; allow full unpickling from trusted checkpoints
        training_state = torch.load(
            training_state_file,
            map_location=self.device,
            weights_only=False
        )
        self.optimizer.load_state_dict(training_state["optimizer"])

        if self.lr_scheduler and training_state.get("lr_scheduler"):
            self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])

        if self.ema and training_state.get("ema"):
            self.ema.load_state_dict(training_state["ema"])

        step = training_state.get("step", 0)
        self.step_counter = step

        logger.info(f"Loaded checkpoint from {checkpoint_dir} at step {step}")
        return step


@hydra.main(config_path="../configs", config_name="default_policy", version_base="1.3")
def train(cfg: DictConfig):

    # Require explicit wandb project name
    if cfg.wandb.enable and not cfg.wandb.project:
        print("ERROR: wandb.project를 지정해주세요.")
        print("  예시: wandb.project=\"Advanced AI Term-Project\"")
        raise SystemExit(1)

    # Setup
    setup_logging(save_dir=cfg.log_dir, debug=cfg.debug)
    logger.info("Starting training...")
    logger.info(f"Using policy: {cfg.policy.name}")

    # Set seeds
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Create directories and save config
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d%H%M")
    OmegaConf.save(config=cfg, f=cfg.log_dir / Path(f"train_config_{time_str}.yaml"))

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create dataset stats. The dataset is the single source of truth for the
    # observation/action dimensions — a hand-written value in task.yaml that
    # drifts from the data has caused a silent mismatch before.
    dataset_meta, stats = create_dataset_stats(cfg)
    derive_task_meta(cfg.task, dataset_meta)
    log_dataset_image_resolution(cfg, dataset_meta)

    # Create policy. make_policy also returns the normalization processors:
    # LeRobot keeps normalization outside the module, ours keep it inside, and
    # the trainer speaks one contract either way.
    policy, preprocessor, postprocessor = make_policy(cfg, dataset_meta, stats)
    num_params = sum(p.numel() for p in policy.parameters())
    num_trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params/1e6:.2f}M | Trainable params: {num_trainable_params/1e6:.2f}M")

    # Create dataset and dataloader
    total_episodes = dataset_meta.num_episodes
    if cfg.val.num_episodes != None and cfg.val.num_episodes > 0:
        train_dataset = create_dataset(policy, cfg, list(range(total_episodes - cfg.val.num_episodes)))
        train_dataloader = create_dataloader(train_dataset, cfg, is_training=True)
        val_dataset = create_dataset(policy, cfg, list(range(total_episodes - cfg.val.num_episodes, total_episodes)))
        val_dataloader = create_dataloader(val_dataset, cfg, is_training=False)

        logger.info(f"Train: {train_dataset.num_frames} frames | {train_dataset.num_episodes} episodes | {len(train_dataloader)} batches")
        logger.info(f"Val: {val_dataset.num_frames} frames | {val_dataset.num_episodes} episodes | {len(val_dataloader)} batches")
    else:
        train_dataset = create_dataset(policy, cfg)
        train_dataloader = create_dataloader(train_dataset, cfg, is_training=True)
        val_dataloader = None
        logger.info(f"Train: {train_dataset.num_frames} frames | {train_dataset.num_episodes} episodes | {len(train_dataloader)} batches")

    # Online evaluation: rollout success rate. This is what training is judged
    # by — offline loss is not it.
    eval_env = None
    if cfg.val.val_online_freq > 0:
        if not is_sim_task(cfg.task):
            raise ValueError(
                f"task '{cfg.task.name}' 은 실물이라 학습 중 rollout 평가를 할 수 없다. "
                "val.val_online_freq=0 으로 두고 별도로 평가한다."
            )
        eval_env = make_eval_env(cfg.task)

    runner = PolicyTrainer(
        cfg,
        policy,
        device=cfg.device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        eval_env=eval_env,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    # Resume from checkpoint if specified
    start_step = 0
    if cfg.resume:
        if cfg.checkpoint_path:
            checkpoint_path = Path(cfg.checkpoint_path)
        else:
            checkpoint_path = get_latest_checkpoint(cfg.checkpoint_dir)

        if checkpoint_path:
            start_step = runner.load_checkpoint(checkpoint_path)
            logger.info(f"Resumed from checkpoint at step {start_step}")
        else:
            logger.warning(f"No checkpoint found, starting from scratch")

    # Training loop
    runner.train(
        num_steps=cfg.train.steps,
        start_step=start_step,
    )

    # Cleanup
    if eval_env:
        eval_env.close()

    logger.info("Training completed!")


def main():
    train()


if __name__ == "__main__":
    main()
