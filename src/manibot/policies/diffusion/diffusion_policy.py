"""
Diffusion Policy for visuomotor manipulation.
Adapted from Chi et al., 2023 and LeRobot implementation.

Performs DDPM/DDIM diffusion directly in action space,
conditioned on visual observations via FiLM modulation.
Uses SpatialSoftmax vision encoder and MinMax normalization.
"""

import torch
import logging
import einops
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

from manibot.policies.factory import registry
from manibot.policies import BasePolicy
from manibot.policies.observers.resnet_observer import DiffusionRgbEncoder
from manibot.model.diffusion import ConditionalUnet1d
from manibot.utils.normalize import NormalizeMinMax, UnnormalizeMinMax
from manibot.utils.visualize import plot_joint_trajectories, plot_input_image

logger = logging.getLogger(__name__)

NOISE_SCHEDULER_REGISTRY = {
    "DDPM": DDPMScheduler,
    "DDIM": DDIMScheduler,
}


@registry.register_policy("diffusion")
class DiffusionPolicy(BasePolicy):
    def __init__(self, config, stats):
        super().__init__(config, stats)

        # Override normalization: MinMax for state and action (same as LeRobot)
        self.normalize_inputs = NormalizeMinMax(
            [config.task.state_key], stats
        )
        self.normalize_targets = NormalizeMinMax(
            [config.task.action_key], stats
        )
        self.unnormalize_outputs = UnnormalizeMinMax(
            [config.task.action_key], stats
        )

        # Vision encoder — DiffusionRgbEncoder (ResNet18 + GroupNorm + SpatialSoftmax)
        diffusion_cfg = config.policy
        self.use_separate_encoder = diffusion_cfg.get("use_separate_rgb_encoder_per_camera", False)
        num_cameras = len(config.task.image_keys)

        encoder_kwargs = dict(
            resize_shape=tuple(config.resize_shape) if config.resize_shape is not None else None,
            crop_shape=tuple(config.crop_shape) if config.crop_shape is not None else None,
            crop_is_random=diffusion_cfg.get("crop_is_random", True),
            spatial_softmax_num_keypoints=diffusion_cfg.get("spatial_softmax_num_keypoints", 32),
        )

        if self.use_separate_encoder:
            self.rgb_encoder = torch.nn.ModuleList([
                DiffusionRgbEncoder(**encoder_kwargs) for _ in range(num_cameras)
            ])
        else:
            self.rgb_encoder = DiffusionRgbEncoder(**encoder_kwargs)

        # Compute global_cond_dim
        single_encoder = self.rgb_encoder[0] if self.use_separate_encoder else self.rgb_encoder
        img_feature_dim = single_encoder.feature_dim * num_cameras
        state_dim = config.task.state_dim
        self.global_cond_dim = state_dim + img_feature_dim

        # Noise scheduler
        scheduler_cfg = config.policy.noise_scheduler
        scheduler_cls = NOISE_SCHEDULER_REGISTRY[scheduler_cfg.type]
        self.noise_scheduler = scheduler_cls(
            num_train_timesteps=scheduler_cfg.num_train_timesteps,
            beta_start=scheduler_cfg.get("beta_start", 0.0001),
            beta_end=scheduler_cfg.get("beta_end", 0.02),
            beta_schedule=scheduler_cfg.beta_schedule,
            prediction_type=scheduler_cfg.prediction_type,
            clip_sample=scheduler_cfg.get("clip_sample", True),
            clip_sample_range=scheduler_cfg.get("clip_sample_range", 1.0),
        )
        self.num_inference_steps = scheduler_cfg.num_inference_steps

        # 1D UNet
        unet_cfg = config.policy.unet
        self.unet = ConditionalUnet1d(
            input_dim=config.task.action_dim,
            global_cond_dim=self.global_cond_dim * config.policy.obs_horizon,
            down_dims=list(unet_cfg.down_dims),
            kernel_size=unet_cfg.kernel_size,
            n_groups=unet_cfg.n_groups,
            diffusion_step_embed_dim=unet_cfg.diffusion_step_embed_dim,
            use_film_scale_modulation=unet_cfg.use_film_scale_modulation,
        )

        logger.info(f"UNet parameters: {sum(p.numel() for p in self.unet.parameters()) / 1e6:.2f}M")
        logger.info(f"Global cond dim: {self.global_cond_dim} (state={state_dim} + img={img_feature_dim})")
        logger.info(f"Scheduler: {scheduler_cfg.type}, train_steps={scheduler_cfg.num_train_timesteps}, "
                     f"inference_steps={self.num_inference_steps}")

        self.reset()

    def _encode_observations(self, batch):
        """Encode images + state into global conditioning vector.

        Returns:
            (B, global_cond_dim) conditioning vector.
        """
        # State: already normalized by normalize_inputs
        state = batch[self.config.task.state_key]  # (B, obs_horizon, state_dim)
        B, S = state.shape[:2]

        # Images: stack all cameras → (B, S, N, C, H, W)
        images = torch.stack([batch[key] for key in self.config.task.image_keys], dim=2)
        N = images.shape[2]

        if self.use_separate_encoder:
            # (B, S, N, C, H, W) → (N, B*S, C, H, W)
            images_per_cam = einops.rearrange(images, "b s n c h w -> n (b s) c h w")
            img_features = torch.cat([
                encoder(cam_imgs)
                for encoder, cam_imgs in zip(self.rgb_encoder, images_per_cam)
            ])
            # (N*B*S, feat) → (B, S, N*feat)
            img_features = einops.rearrange(
                img_features, "(n b s) d -> b s (n d)", b=B, s=S, n=N
            )
        else:
            # (B, S, N, C, H, W) → (B*S*N, C, H, W)
            flat_images = einops.rearrange(images, "b s n c h w -> (b s n) c h w")
            img_features = self.rgb_encoder(flat_images)
            # (B*S*N, feat) → (B, S, N*feat)
            img_features = einops.rearrange(
                img_features, "(b s n) d -> b s (n d)", b=B, s=S, n=N
            )

        # Concatenate state + image features, then flatten
        global_cond = torch.cat([state, img_features], dim=-1)
        global_cond = global_cond.flatten(start_dim=1)

        return global_cond

    def compute_loss(self, batch):
        global_cond = self._encode_observations(batch)
        actions = batch[self.config.task.action_key]  # (B, pred_horizon, action_dim)

        noise = torch.randn_like(actions)
        B = actions.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,),
            device=actions.device, dtype=torch.long,
        )

        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        pred = self.unet(noisy_actions, timesteps, global_cond)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = actions
        else:
            raise ValueError(f"Unknown prediction_type: {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(pred, target)
        return loss, {"diffusion_loss": loss.item()}

    def generate_actions(self, batch, global_cond=None):
        """Generate a full action chunk via diffusion denoising.

        Args:
            batch: Normalized observation dict (output of normalize_inputs).
            global_cond: Optional pre-encoded conditioning vector (B, global_cond_dim).
                         If None, _encode_observations(batch) is called internally.
                         Pass a pre-computed value to avoid double-encoding (ADR-004).

        Returns:
            Tensor(B, pred_horizon, action_dim) — normalized predicted actions.
        """
        if global_cond is None:
            global_cond = self._encode_observations(batch)
        B = global_cond.shape[0]
        sample = torch.randn(
            (B, self.pred_horizon, self.action_dim),
            device=global_cond.device,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            t_batch = t.unsqueeze(0).expand(B).to(global_cond.device)
            pred = self.unet(sample, t_batch, global_cond)
            sample = self.noise_scheduler.step(pred, t, sample).prev_sample
        return sample

    @torch.no_grad()
    def select_action(self, batch):
        self.eval()
        batch = {
            k: v.unsqueeze(1)
            for k, v in batch.items()
            if k in self.config.task.image_keys + [self.config.task.state_key]
        }
        batch = self.normalize_inputs(batch)

        if len(self._action_queue) == 0:
            actions = self.generate_actions(batch)
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            actions = actions[:, start:end]
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()


    def reset(self):
        super().reset()

    def get_optimizer(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config.optimizer_lr,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
            weight_decay=self.config.optimizer_weight_decay,
        )

    def get_scheduler(self, optimizer, num_training_steps):
        return get_scheduler(
            name=self.config.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.config.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def visualize(self, batch, num_samples: int = 1) -> dict:
        """Predicted-vs-GT per-dimension action trajectories + a sample encoder
        input image (post resize/crop), logged to wandb by the trainer."""
        self.eval()
        batch = {k: v[:num_samples] for k, v in batch.items()}
        batch = self.normalize_inputs(batch)

        with torch.no_grad():
            pred_norm = self.generate_actions(batch)
        pred = self.unnormalize_outputs({"action": pred_norm})["action"]
        gt = batch[self.config.task.action_key]

        pred_np = pred.cpu().numpy()
        gt_np = gt.cpu().numpy()

        viz = {}
        for i in range(num_samples):
            viz[f"action_traj_{i}"] = plot_joint_trajectories(pred_np[i], gt_np[i])

        # Sample encoder input image (post resize/crop) for the first camera/sample,
        # most recent observation frame — a quick sanity check on preprocessing.
        image_key = self.config.task.image_keys[0]
        img = batch[image_key][0, -1].unsqueeze(0)  # (1, C, H, W)
        encoder = self.rgb_encoder[0] if self.use_separate_encoder else self.rgb_encoder
        with torch.no_grad():
            x = encoder.resize(img) if encoder.resize is not None else img
            x = encoder.center_crop(x) if encoder.center_crop is not None else x
        viz[f"input_image_{image_key}"] = plot_input_image(x.squeeze(0).cpu().numpy())

        return viz

    def encode_observations(self, batch):
        """Public API for encoding observations into the global conditioning vector.

        Returns:
            (B, global_cond_dim) tensor — same as _encode_observations output.
        """
        return self._encode_observations(batch)






