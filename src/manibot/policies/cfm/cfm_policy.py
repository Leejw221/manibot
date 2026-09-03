"""Conditional Flow Matching policy scaffold.

This policy is migrated from the diffusion policy template and gradually
replaces diffusion-specific training/inference with flow matching.
"""

import torch
import logging
import einops
from diffusers.optimization import get_scheduler

from manibot.policies.factory import registry
from manibot.policies.flow.flow_matchers import get_flow_matcher
from manibot.policies import BasePolicy
from manibot.policies.observers.resnet_observer import DiffusionRgbEncoder
from manibot.model.diffusion import ConditionalUnet1d
from manibot.utils.normalize import NormalizeMinMax, UnnormalizeMinMax

logger = logging.getLogger(__name__)


@registry.register_policy("cfm")
class CfmPolicy(BasePolicy):
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
            resize_shape=tuple(config.resize_shape),
            crop_shape=tuple(config.crop_shape),
            crop_is_random=diffusion_cfg.get("crop_is_random", True),
            spatial_softmax_num_keypoints=diffusion_cfg.get("spatial_softmax_num_keypoints", 32),
        )

        if self.use_separate_encoder:
            self.rgb_encoder = torch.nn.ModuleList([
                DiffusionRgbEncoder(**encoder_kwargs) for _ in range(num_cameras)
            ])
        else:
            self.rgb_encoder = DiffusionRgbEncoder(**encoder_kwargs)

        # Compute flattened global conditioning dimension.
        single_encoder = self.rgb_encoder[0] if self.use_separate_encoder else self.rgb_encoder
        img_feature_dim = single_encoder.feature_dim * num_cameras
        state_dim = config.task.state_dim
        self.global_cond_dim_per_step = state_dim + img_feature_dim
        self.global_cond_dim = self.obs_horizon * self.global_cond_dim_per_step

        required_pred_horizon = self.obs_horizon - 1 + self.action_horizon
        if self.pred_horizon < required_pred_horizon:
            raise ValueError(
                "Invalid horizon setup: "
                f"pred_horizon={self.pred_horizon} must be >= {required_pred_horizon} "
                f"(obs_horizon={self.obs_horizon}, action_horizon={self.action_horizon})."
            )

        # Flow matcher (used for CFM training loss)
        flow_matcher_cfg = config.policy.flow_matcher
        self.FM = get_flow_matcher(**flow_matcher_cfg)

        # 1D UNet
        unet_cfg = config.policy.unet
        self.unet = ConditionalUnet1d(
            input_dim=config.task.action_dim,
            global_cond_dim=self.global_cond_dim,
            down_dims=list(unet_cfg.down_dims),
            kernel_size=unet_cfg.kernel_size,
            n_groups=unet_cfg.n_groups,
            diffusion_step_embed_dim=unet_cfg.diffusion_step_embed_dim,
            use_film_scale_modulation=unet_cfg.use_film_scale_modulation,
        )

        logger.info(f"UNet parameters: {sum(p.numel() for p in self.unet.parameters()) / 1e6:.2f}M")
        logger.info(
            f"Global cond dim: {self.global_cond_dim} "
            f"(obs_horizon={self.obs_horizon} x per_step={self.global_cond_dim_per_step})"
        )
        logger.info(f"Flow matcher: {flow_matcher_cfg.name}, sampling_steps={flow_matcher_cfg.num_sampling_steps}")

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
        global_cond = torch.cat([state, img_features], dim=-1)  # (B, S, global_cond_dim)
        global_cond = global_cond.flatten(start_dim=1)  # (B, S * global_cond_dim)
        return global_cond

    def compute_loss(self, batch):
        global_cond = self._encode_observations(batch)
        actions = batch[self.config.task.action_key]  # (B, pred_horizon, action_dim)

        loss, metrics = self.FM.compute_loss(
            self.unet,
            target=actions,
            global_cond=global_cond,
        )
        metrics = dict(metrics)
        metrics["flow_loss"] = loss.item()
        return loss, metrics

    @torch.no_grad()
    def generate_actions(self, batch):
        global_cond = self._encode_observations(batch)
        B = global_cond.shape[0]
        num_steps = self.config.policy.flow_matcher.num_sampling_steps

        actions = self.FM.sample(
            self.unet,
            shape=(B, self.pred_horizon, self.action_dim),
            device=global_cond.device,
            num_steps=num_steps,
            return_traces=False,
            global_cond=global_cond,
        )
        if isinstance(actions, tuple):
            actions = actions[0]
        return actions

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
            action_key = self.config.task.action_key
            actions = self.unnormalize_outputs({action_key: actions})[action_key]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

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
