"""ACT (Action Chunking Transformer) policy.

Zhao et al., 2023 — "Learning Fine-Grained Bimanual Manipulation with Low-Cost
Hardware". Internals (transformer layers with per-layer positional embedding,
shared ResNet18 backbone with spatial feature-map tokens, learnable decoder
query embeddings, FrozenBatchNorm2d, Xavier-uniform reset) are aligned with
lerobot's ``modeling_act.py``. The outer wiring (``ActPolicy(BasePolicy)``,
Hydra config, flare stats, optimizer with separate backbone lr, external
temporal-ensemble merger) follows our flare framework.

Key differences from a naive ``nn.TransformerEncoder`` build:
    * positional embedding is added to **Q and K only** at **every** layer
      (DETR/ACT style), not once before the encoder;
    * camera features keep their spatial layout — each ResNet18 ``layer4`` cell
      becomes a token with a 2D sinusoidal pos embed, rather than being pooled
      / SpatialSoftmaxed into a fixed-size vector;
    * the decoder uses ``chunk_size`` learnable query embeddings (DETR object
      queries), and the first-layer decoder input is a zero tensor.

LeRobot's in-policy ``ACTTemporalEnsembler`` is **not** ported. Temporal
ensembling lives in our external merger (see ``flare.inference.merger``)
so it composes uniformly with diffusion / cfm / sfp.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from diffusers.optimization import get_scheduler
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from manibot.policies.factory import registry
from manibot.policies import BasePolicy

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers (1:1 ports from lerobot modeling_act.py)                            #
# --------------------------------------------------------------------------- #

def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> torch.Tensor:
    """1D sinusoidal pos embedding (Attention is All You Need)."""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(p) for p in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


def get_activation_fn(activation: str) -> Callable:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation must be relu/gelu/glu, got {activation!r}")


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal pos embedding for ResNet feature maps (lerobot port)."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W). Output: (1, C_total, H, W) where C_total = 2 * dimension.
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi
        inverse_freq = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )
        x_range = x_range.unsqueeze(-1) / inverse_freq
        y_range = y_range.unsqueeze(-1) / inverse_freq
        pos_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (1, C_total, H, W)
        return pos


# --------------------------------------------------------------------------- #
# Encoder / Decoder layers (lerobot port — per-layer pos embed on Q/K)        #
# --------------------------------------------------------------------------- #

class ACTEncoderLayer(nn.Module):
    """Single transformer encoder layer that re-adds pos embed to Q,K every
    forward pass. Direct ``nn.MultiheadAttention`` call so we can keep V raw.
    """

    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        pre_norm: bool,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)

        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        pre_norm: bool,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ACTEncoderLayer(dim_model, n_heads, dim_feedforward, dropout, activation, pre_norm)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim_model) if pre_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        return self.norm(x)


class ACTDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        pre_norm: bool,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)

        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm

    @staticmethod
    def _add(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        decoder_pos_embed: torch.Tensor | None = None,
        encoder_pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self._add(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self._add(x, decoder_pos_embed),
            key=self._add(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        pre_norm: bool,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ACTDecoderLayer(dim_model, n_heads, dim_feedforward, dropout, activation, pre_norm)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        decoder_pos_embed: torch.Tensor | None = None,
        encoder_pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
            )
        return self.norm(x)


# --------------------------------------------------------------------------- #
# Image encoder — shared ResNet18 + 1x1 projection (lerobot's backbone)       #
# --------------------------------------------------------------------------- #

class ACTImageEncoder(nn.Module):
    """Crop → ImageNet normalize → ResNet18 layer4 → 1x1 conv to dim_model.

    Returns ``(B, dim_model, h, w)`` spatial feature map. One shared instance
    handles every camera (lerobot's convention).
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        crop_shape: tuple[int, int] | None,
        crop_is_random: bool,
        dim_model: int,
        vision_backbone: str = "resnet18",
        pretrained_backbone_weights: str | None = "IMAGENET1K_V1",
        replace_final_stride_with_dilation: bool = False,
    ) -> None:
        super().__init__()

        if crop_shape is not None:
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            self.random_crop = (
                torchvision.transforms.RandomCrop(crop_shape) if crop_is_random else self.center_crop
            )
        else:
            self.center_crop = None
            self.random_crop = None

        self.register_buffer("img_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

        # Resolve torchvision weights enum from the string in the config
        weights = None
        if pretrained_backbone_weights is not None:
            weights_enum = getattr(torchvision.models, f"ResNet18_Weights", None)
            if weights_enum is None:
                raise ValueError(f"Unknown weights enum for backbone {vision_backbone!r}")
            weights = getattr(weights_enum, pretrained_backbone_weights)

        backbone_model = getattr(torchvision.models, vision_backbone)(
            replace_stride_with_dilation=[False, False, replace_final_stride_with_dilation],
            weights=weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
        self.input_proj = nn.Conv2d(backbone_model.fc.in_features, dim_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) with pixel values in [0, 1].
        if self.center_crop is not None:
            x = self.random_crop(x) if self.training else self.center_crop(x)
        x = (x - self.img_mean) / self.img_std
        feat = self.backbone(x)["feature_map"]
        feat = self.input_proj(feat)
        return feat


# --------------------------------------------------------------------------- #
# Full ACT model                                                              #
# --------------------------------------------------------------------------- #

class ACTModel(nn.Module):
    """CVAE-conditioned transformer for chunked action prediction.

    Token layout (transformer encoder input, sequence-first):
        [latent, state, *flattened_camera_feature_map_for_cam_1, *... cam_2, ...]

    The first ``n_1d_tokens`` (= latent + state = 2) get learnable 1D pos
    embeddings; each camera's H*W spatial tokens get the shared 2D sinusoidal
    pos embedding.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        num_cameras: int,
        dim_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        pre_norm: bool,
        n_encoder_layers: int,
        n_decoder_layers: int,
        use_vae: bool,
        latent_dim: int,
        n_vae_encoder_layers: int,
        crop_shape: tuple[int, int] | None,
        crop_is_random: bool,
        vision_backbone: str,
        pretrained_backbone_weights: str | None,
        replace_final_stride_with_dilation: bool,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.dim_model = dim_model
        self.latent_dim = latent_dim
        self.use_vae = use_vae

        # --- VAE encoder ---
        if use_vae:
            self.vae_encoder = ACTEncoder(
                num_layers=n_vae_encoder_layers,
                dim_model=dim_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm,
            )
            self.vae_encoder_cls_embed = nn.Embedding(1, dim_model)
            self.vae_encoder_robot_state_input_proj = nn.Linear(state_dim, dim_model)
            self.vae_encoder_action_input_proj = nn.Linear(action_dim, dim_model)
            self.vae_encoder_latent_output_proj = nn.Linear(dim_model, latent_dim * 2)

            num_input_token_encoder = 1 + 1 + chunk_size  # cls + state + actions
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, dim_model).unsqueeze(0),
            )

        # --- Image backbone (shared across cameras) ---
        self.image_encoder = ACTImageEncoder(
            crop_shape=crop_shape,
            crop_is_random=crop_is_random,
            dim_model=dim_model,
            vision_backbone=vision_backbone,
            pretrained_backbone_weights=pretrained_backbone_weights,
            replace_final_stride_with_dilation=replace_final_stride_with_dilation,
        )

        # --- Transformer encoder ---
        self.encoder = ACTEncoder(
            num_layers=n_encoder_layers,
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            pre_norm=pre_norm,
        )

        # Per-modality input projections for transformer encoder
        self.encoder_latent_input_proj = nn.Linear(latent_dim, dim_model)
        self.encoder_robot_state_input_proj = nn.Linear(state_dim, dim_model)

        # 1D learnable pos embed for latent + state
        self.n_1d_tokens = 2  # latent, state
        self.encoder_1d_feature_pos_embed = nn.Embedding(self.n_1d_tokens, dim_model)
        # 2D sinusoidal pos embed for camera spatial features
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(dim_model // 2)

        # --- Transformer decoder ---
        self.decoder = ACTDecoder(
            num_layers=n_decoder_layers,
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            pre_norm=pre_norm,
        )
        self.decoder_pos_embed = nn.Embedding(chunk_size, dim_model)

        # Action head
        self.action_head = nn.Linear(dim_model, action_dim)

        self.num_cameras = num_cameras
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in (*self.encoder.parameters(), *self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        state: torch.Tensor,
        images_per_cam: list[torch.Tensor],
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, torch.Tensor | None]]:
        """
        Args:
            state: ``(B, state_dim)``.
            images_per_cam: list of ``(B, C, H, W)`` tensors, one per camera.
            actions: ``(B, chunk_size, action_dim)`` (required only for VAE
                encoder during training, ignored otherwise).
        Returns:
            ``pred_actions`` of shape ``(B, chunk_size, action_dim)`` and
            ``(mu, log_sigma_x2)`` each of shape ``(B, latent_dim)`` or
            ``(None, None)`` if VAE is bypassed.
        """
        B = state.shape[0]
        device = state.device

        # ---- VAE encoder ----
        if self.use_vae and actions is not None and self.training:
            cls = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=B)
            state_tok = self.vae_encoder_robot_state_input_proj(state).unsqueeze(1)  # (B, 1, D)
            action_tok = self.vae_encoder_action_input_proj(actions)  # (B, chunk, D)
            vae_in = torch.cat([cls, state_tok, action_tok], dim=1)  # (B, 1+1+chunk, D)
            pos = self.vae_encoder_pos_enc.clone().detach()  # (1, 1+1+chunk, D)

            cls_out = self.vae_encoder(
                vae_in.permute(1, 0, 2),  # (S, B, D)
                pos_embed=pos.permute(1, 0, 2),  # (S, 1, D) broadcasts to B
            )[0]  # (B, D)
            params = self.vae_encoder_latent_output_proj(cls_out)
            mu = params[:, : self.latent_dim]
            log_sigma_x2 = params[:, self.latent_dim :]
            latent = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent = torch.zeros(B, self.latent_dim, device=device, dtype=state.dtype)

        # ---- Build transformer encoder input ----
        # 1D tokens: latent, state
        encoder_tokens: list[torch.Tensor] = [
            self.encoder_latent_input_proj(latent),  # (B, D)
            self.encoder_robot_state_input_proj(state),  # (B, D)
        ]
        encoder_pos: list[torch.Tensor] = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Each item in encoder_pos is shape (1, D) — broadcasts on batch dim when added.

        # 2D tokens: spatial features from each camera (shared backbone)
        for img in images_per_cam:
            feat = self.image_encoder(img)  # (B, D, h, w)
            cam_pos = self.encoder_cam_feat_pos_embed(feat).to(dtype=feat.dtype)  # (1, D, h, w)
            feat = einops.rearrange(feat, "b c h w -> (h w) b c")
            cam_pos = einops.rearrange(cam_pos, "b c h w -> (h w) b c")
            encoder_tokens.extend(list(feat))
            encoder_pos.extend(list(cam_pos))

        encoder_tokens = torch.stack(encoder_tokens, dim=0)  # (S, B, D)
        encoder_pos = torch.stack(encoder_pos, dim=0)  # (S, 1, D)

        # ---- Transformer encoder + decoder ----
        memory = self.encoder(encoder_tokens, pos_embed=encoder_pos)

        decoder_in = torch.zeros(
            (self.chunk_size, B, self.dim_model), dtype=encoder_pos.dtype, device=device
        )
        decoder_out = self.decoder(
            decoder_in,
            memory,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),  # (chunk, 1, D)
            encoder_pos_embed=encoder_pos,
        )
        decoder_out = decoder_out.transpose(0, 1)  # (B, chunk, D)
        pred_actions = self.action_head(decoder_out)
        return pred_actions, (mu, log_sigma_x2)


# --------------------------------------------------------------------------- #
# Flare-side wrapper                                                          #
# --------------------------------------------------------------------------- #

@registry.register_policy("act")
class ActPolicy(BasePolicy):
    """ACT policy for the FLARE training framework."""

    def __init__(self, config, stats):
        super().__init__(config, stats)
        act_cfg = config.policy

        # Sanity check — lerobot's ACT only supports a single obs step.
        if self.obs_horizon != 1:
            raise ValueError(
                f"ACT only supports obs_horizon=1 (got {self.obs_horizon}). "
                "If you need multi-step observations consider Diffusion / SFP."
            )

        num_cameras = len(config.task.image_keys)

        # Image preprocessing config (these live at config root, not under policy)
        crop_shape = tuple(config.crop_shape) if config.get("crop_shape", None) else None

        self.model = ACTModel(
            state_dim=config.task.state_dim,
            action_dim=config.task.action_dim,
            chunk_size=self.pred_horizon,
            num_cameras=num_cameras,
            dim_model=act_cfg.get("dim_model", 512),
            n_heads=act_cfg.get("n_heads", 8),
            dim_feedforward=act_cfg.get("dim_feedforward", 3200),
            dropout=act_cfg.get("dropout", 0.1),
            activation=act_cfg.get("feedforward_activation", "relu"),
            pre_norm=act_cfg.get("pre_norm", False),
            n_encoder_layers=act_cfg.get("n_encoder_layers", 4),
            n_decoder_layers=act_cfg.get("n_decoder_layers", 1),
            use_vae=act_cfg.get("use_vae", True),
            latent_dim=act_cfg.get("latent_dim", 32),
            n_vae_encoder_layers=act_cfg.get("n_vae_encoder_layers", 4),
            crop_shape=crop_shape,
            crop_is_random=act_cfg.get("crop_is_random", True),
            vision_backbone=act_cfg.get("vision_backbone", "resnet18"),
            pretrained_backbone_weights=act_cfg.get("pretrained_backbone_weights", "IMAGENET1K_V1"),
            replace_final_stride_with_dilation=act_cfg.get("replace_final_stride_with_dilation", False),
        )

        self.use_vae = self.model.use_vae
        self.kl_weight = act_cfg.get("kl_weight", 10.0)

        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        backbone_params = sum(p.numel() for p in self.model.image_encoder.backbone.parameters()) / 1e6
        logger.info(
            f"ACT model: {total_params:.2f}M total (backbone {backbone_params:.2f}M), "
            f"dim_model={self.model.dim_model}, chunk_size={self.model.chunk_size}, "
            f"use_vae={self.use_vae}, kl_weight={self.kl_weight}"
        )

        self.reset()

    # ---------- Input plumbing ----------
    def _extract_state_and_images(self, batch):
        """Pull (state, images_per_cam) out of a normalized batch.

        Both training and inference batches carry obs with a leading
        obs_horizon dim (= 1 for ACT). We squeeze it here so the model sees
        plain (B, ...) tensors.
        """
        state = batch[self.config.task.state_key]
        if state.ndim == 3:  # (B, obs_horizon, state_dim)
            state = state.squeeze(1)
        elif state.ndim != 2:
            raise ValueError(f"Unexpected state shape: {tuple(state.shape)}")

        images_per_cam: list[torch.Tensor] = []
        for key in self.config.task.image_keys:
            img = batch[key]
            if img.ndim == 5:  # (B, obs_horizon, C, H, W)
                img = img.squeeze(1)
            elif img.ndim != 4:
                raise ValueError(f"Unexpected image shape for {key}: {tuple(img.shape)}")
            images_per_cam.append(img)
        return state, images_per_cam

    # ---------- BasePolicy interface ----------
    def compute_loss(self, batch):
        state, images_per_cam = self._extract_state_and_images(batch)
        actions = batch[self.config.task.action_key]  # (B, chunk_size, action_dim)

        pred_actions, (mu, log_sigma_x2) = self.model(state, images_per_cam, actions=actions)

        l1_loss = F.l1_loss(pred_actions, actions)
        metrics = {"l1_loss": l1_loss.item()}

        if self.use_vae and mu is not None and log_sigma_x2 is not None:
            # KL divergence per batch element, then mean over batch.
            # log_sigma_x2 follows lerobot's convention of being 2*log(sigma).
            kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(dim=-1).mean()
            loss = l1_loss + self.kl_weight * kld
            metrics["kld_loss"] = kld.item()
        else:
            loss = l1_loss

        metrics["total_loss"] = loss.item()
        return loss, metrics

    @torch.no_grad()
    def generate_actions(self, batch):
        state, images_per_cam = self._extract_state_and_images(batch)
        pred_actions, _ = self.model(state, images_per_cam)
        return pred_actions

    # ---------- Optimizer / scheduler ----------
    def get_optimizer(self):
        # Separate learning rates for image backbone (frozen-BN ResNet18) and
        # everything else — original ACT used 1e-5 for the backbone vs 1e-4
        # for the rest. We follow the same convention but make it configurable.
        backbone_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "image_encoder.backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        lr = self.config.optimizer_lr
        backbone_lr = self.config.policy.get("backbone_lr", lr)
        param_groups = [
            {"params": other_params, "lr": lr},
            {"params": backbone_params, "lr": backbone_lr},
        ]
        return torch.optim.AdamW(
            params=param_groups,
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
