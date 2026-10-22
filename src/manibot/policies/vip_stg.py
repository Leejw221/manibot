"""공유 백본 Diffusion Policy + STG(steps-to-go) 헤드 — 인코더 하나를 행동(BC)과 STG 손실이 함께 학습한다.

희민 self-improving-gym 의 DiffusionPolicyImageStg(projects/square_assembly/.../diffusion_policy_stg.py,
2026-10-21 확인)를 manibot 의 LeRobot DiffusionPolicy 위로 옮겼다. 원본과 같게 둔 것:
- 인코더: 사전학습 VIP ResNet50(fc -> 1024)을 두 카메라가 공유. conv1·bn1·layer1·layer2 를 얼리고 BatchNorm 은
  늘 eval. crop(학습 랜덤·평가 중앙) -> bilinear 224 -> ImageNet 정규화. 인코더는 bf16.
- 특징 표준화: 카메라별로 차원별 평균, 크기는 차원별 표준편차의 RMS 스칼라 하나(원본 _camera_stats — 차원별로
  나누면 거의 상수인 차원이 미세조정 중 수천 배로 커져 STG 손실이 폭주했다).
- stg_head: global_cond -> num_bins 선형층 하나(bias 없음). 손실 = BC + stg_weight * CE, 행은 bc_mask/stg_mask 로 가른다.
  라운드 학습(배치에 bc_rate 가 있을 때)은 BC 를 batch_shares 로 샘플 가중한다. CE 는 가중하지 않는다(계획 결정 8).
원본과 다른 것: UNet·노이즈 스케줄러·BC 손실·정규화 전처리는 기존 base 와 같은 LeRobot DiffusionPolicy 그대로다
(이미지는 전처리기가 ImageNet 으로 이미 정규화해 들어온다 — 채널별 선형 정규화와 bilinear 보간은 순서를 바꿔도 같다).
"""

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.checkpoint import checkpoint
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
    DiffusionPolicy,
    _make_noise_scheduler,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

VIP_DIM = 1024


def batch_shares(rate, floor):
    """라운드 BC 의 배치 몫 (B,) — 사용자 10-22 (가) 방식. v = max(0, rate), 몫 = v / sum(v).
    시연·개입(floor)은 몫이 1/B 아래로 내려가지 않게 올리고, 다시 나누지 않는다 — 하한이 걸린 만큼 몫의 합이
    1 을 넘는다. 1/B 는 가중 없는 BC(배치 평균)에서 샘플 하나가 받는 몫이다 = "최소한 비가중만큼은 배운다".
    robot 은 하한 없이 진척률대로(음수는 0). 결정 경위(2026-10-22)는 scripts/train_vip_stg.py docstring 의 기록."""
    v = rate.float().clamp_min(0)
    raw = v / v.sum().clamp_min(1e-8)
    floor = floor.bool()
    share = torch.where(floor, raw.clamp_min(1.0 / len(raw)), raw)
    hit = floor & (raw < 1.0 / len(raw))
    return share, {"share_sum": float(share.sum()),
                   "floor_hit": float(hit.sum() / floor.sum().clamp_min(1))}


class VipEncoder(nn.Module):
    """(B, 3, H, W) ImageNet 정규화된 이미지 -> (B, 1024) 카메라별 표준화 특징. convnet 은 카메라끼리 공유한다."""

    def __init__(self, n_cameras, crop_shape, size, frozen_stages):
        super().__init__()
        convnet = torchvision.models.resnet50(weights=None)
        convnet.fc = nn.Linear(convnet.fc.in_features, VIP_DIM)
        for name in frozen_stages:
            getattr(convnet, name).requires_grad_(False)
        self.convnet = convnet
        self.random_crop = torchvision.transforms.RandomCrop(tuple(crop_shape))
        self.center_crop = torchvision.transforms.CenterCrop(tuple(crop_shape))
        self.size = int(size)
        # True 면 학습 가능한 층(layer3·layer4)의 안쪽 활성값을 버리고 역전파 때 다시 계산한다 — 결과는 같고 메모리만
        # 준다(원본 vip_grad_checkpoint, 24GB 에서 OOM 이 나서 넣은 옵션). BN 이 늘 eval 이라 재계산이 같다.
        self.grad_checkpoint = False
        # 학습 스크립트가 시작 때 학습 데이터로 채운다(fit_feature_stats). 체크포인트에 같이 저장된다.
        self.register_buffer("feat_mean", torch.zeros(n_cameras, VIP_DIM))
        self.register_buffer("feat_std", torch.ones(n_cameras, VIP_DIM))

    def load_vip(self, path):
        """VIP 공식 체크포인트(model.pt 의 'vip', DataParallel 키 module.convnet.*)를 convnet 에 싣는다."""
        state = torch.load(path, map_location="cpu", weights_only=False)["vip"]
        prefix = "module.convnet."
        self.convnet.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})

    def raw_features(self, x):
        x = self.random_crop(x) if self.training else self.center_crop(x)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        # ResNet50 @224 가 학습 시간의 대부분이라 인코더만 bf16 (원본과 같다)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            f = self._convnet(x)
        return f.float()

    def _convnet(self, x):
        if not (self.grad_checkpoint and self.training and torch.is_grad_enabled()):
            return self.convnet(x)
        net = self.convnet
        x = net.layer2(net.layer1(net.maxpool(net.relu(net.bn1(net.conv1(x))))))
        for layer in (net.layer3, net.layer4):
            x = checkpoint(layer, x, use_reentrant=False)
        return net.fc(torch.flatten(net.avgpool(x), 1))

    def forward(self, x, camera):
        return (self.raw_features(x) - self.feat_mean[camera]) / self.feat_std[camera]

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self


class VipStgDiffusionModel(DiffusionModel):
    def __init__(self, config, num_bins, crop_shape, size, frozen_stages):
        # DiffusionModel.__init__ 은 ResNet18 인코더와 그 크기의 UNet 을 만든다 — 인코더가 바뀌면 조건 차원이
        # 달라지므로 부모 생성자를 건너뛰고 같은 부품(UNet·스케줄러)을 VIP 특징 차원으로 만든다.
        nn.Module.__init__(self)
        self.config = config
        n_cameras = len(config.image_features)
        self.rgb_encoder = VipEncoder(n_cameras, crop_shape, size, frozen_stages)
        global_cond_dim = (config.robot_state_feature.shape[0] + VIP_DIM * n_cameras) * config.n_obs_steps
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = (config.num_inference_steps if config.num_inference_steps is not None
                                    else self.noise_scheduler.config.num_train_timesteps)
        self.stg_head = nn.Linear(global_cond_dim, num_bins, bias=False)

    def _prepare_global_conditioning(self, batch):
        b, s = batch[OBS_STATE].shape[:2]
        images = batch[OBS_IMAGES]  # (b, s, n, c, h, w)
        feats = [batch[OBS_STATE]]
        for cam in range(images.shape[2]):
            x = einops.rearrange(images[:, :, cam], "b s c h w -> (b s) c h w")
            feats.append(self.rgb_encoder(x, cam).reshape(b, s, -1))
        return torch.cat(feats, dim=-1).flatten(start_dim=1)

    def bc_loss(self, global_cond, action, action_is_pad, per_sample=False):
        """DiffusionModel.compute_loss 와 같은 계산을 이미 만든 global_cond 로 한다(인코더를 STG 와 한 번만 돈다).
        per_sample=True 면 (B,) 샘플별 손실 — 가중 없이 평균하면 기본값과 같다(패딩 마스크가 없을 때)."""
        eps = torch.randn(action.shape, device=action.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (action.shape[0],),
                                  device=action.device).long()
        pred = self.unet(self.noise_scheduler.add_noise(action, eps, timesteps), timesteps, global_cond=global_cond)
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = action
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")
        loss = F.mse_loss(pred, target, reduction="none")
        if per_sample:
            if self.config.do_mask_loss_for_padding:
                mask = (~action_is_pad).unsqueeze(-1)
                return (loss * mask).sum((1, 2)) / (mask.sum((1, 2)) * loss.shape[-1]).clamp_min(1)
            return loss.mean((1, 2))
        if self.config.do_mask_loss_for_padding:
            mask = (~action_is_pad).unsqueeze(-1)
            return (loss * mask).sum() / (mask.sum() * loss.shape[-1]).clamp_min(1)
        return loss.mean()


class VipStgDiffusionPolicy(DiffusionPolicy):
    def __init__(self, config, num_bins, crop_shape, size, frozen_stages, encoder_lr_scale, optim):
        # DiffusionPolicy.__init__ 은 DiffusionModel(ResNet18) 을 만든다 — 같은 일을 VIP 모델로 한다.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.diffusion = VipStgDiffusionModel(config, num_bins, crop_shape, size, frozen_stages)
        self.encoder_lr_scale = float(encoder_lr_scale)
        self._optim = dict(optim)
        self.stg_weight = 0.0  # 학습기가 warmup 으로 올린다
        self.reset()

    def _stack_images(self, batch):
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack([batch[k] for k in self.config.image_features], dim=-4)
        return batch

    def forward(self, batch):
        if "stg_label" not in batch:
            return super().forward(batch)
        batch = self._stack_images(batch)
        global_cond = self.diffusion._prepare_global_conditioning(batch)
        bc, stg = batch["bc_mask"].bool(), batch["stg_mask"].bool()
        zero = global_cond.sum() * 0.0
        info = {}
        if not bc.any():
            bc_loss = zero
        elif "bc_rate" in batch:
            per = self.diffusion.bc_loss(global_cond[bc], batch[ACTION][bc], batch["action_is_pad"][bc],
                                         per_sample=True)
            share, info = batch_shares(batch["bc_rate"][bc], batch["bc_floor"][bc])
            bc_loss = (share.to(per.dtype) * per).sum()
        else:
            bc_loss = self.diffusion.bc_loss(global_cond[bc], batch[ACTION][bc], batch["action_is_pad"][bc])
        stg_loss = (F.cross_entropy(self._stg_logits(global_cond[stg]), batch["stg_label"][stg].long())
                    if stg.any() else zero)
        loss = bc_loss + self.stg_weight * stg_loss
        return loss, {"bc_loss": float(bc_loss.detach()), "stg_loss": float(stg_loss.detach()),
                      "stg_weight": float(self.stg_weight), **info}

    def _stg_logits(self, global_cond):
        # 원본은 헤드를 fp32 로 돌린다(학습기에 autocast 가 없다) — AMP 안에서도 헤드만 fp32 로 계산한다
        with torch.autocast("cuda", enabled=False):
            return self.diffusion.stg_head(global_cond.float())

    @torch.no_grad()
    def stg_distribution(self, batch):
        """관측 창 -> STG bin 확률 (B, num_bins). 기댓값 d = sum(p_i * i) 가 남은 스텝 추정이다."""
        return self._stg_logits(self.diffusion._prepare_global_conditioning(self._stack_images(batch))).softmax(-1)

    def get_optimizer(self):
        """인코더(학습 가능한 층)만 lr x encoder_lr_scale. 얼린 층은 옵티마이저에 넣지 않는다."""
        enc = [p for p in self.diffusion.rgb_encoder.parameters() if p.requires_grad]
        enc_ids = {id(p) for p in enc}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in enc_ids]
        o = self._optim
        return torch.optim.AdamW(
            [{"params": rest, "lr": o["lr"]}, {"params": enc, "lr": o["lr"] * self.encoder_lr_scale}],
            lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"], weight_decay=o["weight_decay"])


def make_vip_stg_policy(cfg, dataset_meta, stats):
    """(policy, preprocessor, postprocessor). DiffusionConfig·전처리기는 기존 base 와 같은 경로(_lerobot_diffusion)로 만든다."""
    from manibot.policies.factory import _lerobot_diffusion

    ref, pre, post = _lerobot_diffusion(cfg, dataset_meta, stats)
    config = ref.config
    del ref
    vs = cfg.policy.vip_stg
    policy = VipStgDiffusionPolicy(
        config, num_bins=vs.num_bins, crop_shape=cfg.crop_shape, size=vs.size,
        frozen_stages=list(vs.frozen_stages), encoder_lr_scale=vs.encoder_lr_scale,
        optim={"lr": cfg.optimizer_lr, "betas": cfg.optimizer_betas, "eps": cfg.optimizer_eps,
               "weight_decay": cfg.optimizer_weight_decay})
    policy.diffusion.rgb_encoder.grad_checkpoint = bool(vs.get("grad_checkpoint", False))
    return policy, pre, post
