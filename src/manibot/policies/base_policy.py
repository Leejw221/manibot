import time
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from diffusers.training_utils import EMAModel
from huggingface_hub import PyTorchModelHubMixin

from manibot.utils.normalize import Normalize, Unnormalize

logger = logging.getLogger(__name__)


class BasePolicy(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config, stats):
        super().__init__()

        self.config = config
        self.stats = stats

        self.pred_horizon = config.policy.pred_horizon
        self.action_horizon = config.policy.action_horizon
        self.obs_horizon = config.policy.obs_horizon
        self.action_dim = config.task.action_dim

        self.normalize_inputs = Normalize(config.task.image_keys+[config.task.state_key], stats)
        self.normalize_targets = Normalize([config.task.action_key], stats)
        self.unnormalize_outputs = Unnormalize([config.task.action_key], stats)
        self._action_queue = None

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def generate_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        loss, metrics = self.compute_loss(batch)
        return loss, metrics

    def get_ema(self):
        return EMAModel(
            parameters=self.parameters(),
            power=self.config.ema_power,
        ) if self.config.use_ema else None

    def get_action_indices(self):
        """행동 창 — 관측 창과 **같은 앵커**에서 pred_horizon 개.

        LeRobot·Diffusion Policy 원문 규약이다. 아래 obs 인덱스와 짝으로 봐야 한다:
        관측 [-(h-1) .. 0] · 행동 [-(h-1) .. -(h-1)+Tp-1] 이 원문이고, 여기서는 두 창을
        같은 오프셋만큼 옮겨 관측 [0..h-1] · 행동 [0..Tp-1] 로 쓴다 — **상대 관계가 같다.**
        추론에서 chunk[h-1] 이 "지금"의 행동이고, 그래서 anchor_offset = h-1 이다
        (LeRobot 은 generate_actions 안에서 `actions[:, n_obs_steps-1:]` 로 같은 일을 한다,
        modeling_diffusion.py:328-331).
        """
        return list(range(self.pred_horizon))

    def get_observation_indices(self):
        """관측 창 — 앵커에서 h 개.

        ⚠ 2026-09-09 에 이걸 [-(h-1)..0] 으로 바꿔 "행동이 지금부터 시작"하게 만들려다
        되돌렸다. LeRobot 기본값(configuration_diffusion.py:251-256)이
            observation_delta_indices = range(1-h, 1)
            action_delta_indices      = range(1-h, 1-h+Tp)
        라 **행동 창이 관측 창의 시작에 앵커돼 있다** — 우리 규약과 같다. 원문이 그쪽이다.
        """
        return list(range(self.obs_horizon))

    def reset(self):
        self._action_queue = deque([], maxlen=self.action_horizon)

    def get_ood_detector(self):
        """Get OOD detector for this policy.
        
        Returns:
            OOD detector instance or None if not available
        """
        return None
    def validate(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        self.eval()
        batch = self.normalize_inputs(batch)
        batch_size = batch[self.config.task.action_key].shape[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            pred_norm = self.generate_actions(batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time_ms = (time.perf_counter() - t0) * 1000 / batch_size # milliseconds

        pred = self.unnormalize_outputs({"action": pred_norm})["action"]
        gt = batch[self.config.task.action_key]
        action_mse = F.mse_loss(pred, gt).item()

        # Normalize targets so the flow loss is computed on the same scale as
        # training (forward() applies normalize_targets before compute_loss).
        # action_mse above intentionally stays in raw action space.
        batch = self.normalize_targets(batch)
        loss, metrics = self.compute_loss(batch)

        metrics["action_mse"] = action_mse
        metrics["loss"] = loss.item()

        total_actions = batch_size * self.action_horizon
        metrics["time_per_chunk_ms"] = gen_time_ms
        metrics["time_per_action_ms"] = gen_time_ms / total_actions
        metrics["chunks_per_sec"] = 1000 / gen_time_ms

        return metrics
