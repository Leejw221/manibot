import os
import re
import glob
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def get_latest_checkpoint(checkpoint_dir):
    """Get latest checkpoint directory (step_* format)."""
    checkpoint_dir = Path(checkpoint_dir)

    # Look for step_* directories
    step_dirs = list(checkpoint_dir.glob("step_*"))
    if not step_dirs:
        logger.warning(f"No step_* directories found in {checkpoint_dir}")
        return None

    # Extract step numbers and find maximum
    step_numbers = []
    for step_dir in step_dirs:
        match = re.search(r'step_(\d+)', step_dir.name)
        if match:
            step_numbers.append((int(match.group(1)), step_dir))

    if not step_numbers:
        logger.warning(f"No valid step directories found in {checkpoint_dir}")
        return None

    # Return directory with highest step number
    latest_step, latest_dir = max(step_numbers, key=lambda x: x[0])
    logger.info(f"Found latest checkpoint: {latest_dir} (step {latest_step})")
    return latest_dir


def get_best_checkpoint(checkpoint_dir, best_name="best"):
    """Get best checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)

    # Look for best_* directories
    if best_name == "best":
        # Find any best_* directory
        best_dirs = list(checkpoint_dir.glob("best_*"))
        if not best_dirs:
            return None
        for priority in ["best_pc_success", "avg_max_reward"]:
            for best_dir in best_dirs:
                if priority in best_dir.name:
                    return best_dir
        return best_dirs[0]
    else:
        best_dir = checkpoint_dir / best_name
        return best_dir if best_dir.exists() else None


def parse_checkpoint_patterns(checkpoint_dir, ckpt_pattern):
    """Parse checkpoint pattern into list of checkpoint paths."""
    if isinstance(ckpt_pattern, int):
        step_dir = Path(checkpoint_dir) / f"step_{ckpt_pattern:010d}"
        return [step_dir] if step_dir.exists() else []

    if ckpt_pattern == "latest":
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        return [latest_checkpoint] if latest_checkpoint else []

    if ckpt_pattern.startswith("best"):
        best_ckpt = get_best_checkpoint(checkpoint_dir, best_name=ckpt_pattern)
        return [best_ckpt] if best_ckpt else []

    if ":" in ckpt_pattern:
        parts = ckpt_pattern.split(":")
        if len(parts) not in [2, 3]:
            raise ValueError(f"Invalid slicing syntax: {ckpt_pattern}")

        start = int(parts[0])
        end = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1

        checkpoint_dir = Path(checkpoint_dir)
        ckpt_paths = []
        for i in range(start, end + 1, step):
            step_dir = checkpoint_dir / f"step_{i:010d}"
            if step_dir.exists():
                ckpt_paths.append(step_dir)
        return ckpt_paths

    # Try to parse as step number
    try:
        step_num = int(ckpt_pattern)
        step_dir = Path(checkpoint_dir) / f"step_{step_num:010d}"
        return [step_dir] if step_dir.exists() else []
    except ValueError:
        logger.warning(f"Unknown checkpoint pattern: {ckpt_pattern}")
        return []


def get_checkpoint_paths(checkpoint_dir, pattern):
    """Get list of valid checkpoint paths matching pattern."""
    ckpt_list = parse_checkpoint_patterns(checkpoint_dir, pattern)
    if not ckpt_list:
        logger.warning(f"No checkpoints found for pattern: {pattern}")
        return []

    valid_paths = []
    for ckpt_path in ckpt_list:
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue
        # Verify it's a valid checkpoint directory
        if not (ckpt_path / "training_state.pt").exists():
            logger.warning(f"Invalid checkpoint directory (missing training_state.pt): {ckpt_path}")
            continue
        valid_paths.append(ckpt_path)

    return valid_paths


def load_model_weights(model, checkpoint_dir, device):
    """Load model weights from a checkpoint directory into an existing model."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    safetensors_path = checkpoint_dir / "model.safetensors"
    pytorch_path = checkpoint_dir / "pytorch_model.bin"

    state_dict = None
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise ImportError(
                "safetensors is required to load model.safetensors"
            ) from e
        state_dict = load_file(str(safetensors_path))
    elif pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location=device)
    else:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_dir} (expected model.safetensors or pytorch_model.bin)"
        )

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        logger.warning(
            "Loaded checkpoint with missing/unexpected keys. "
            f"Missing: {missing_keys} | Unexpected: {unexpected_keys}"
        )

    return model


def load_ema_weights(model, checkpoint_dir, device):
    """체크포인트의 **EMA 가중치**를 모델에 덮어쓴다. 없으면 False.

    `save_checkpoint` 은 raw 가중치를 `model.safetensors` 에 쓰고 EMA 는
    `training_state.pt` 안에 따로 둔다. 그래서 `load_model_weights` 만 부르면 학습 중
    가중치가 로드되는데, **학습 중 검증(validate_offline/online)은 EMA 를 쓴다** —
    같은 체크포인트를 두고 학습 로그의 수치와 `scripts/eval.py` 의 수치가 갈린다.
    diffusion policy 는 EMA 차이가 성능으로 나타나므로 평가는 EMA 쪽에 맞춘다.

    diffusers `EMAModel` 은 `EMAModel(parameters=policy.parameters())` 로 만들어져
    `shadow_params` 의 순서가 `model.parameters()` 순서와 같다 — 그 순서로 복사한다
    (safetensors 의 키 순서와는 다르다. 거기엔 파라미터가 아닌 버퍼도 섞여 있다).
    """
    state_file = Path(checkpoint_dir) / "training_state.pt"
    if not state_file.exists():
        return False
    ema = torch.load(state_file, map_location=device, weights_only=False).get("ema")
    if not ema or "shadow_params" not in ema:
        return False
    shadow = ema["shadow_params"]
    params = list(model.parameters())
    if len(shadow) != len(params):
        logger.warning(f"EMA 파라미터 수 불일치 ({len(shadow)} vs {len(params)}) — 건너뛴다")
        return False
    with torch.no_grad():
        for s, p in zip(shadow, params):
            p.copy_(s.to(p.device, dtype=p.dtype))
    return True


def build_ref_policy(cfg, dataset_meta, stats, checkpoint, device):
    """선호 최적화용 **동결 참조 정책**을 만든다.

    ⚠ 이 정책을 학습 network 의 서브모듈로 넣으면 안 된다 — optimizer·EMA 가 따라 잡고,
    체크포인트 키가 한 겹 깊어져 eval 이 `strict=False` 로 **조용히 아무것도 안 싣는다**.
    Trainer 의 속성으로만 들고 다닌다.

    ⚠ `train(True)` 로 두는 이유: 정책 안의 랜덤 크롭이 `self.training` 으로 갈린다.
    eval 모드면 pi_ref 만 센터 크롭이 되어 두 정책이 다른 이미지를 본다 — 그 잡음이
    정책 신호의 0.81배였다 [측정 2026-09-12]. grad 는 requires_grad_(False) 로 막는다.
    """
    from manibot.policies.factory import make_policy

    ref, _, _ = make_policy(cfg, dataset_meta, stats)
    ref = ref.to(device)
    load_model_weights(ref, checkpoint, device)
    used = load_ema_weights(ref, checkpoint, device)
    ref.train(True)
    ref.requires_grad_(False)
    return ref, used
