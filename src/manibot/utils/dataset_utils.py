from typing import Iterator, Union

import torch
from pprint import pformat

from omegaconf import DictConfig

import logging

logger = logging.getLogger(__name__)



class EpisodeAwareSampler:
    def __init__(
        self,
        episode_data_index: dict,
        episode_indices_to_use: Union[list, None] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            episode_data_index: Dictionary with keys 'from' and 'to' containing the start and end indices of each episode.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(episode_data_index["from"], episode_data_index["to"], strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(
                    range(start_index.item() + drop_n_first_frames, end_index.item() - drop_n_last_frames)
                )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


def cycle(iterable):
    """The equivalent of itertools.cycle, but safe for Pytorch dataloaders.

    See https://github.com/pytorch/pytorch/issues/23900 for information on why itertools.cycle is not safe.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def create_dataset_stats(cfg: DictConfig):
    """Create dataset metadata and statistics."""
    from manibot.datasets.zarr_dataset import ZarrDatasetMeta

    dataset_meta = ZarrDatasetMeta(
        repo_id=cfg.task.dataset_repo_id,
        root=cfg.task.dataset_root
    )
    stats = dataset_meta.stats
    stats.update(cfg.task.override_stats)
    return dataset_meta, stats


def log_dataset_image_resolution(cfg: DictConfig, dataset_meta) -> None:
    """Print the resolution actually stored in the converted dataset (convert.py's
    --resize) next to what this run's resize_shape/crop_shape expect, so a mismatch
    between the two (independently-maintained) values shows up at the start of
    training instead of as a silent blur or a crop-size error mid-run."""
    if cfg.get("resize_shape") is None:
        return  # 데이터 해상도를 그대로 쓴다 — 맞춰볼 대상이 없다
    resize_shape = tuple(cfg.resize_shape)
    crop_shape = tuple(cfg.crop_shape) if cfg.get("crop_shape") else None

    for key in cfg.task.image_keys:
        stored_shape = dataset_meta.features.get(key, {}).get("shape")
        if stored_shape is None:
            continue
        stored_hw = tuple(stored_shape[:2])
        logger.info(f"Dataset image '{key}': stored={stored_hw}, resize_shape={resize_shape}, crop_shape={crop_shape}")

        if stored_hw != resize_shape:
            logger.warning(
                f"'{key}': stored resolution {stored_hw} != resize_shape {resize_shape} "
                "— the encoder will resize again (upscaling if stored is smaller), "
                "so image quality/crop augmentation range may not match what resize_shape implies. "
                "Re-run convert.py with --resize matching resize_shape, or update resize_shape to match."
            )
        if crop_shape is not None and (crop_shape[0] > stored_hw[0] or crop_shape[1] > stored_hw[1]):
            logger.warning(
                f"'{key}': crop_shape {crop_shape} is larger than stored resolution {stored_hw} "
                "— RandomCrop/CenterCrop will error once training reaches this batch."
            )


def create_dataset(policy, cfg: DictConfig, episodes=None):
    from manibot.datasets.zarr_dataset import ZarrDataset

    logger.info(f"Creating dataset with repo_id={cfg.task.dataset_repo_id} and root={cfg.task.dataset_root}")

    # 두 계열의 관측 창 규약이 다르다. 우리 이전 정책들은 창이 "지금"에서 시작하고
    # (그래서 anchor_offset=obs_horizon-1 로 보정한다), LeRobot 은 창이 "지금"에서
    # 끝나고 행동이 "지금"부터 시작한다(anchor_offset=0). obs_horizon=1 이면 둘이
    # 같지만 늘리는 순간 갈라지므로 정책에게 물어본다.
    if hasattr(policy, "get_observation_indices"):
        obs_indices = policy.get_observation_indices()
        action_indices = policy.get_action_indices()
    else:
        n_obs = cfg.policy.obs_horizon
        obs_indices = list(range(-(n_obs - 1), 1))
        action_indices = list(range(cfg.policy.pred_horizon))

    delta_timestamps = {
        **{k: [i / cfg.task.fps for i in obs_indices] for k in cfg.task.image_keys},
        cfg.task.state_key: [i / cfg.task.fps for i in obs_indices],
        cfg.task.action_key: [i / cfg.task.fps for i in action_indices],
    }

    logger.info(f"Delta timestamps:\n{pformat(delta_timestamps, indent=4)}")

    dataset = ZarrDataset(
        repo_id=cfg.task.dataset_repo_id,
        root=cfg.task.dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
    )
    return dataset


def create_dataloader(dataset, cfg: DictConfig, is_training=True):
    if hasattr(cfg, "policy"):
        drop_n_last_frames = cfg.policy.get(
            "drop_n_last_frames",
            cfg.policy.pred_horizon - cfg.policy.action_horizon - cfg.policy.obs_horizon + 1
        )
    else:
        drop_n_last_frames = cfg.network.get(
            "drop_n_last_frames",
            cfg.network.pred_horizon - cfg.network.action_horizon - cfg.network.obs_horizon + 1
        )
    if drop_n_last_frames > 0:
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=drop_n_last_frames,
            shuffle=is_training,
        )
    else:
        shuffle = is_training
        sampler = None

    batch_size = cfg.train.batch_size if is_training else cfg.val.batch_size

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.train.num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader

