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

    # 두 계열의 창 규약이 다르므로 정책에게 물어본다. 우리 정책은 관측 [0..h-1] ·
    # 행동 [0..Tp-1] 로 **같은 앵커**를 쓴다 = 행동 창이 관측 창의 시작에 붙는다
    # (LeRobot 도 같다: observation_delta_indices=range(1-h,1) ·
    #  action_delta_indices=range(1-h, 1-h+Tp)).  그래서 추론 때 첫 (h-1) 칸은 과거이고,
    # **`rollout/policy_server.make_predict_fn` 이 그걸 잘라서** 청크가 "지금"부터
    # 시작하게 만든다.  호출하는 쪽은 그 규약 하나만 안다.
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

    # APO balanced sampling — 매 배치를 정확히 correct/intervention/pre-intv 로 채운다.
    # 기댓값(WeightedRandomSampler)이 아니라 개수로 맞추는 이유: w_i 와 z_0 를 배치 안에서
    # 계산하므로 구성이 흔들리면 그 통계가 같이 흔들린다.
    ft = cfg.get("finetune", None)
    if is_training and ft is not None and ft.get("enabled", False) and ft.get("labels"):
        import numpy as np

        from manibot.datasets.apo_sampler import BalancedBatchSampler, split_pools
        from manibot.utils.intervention_labels import preference

        lab = np.load(ft.labels)
        sign, _ = preference(lab["S"])
        allowed = list(sampler) if sampler is not None else None   # drop_n_last_frames 존중
        pools = split_pools(sign, lab["has_intv"], allowed)
        bs = BalancedBatchSampler(pools, batch_size, tuple(ft.balanced))
        logger.info(f"APO balanced sampler: 풀 {[len(p) for p in pools]} · "
                    f"배치당 {bs.per} · 1 에폭 {len(bs)} 배치")
        return torch.utils.data.DataLoader(
            dataset, num_workers=cfg.train.num_workers, batch_sampler=bs,
            pin_memory=True, persistent_workers=cfg.train.num_workers > 0)

    num_workers = cfg.train.num_workers
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
        # 학습 루프가 epoch 마다 이터레이터를 새로 만들기 때문에, 이 옵션이 없으면
        # 워커를 매 epoch fork 한다(lift 는 127 스텝마다). fork 는 부모 프로세스를
        # 통째로 복제하므로 횟수를 줄이는 것 자체가 이득이다.
        persistent_workers=num_workers > 0,
    )
    return dataloader

