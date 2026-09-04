# Convert LeRobot datasets to Zarr format for fast training.
#
# output_name is auto-derived from repo_id; use -o/--output to override.
#
# Images are stored at native resolution; resizing/cropping happens in the
# policy's vision encoder (resize_shape/crop_shape) right before the model.
#
# Usage:
#   Convert from HuggingFace cache (~/.cache/huggingface/lerobot/<repo_id>/):
#       python convert.py -r Leejungwook/cube_stack
#   Convert from local path (repo_id auto-inferred from last two parts of path):
#       python convert.py --local-dir /path/to/Leejungwook/new_task
#   Override output directory:
#       python convert.py -r Leejungwook/new_task -o data/piper_new_task

import argparse
import json
import shutil
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as torch_mp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


# Use file_system sharing strategy to avoid /dev/shm exhaustion
# with large multi-camera datasets (especially bimanual).
torch_mp.set_sharing_strategy('file_system')

from manibot.datasets.replay_buffer import ReplayBuffer
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Project root (src/flare/scripts/convert.py -> manipulation_pipeline)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Default output directory
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"

# Default image resolution (H, W) stored in the zarr — the policy's resize_shape,
# i.e. the model input size before the random/center crop. Keeps the in-RAM zarr
# small while preserving the crop augmentation. Use --native to store full res.
DEFAULT_RESIZE = (240, 320)


def make_json_serializable(obj):
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def create_zarr_dataset_from_lerobot(
    repo_id: str,
    root: Path,
    episodes: list[int] | None = None,
    remove_keys: list[str] | None = None,
    local_dir: str | Path | None = None,
    target_fps: int | None = None,
    resize: tuple[int, int] | None = None,
):
    if remove_keys is None:
        remove_keys = []

    # Load LeRobot dataset
    lerobot_kwargs = {"repo_id": repo_id}
    if episodes is not None:
        lerobot_kwargs["episodes"] = episodes
    if local_dir is not None:
        lerobot_kwargs["root"] = Path(local_dir)

    print(f"Loading: {repo_id}")
    if local_dir:
        print(f"  Source: {local_dir}")
    print(f"  Output: {root}")
    print(f"  Images: {'native resolution' if resize is None else f'resized to {resize[0]}x{resize[1]}'}")

    dataset = LeRobotDataset(**lerobot_kwargs)

    # Episode boundaries — read from source (use dataset.root provided by lerobot)
    source_dir = Path(dataset.root)
    
    # Read all episode parquet files (may be split across multiple files)
    ep_parquet_dir = source_dir / "meta" / "episodes"
    if not ep_parquet_dir.exists():
        raise FileNotFoundError(
            f"Cannot find episode metadata at {ep_parquet_dir}. "
            "Please ensure the dataset has the expected LeRobot format."
        )
    # Read all episode parquet files (may be split across multiple files)
    ep_parquet_dir = source_dir / "meta" / "episodes"
    ep_files = sorted(ep_parquet_dir.rglob("*.parquet"))
    episodes_meta = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
    ep_from = episodes_meta["dataset_from_index"].tolist()
    ep_to = episodes_meta["dataset_to_index"].tolist()

    # Metadata
    fps = dataset.meta.fps
    num_episodes = dataset.num_episodes
    
    if target_fps is not None:
        if fps % target_fps != 0:
            raise ValueError(f"Source FPS ({fps}) must be divisible by target FPS ({target_fps}).")
        downsample_ratio = fps // target_fps
        out_fps = target_fps
    else:
        downsample_ratio = 1
        out_fps = fps

    out_num_frames = sum(len(range(ep_from[i], ep_to[i], downsample_ratio)) for i in range(num_episodes))

    features = {k: v for k, v in dataset.features.items() if k not in remove_keys}

    camera_keys = [k for k in dataset.meta.camera_keys if k in features]
    video_keys = [k for k in dataset.meta.video_keys if k in features]
    image_keys = [k for k in (dataset.meta.image_keys if hasattr(dataset.meta, 'image_keys') else []) if k in features]

    # When resizing, reflect the new resolution in the stored feature shapes (HWC).
    # The default resize is 4:3 while most cameras here record 16:9, so report the
    # change and flag any aspect-ratio distortion instead of silently squashing.
    if resize is not None:
        h_dst, w_dst = resize
        for k in sorted(set(camera_keys + video_keys + image_keys)):
            shape = features.get(k, {}).get("shape")
            if shape is None:
                continue
            h_src, w_src = shape[0], shape[1]
            print(f"  {k}: {h_src}x{w_src} -> {h_dst}x{w_dst}")
            if abs(w_src / h_src - w_dst / h_dst) > 1e-2:
                print(f"    WARNING: aspect ratio changes "
                      f"({w_src / h_src:.3f} -> {w_dst / h_dst:.3f}). "
                      f"Pass --resize H W to match, or --native to keep the source size.")
            features[k] = {**features[k], "shape": [h_dst, w_dst, 3]}

    stats = {k: v for k, v in dataset.meta.stats.items() if k in features}

    tasks_df = dataset.meta.tasks
    tasks = {int(row.task_index): task_name for task_name, row in tasks_df.iterrows()}
    tasks_reversed = {v: k for k, v in tasks.items()}

    # Remove old output
    if root.exists():
        print(f"Removing existing: {root}")
        shutil.rmtree(root)

    # Create replay buffer
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=root, mode="a")

    # Save metadata
    config = {
        "repo_id": repo_id,
        "stats": stats,
        "num_frames": out_num_frames,
        "num_episodes": num_episodes,
        "features": features,
        "camera_keys": camera_keys,
        "video_keys": video_keys,
        "image_keys": image_keys,
        "fps": out_fps,
        "tasks": tasks,
    }
    with open(root / "config.json", "w") as f:
        json.dump(make_json_serializable(config), f, indent=4)

    def convert(k, v: torch.Tensor):
        dtype = features[k]["dtype"]
        if dtype in ["image", "video"]:
            # v: (N, C, H, W) float in [0, 1]. Optionally resize before storing so
            # the zarr (loaded fully into RAM by ZarrDataset) stays small.
            if resize is not None:
                v = torch.nn.functional.interpolate(
                    v, size=resize, mode="bilinear", align_corners=False
                )
            v = v.permute(0, 2, 3, 1)
            v = (v * 255).to(torch.uint8).numpy()
        else:
            v = v.numpy()
        return v

    # Convert episodes
    for i in range(num_episodes):
        from_idx, to_idx = ep_from[i], ep_to[i]
        indices = list(range(from_idx, to_idx, downsample_ratio))
        ep_len = len(indices)
        print(f"  Episode {i}/{num_episodes} ({ep_len} frames)...")
        subset = Subset(dataset, indices)
        dataloader = DataLoader(
            subset, batch_size=16, shuffle=False, num_workers=0
        )
        data = []
        for batch in tqdm(dataloader, leave=False):
            if "task_index" in batch:
                batch["task_index"] = torch.tensor(
                    [tasks_reversed[k] for k in batch["task"]], dtype=int
                )
                del batch["task"]
            batch["episode_index"] = torch.full_like(batch["episode_index"], i)
            data.append(batch)

        batch = {k: torch.cat([d[k] for d in data], dim=0) for k in data[0].keys()}
        assert batch["action"].shape[0] == ep_len
        batch = {k: convert(k, v) for k, v in batch.items() if k in features}
        replay_buffer.add_episode(batch, compressors="disk")

    print(f"\nDone! {num_episodes} episodes, {out_num_frames} frames → {root}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot datasets to Zarr format.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-r", "--repo", type=str, metavar="REPO_ID",
                       help="HuggingFace repo ID (e.g. Leejungwook/cube_stack).")
    group.add_argument("--local-dir", type=str, default=None,
                       help="Local path to LeRobot dataset. repo_id auto-inferred from path.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory. Default: data/<dataset_name>")
    parser.add_argument("--target-fps", type=int, default=None,
                        help="Target FPS to downsample the dataset (e.g., 10).")
    parser.add_argument("--resize", type=int, nargs=2, default=list(DEFAULT_RESIZE), metavar=("H", "W"),
                        help=f"Resize images to H W before storing. Default: {DEFAULT_RESIZE[0]} {DEFAULT_RESIZE[1]} "
                             "(the policy resize_shape / model input size). e.g. --resize 360 640")
    parser.add_argument("--native", action="store_true",
                        help="Store images at native resolution (no resize). Large in-RAM zarr.")

    args = parser.parse_args()

    # Native overrides the resize default; otherwise use the (default or given) H W.
    resize = None if args.native else tuple(args.resize)

    local_dir = None
    if args.local_dir:
        local_dir = args.local_dir
        # Auto-infer repo_id from local path: .../Leejungwook/cube_stack → Leejungwook/cube_stack
        parts = Path(local_dir).parts
        repo_id = f"{parts[-2]}/{parts[-1]}"
        print(f"Auto-inferred repo_id: {repo_id}")
    else:
        repo_id = args.repo

    # Auto-derive output_name from repo_id: "Leejungwook/foo-bar" → "piper_foo_bar"
    auto_output_name = "piper_" + repo_id.split("/")[-1].replace("-", "_")
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / auto_output_name

    create_zarr_dataset_from_lerobot(
        repo_id=repo_id,
        root=output_dir,
        episodes=None,
        remove_keys=[],
        local_dir=local_dir,
        target_fps=args.target_fps,
        resize=resize,
    )


if __name__ == "__main__":
    main()
