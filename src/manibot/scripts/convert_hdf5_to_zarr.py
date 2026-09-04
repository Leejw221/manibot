"""robomimic 형식 HDF5(collect.py/merge_rounds.py 산출물) -> Zarr(ReplayBuffer) 변환.

collect.py(사람 개입 수집)·merge_rounds.py(라운드 누적)는 그대로 hdf5를 쓴다 - 둘 다 robomimic
관례(마스크 필터키, demo_i 그룹)에 맞물려 있어 굳이 바꿀 이유가 없다. 이 스크립트는 그 결과물을
**학습 직전에** Zarr로 변환하는 별도 단계다.

바꾸는 이유(2026-07-27 밤): robomimic SequenceDataset은 h5py의 fork-불안정성 때문에
num_workers>=1이면 이미지 전체를 메모리에 캐싱해야 하는데(diffusion_trainer.py의 cache_mode
분기 주석 참고), 라운드가 쌓여 데이터가 커질수록 이게 OOM으로 이어진다(round0 44demo/23815
프레임에서 실측). Zarr는 이 제약이 없어 멀티워커 + 저메모리 로딩을 동시에 가능케 한다.

이미지는 collect.py가 이미 raw(HWC, uint8) 포맷으로 저장하므로(2026-07-27 이미지 포맷 버그
수정) 별도 변환 없이 그대로 복사한다.

학습 파이프라인 연결: task.yaml에 `dataset_backend=zarr`+`zarr_path=...`를 얹으면(로봇수트
task도 CLI에서 `+task.dataset_backend=zarr +task.zarr_path=...`로 추가 가능) 시뮬레이터
(env_backend, robosuite 그대로 유지)는 안 건드리고 학습 데이터 저장 포맷만 hdf5->Zarr로
바뀐다(task_utils.uses_zarr_dataset이 이 둘을 독립 축으로 분리— 2026-07-27 밤). "저장은
Zarr, 시뮬레이션은 robosuite"로 Square 200개 데모 학습을 실전 검증함(2026-07-29,
num_workers=4로 OOM 없이 정상 진행).

사용: python -m manibot.scripts.convert_hdf5_to_zarr task=transport_demo20 \
    hdf5_path=data/intervention/transport_round0_cumulative.hdf5 \
    zarr_path=data/intervention/transport_round0_cumulative.zarr
(hdf5_path을 생략하면 task.hdf5_path을 그대로 쓴다 - 원본 robomimic 데이터셋 변환용.)
"""

import json
from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig

from omegaconf import OmegaConf

from manibot.utils.task_utils import is_image_task


def convert_hdf5_to_zarr(hdf5_path, zarr_path, state_from, cameras, fps, task_name,
                         state_key="observation.state", action_key="action", filter_key=None):
    """robomimic hdf5 -> zarr(ReplayBuffer) + config.json.

    robosuite 이름을 lerobot 이름으로 바꾼다. state_from 을 그 순서대로 이어붙여
    state_key 하나로 만드는데, 평가 때 env 어댑터(envs.robomimic.wrap_lerobot_obs)가
    쓰는 순서와 반드시 같아야 한다 — 어긋나면 학습과 평가가 서로 다른 상태 벡터를
    보게 되고, 그 차이는 에러 없이 성능으로만 나타난다.

    config.json 은 ZarrDataset 이 읽는 메타다. 실물 쪽 convert.py 와 같은 형식이라
    실물·시뮬이 같은 Dataset 클래스를 쓴다.
    """
    from manibot.datasets.replay_buffer import ReplayBuffer

    with h5py.File(hdf5_path, "r") as fin:
        if filter_key:
            demo_names = [d.decode() if isinstance(d, bytes) else d for d in fin["mask"][filter_key][:]]
        else:
            demo_names = list(fin["data"].keys())
        demo_names = sorted(demo_names, key=lambda k: int(k.split("_")[1]))

        root = Path(zarr_path)
        buffer = ReplayBuffer.create_from_path(str(root), mode="a")
        acc = {state_key: [], action_key: []}   # 통계용 — 상태·행동만 모은다(작다)
        image_shapes = {}

        for ep_idx, name in enumerate(demo_names):
            demo = fin[f"data/{name}"]
            obs = demo["obs"]
            n = len(demo["actions"])
            data = {
                state_key: np.concatenate(
                    [obs[k][()].astype(np.float32).reshape(n, -1) for k in state_from], axis=-1
                ),
                action_key: demo["actions"][()].astype(np.float32),
                # ZarrDataset 이 에피소드 경계·시간 정합을 이 둘로 검사한다.
                "episode_index": np.full(n, ep_idx, dtype=np.int64),
                "timestamp": (np.arange(n, dtype=np.float32) / fps),
            }
            for cam, key in cameras.items():
                arr = obs[f"{cam}_image"][()]  # 이미 raw(HWC, uint8) - 변환 불필요
                data[key] = arr
                image_shapes[key] = list(arr.shape[1:])
            if "action_mode" in demo:
                data["action_mode"] = demo["action_mode"][()].astype(np.int64)
            for k in acc:
                acc[k].append(data[k])
            buffer.add_episode(data)
            print(f"{name}: {n} frames -> zarr (누적 {buffer.n_steps} steps)")

    stats = {}
    for k, chunks in acc.items():
        a = np.concatenate(chunks, axis=0)
        stats[k] = {
            "mean": a.mean(axis=0).tolist(), "std": a.std(axis=0).tolist(),
            "min": a.min(axis=0).tolist(), "max": a.max(axis=0).tolist(),
        }
    # 이미지 통계는 넣지 않는다 — task 설정의 override_stats 가 ImageNet 값을 준다
    # (실물 쪽 piper_*.yaml 과 같은 관례). 전 프레임 평균/표준편차를 구하는 비용도 크다.
    features = {
        state_key: {"dtype": "float32", "shape": [len(stats[state_key]["mean"])]},
        action_key: {"dtype": "float32", "shape": [len(stats[action_key]["mean"])]},
        **{k: {"dtype": "image", "shape": v} for k, v in image_shapes.items()},
    }
    config = {
        "repo_id": None,
        "stats": stats,
        "num_frames": int(buffer.n_steps),
        "num_episodes": len(demo_names),
        "features": features,
        "camera_keys": list(image_shapes),
        "video_keys": [],
        "image_keys": list(image_shapes),
        "fps": fps,
        "tasks": {0: task_name},
    }
    with open(root / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"변환 완료: {len(demo_names)} demos, {buffer.n_steps} steps -> {zarr_path}")
    return zarr_path


@hydra.main(config_path="../configs", config_name="convert_hdf5_to_zarr", version_base=None)
def main(cfg: DictConfig):
    sim = cfg.task.sim
    cameras = OmegaConf.to_container(sim.cameras, resolve=True) if is_image_task(cfg.task) else {}
    hdf5_path = cfg.get("hdf5_path", None) or cfg.task.hdf5_path
    convert_hdf5_to_zarr(
        hdf5_path, cfg.task.dataset_root, list(sim.state_from), cameras,
        fps=cfg.task.fps, task_name=cfg.task.name,
        state_key=cfg.task.state_key, action_key=cfg.task.action_key,
        filter_key=cfg.get("filter_key", None),
    )


if __name__ == "__main__":
    main()
