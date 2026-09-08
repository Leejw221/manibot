"""수집한 robomimic HDF5 -> LeRobotDataset(parquet + mp4) -> HuggingFace Hub.

**왜 필요한가**: 우리 수집은 robomimic 형식으로 쓰고(공개 데이터와 같은 변환 경로를 타려고)
학습은 zarr 를 읽는다. 둘 다 우리 안에서만 통하는 형식이라 밖에 내놓을 수 없다. LeRobot 형식은
Hub 에서 영상까지 렌더되고 남이 `LeRobotDataset(repo_id)` 한 줄로 읽는다.

**형식이 곧 관측 이름 규약이다** — task config 의 `sim.cameras` 매핑과 `state_from` 순서를
그대로 쓴다. 수집·변환·학습·업로드가 같은 이름을 쓰게 하려는 것이다.

사용:
    python -m manibot.scripts.export_lerobot task=square_scripted \\
        repo_id=Leejungwook/square_scripted single_task="put the square nut on the peg"
    # 올리지 않고 로컬로만 만들어 확인하려면 push=false
"""

import logging

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="export_lerobot", version_base="1.3")
def export(cfg: DictConfig):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sim = cfg.task.sim
    cams = OmegaConf.to_container(sim.cameras, resolve=True)   # robosuite 이름 -> 우리 이름
    low_dim = list(sim.state_from)
    src = cfg.get("hdf5_path") or cfg.task.hdf5_path

    with h5py.File(src, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        if cfg.max_episodes:
            keys = keys[: cfg.max_episodes]
        d0 = f["data"][keys[0]]
        img_shape = d0[f"obs/{next(iter(cams))}_image"].shape[1:]      # (H, W, 3)
        state_dim = sum(int(np.asarray(d0[f"obs/{k}"][0]).size) for k in low_dim)
        action_dim = d0["actions"].shape[1]

        features = {
            **{name: {"dtype": "video", "shape": list(img_shape),
                      "names": ["height", "width", "channel"]} for name in cams.values()},
            "observation.state": {"dtype": "float32", "shape": [state_dim],
                                  "names": [f"s{i}" for i in range(state_dim)]},
            "action": {"dtype": "float32", "shape": [action_dim],
                       "names": [f"a{i}" for i in range(action_dim)]},
        }
        logger.info(f"{src} · {len(keys)} 에피소드 · state {state_dim} · action {action_dim} · 이미지 {img_shape}")

        ds = LeRobotDataset.create(repo_id=cfg.repo_id, fps=int(cfg.task.fps),
                                   features=features, robot_type=str(sim.robots),
                                   use_videos=True)
        for ep, k in enumerate(keys):
            d = f["data"][k]
            acts = d["actions"][:]
            imgs = {name: d[f"obs/{cam}_image"][:] for cam, name in cams.items()}
            state = np.concatenate(
                [np.asarray(d[f"obs/{c}"][:], dtype=np.float32).reshape(len(acts), -1) for c in low_dim],
                axis=1)
            for t in range(len(acts)):
                ds.add_frame({**{name: imgs[name][t] for name in cams.values()},
                              "observation.state": state[t],
                              "action": np.asarray(acts[t], dtype=np.float32),
                              "task": cfg.single_task})
            ds.save_episode()
            if (ep + 1) % 20 == 0 or ep + 1 == len(keys):
                logger.info(f"  {ep + 1}/{len(keys)} 에피소드")

    # ⚠ finalize() 를 빠뜨리면 **잘린 parquet 이 올라간다**. save_episode 는
    # parallel_encoding=True 라 백그라운드로 쓰는데, 그게 끝나기 전에 push 가 나가면
    # footer 없는 파일이 Hub 에 올라가고 데이터셋 뷰어가
    # "Parquet magic bytes not found in footer" 로 죽는다 (2026-09-08 실측:
    # 5,651,594 바이트 원본이 5,483,546 바이트로 잘려 올라갔다).
    ds.finalize()
    logger.info(f"로컬 생성 완료: {ds.root}")
    if cfg.push:
        ds.push_to_hub(private=cfg.private, tags=list(cfg.tags or []))
        logger.info(f"업로드 완료: https://huggingface.co/datasets/{cfg.repo_id}")
    else:
        logger.info("push=false — 업로드 안 함")


def main():
    export()


if __name__ == "__main__":
    main()
