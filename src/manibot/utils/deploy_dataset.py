"""배포 데이터를 LeRobotDataset 으로 쓰는 공용 코드 — 실물 `record.py` 와 같은 형식.

`collect_intervention`(사람이 개입하는 수집)과 `eval`(개입 없는 롤아웃)이 **같은 형식**으로
써야 한다. 따로 구현하면 필드 이름이나 라벨이 갈라져 나중에 못 합친다. 그래서 여기 모은다.

라벨은 실물 규약을 쓴다 — `action_mode` 0=rollout · 1=intervention.
APO 의 c=0(개입 직전 K 프레임)은 학습 직전에 `intervention_labels.relabel_preintv` 로 만든다.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def make_features(raw_obs, cams, low_dim, action_dim, use_videos=True):
    """robosuite 관측 하나에서 LeRobot 특징 스키마를 만든다.

    `cams` 는 robosuite 이름 -> 우리 관측 이름 매핑이다 (task config 의 sim.cameras).
    """
    img_shape = list(np.asarray(raw_obs[f"{next(iter(cams))}_image"]).shape)
    state_dim = int(sum(np.asarray(raw_obs[k]).size for k in low_dim))
    return {
        **{name: {"dtype": "video" if use_videos else "image", "shape": img_shape,
                  "names": ["height", "width", "channel"]} for name in cams.values()},
        "observation.state": {"dtype": "float32", "shape": [state_dim],
                              "names": [f"s{i}" for i in range(state_dim)]},
        "action": {"dtype": "float32", "shape": [int(action_dim)],
                   "names": [f"a{i}" for i in range(int(action_dim))]},
        "action_mode": {"dtype": "int64", "shape": (1,), "names": None},
    }


def open_or_resume(repo_id, root, fps, features, robot_type, use_videos=True):
    """이미 있으면 이어받고 없으면 만든다.

    중간에 죽으면 finalize 전의 parquet 은 footer 가 없어 못 읽는다 — 그래서 주기적으로
    finalize 하고 여기로 다시 연다. 실물 `rollout_intervention.py` 도 같은 자리에서
    resume 을 쓴다.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # ⚠ 폴더가 있다는 것만으로 판단하면 안 된다 — 로그가 먼저 들어가 폴더만 생겨 있을 수 있다.
    # 데이터셋이 실제로 있는지는 meta/info.json 으로 본다.
    if root and (Path(root) / "meta" / "info.json").exists():
        logger.info(f"이어받기: {root}")
        return LeRobotDataset.resume(repo_id=repo_id, root=root)
    return LeRobotDataset.create(repo_id=repo_id, fps=int(fps), root=root, features=features,
                                 robot_type=str(robot_type), use_videos=use_videos)


def make_frame(raw_obs, cams, low_dim, action, mode, task):
    """robosuite 관측 + 행동 -> LeRobot 프레임 하나.

    robosuite 는 이미지를 아래위 뒤집힌 채로 준다 — robomimic 관례대로 세워 저장한다.
    """
    return {
        **{name: raw_obs[f"{c}_image"][::-1] for c, name in cams.items()},
        "observation.state": np.concatenate(
            [np.asarray(raw_obs[k], dtype=np.float32).ravel() for k in low_dim]),
        "action": np.asarray(action, dtype=np.float32),
        "action_mode": np.array([int(mode)], dtype=np.int64),
        "task": task,
    }


def write_success(root, ep_success):
    """성공 여부는 시뮬에만 있는 정보라 LeRobot 스키마를 안 건드리고 옆에 둔다."""
    (Path(root) / "episode_success.json").write_text(json.dumps(ep_success))


def read_success(root):
    p = Path(root) / "episode_success.json"
    return json.loads(p.read_text()) if p.exists() else []
