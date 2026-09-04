"""task 설정을 읽는 한 곳.

이름 규약: 학습·평가가 보는 관측 이름은 실물이든 시뮬이든 lerobot 형식 하나다
(observation.state · observation.images.<name> · action). robosuite 고유 이름
(robot0_eef_pos, agentview_image)은 task.sim 블록 안에만 존재하고, env 어댑터와
hdf5->zarr 변환이 그 경계에서 이름을 바꾼다. 이렇게 두지 않으면 같은 학습 코드가
시뮬/실물마다 다른 키를 찾게 되고, 결국 분기가 남는다.
"""

from omegaconf import OmegaConf


def is_sim_task(task_cfg):
    """시뮬 env 를 만들 수 있는 task 인지 — task.sim 블록의 유무가 유일한 기준이다."""
    return task_cfg.get("sim", None) is not None


def is_image_task(task_cfg):
    return bool(task_cfg.get("image_keys", None))


def derive_task_meta(task_cfg, dataset_meta):
    """state_dim·action_dim 을 데이터셋 메타(config.json)에서 읽어 task_cfg 에 덮어쓴다.

    손으로 적은 값과 데이터가 어긋나는 사고를 막기 위해 데이터를 유일한 출처로 삼는다
    (2026-07-21 에 실제로 관측 스킴이 바뀌었는데 차원을 안 고쳐 생긴 사고가 있었다).
    어떤 키를 쓸지(image_keys·sim.state_from)는 데이터의 사실이 아니라 실험 설계
    선택이라 건드리지 않는다.
    """
    features = dataset_meta.features
    OmegaConf.set_struct(task_cfg, False)
    task_cfg.state_dim = int(features[task_cfg.state_key]["shape"][-1])
    task_cfg.action_dim = int(features[task_cfg.action_key]["shape"][-1])
    OmegaConf.set_struct(task_cfg, True)
    return task_cfg


def make_eval_env(task_cfg, render=False, renderer="mjviewer", image_size_override=None):
    """평가용 env 생성. 반환되는 env 의 관측은 lerobot 이름으로 나온다.

    image task + render=True 는 여기서 처리하지 않는다(호출부 책임) — cv2 오프스크린
    렌더와 mjviewer 온스크린이 GL 컨텍스트 충돌로 세그폴트하는 게 알려진 지뢰라,
    image 쪽은 생성 *후에* env.env.has_renderer 등을 직접 패치한다.
    """
    if not is_sim_task(task_cfg):
        raise ValueError(
            f"task '{task_cfg.name}' 에는 sim 블록이 없다 — 시뮬 env 를 만들 수 없는 실물 task 다."
        )
    sim = task_cfg.sim
    backend = sim.get("backend", "robosuite")
    if backend != "robosuite":
        raise NotImplementedError(
            f"시뮬 백엔드 '{backend}' 는 아직 없다. 현재 지원: robosuite. "
            "Piper 시뮬이 필요하면 AgileX 공식 MJCF "
            "(agx_arm_sim/mujoco/agilex_arm/agilex_piper) 로 새로 만드는 것이 대체 경로다."
        )

    from manibot.envs.robomimic import make_image_env, make_lowdim_env, wrap_lerobot_obs

    env_kwargs = OmegaConf.to_container(sim.env_kwargs, resolve=True) if sim.get("env_kwargs") else None
    gripper_types = sim.get("gripper_types", None)
    state_from = list(sim.state_from)

    if is_image_task(task_cfg):
        cameras = OmegaConf.to_container(sim.cameras, resolve=True)
        raw = make_image_env(
            sim.env_name, sim.robots,
            lowdim_keys=state_from,
            rgb_keys=[f"{cam}_image" for cam in cameras],
            camera_names=list(cameras),
            image_size=image_size_override or sim.image_size,
            gripper_types=gripper_types, env_kwargs=env_kwargs,
        )
    else:
        cameras = {}
        raw = make_lowdim_env(
            sim.env_name, sim.robots, state_from,
            render=render, renderer=renderer,
            gripper_types=gripper_types, env_kwargs=env_kwargs,
        )
    return wrap_lerobot_obs(raw, state_from, cameras, state_key=task_cfg.state_key)
