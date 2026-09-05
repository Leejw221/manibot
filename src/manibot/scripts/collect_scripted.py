"""스크립트 정책으로 시연을 모아 robomimic 형식 HDF5 로 쓴다.

**왜 스크립트 수집인가**: 사람이 텔레오퍼레이션으로 모으기 전에, 이 하드웨어·이 배치에서
task 가 성립하는지와 학습 경로 전체(수집 -> 변환 -> 학습 -> rollout)가 도는지를 먼저 확인한다.

출력 형식은 `convert_hdf5_to_zarr.py` 가 그대로 읽을 수 있는 robomimic 관례를 따른다:
  data/demo_{i}/actions            (T, action_dim)
  data/demo_{i}/obs/<key>          low_dim 은 (T, d), 이미지는 (T, H, W, 3) uint8
  data/demo_{i}/stage              (T,) int64 — 스크립트 정책의 구간 라벨
`stage` 를 같이 남기는 이유는 나중에 stage 조건화 실험에서 **정답 분절**이 필요하기
때문이다. 학습에 쓰지 않아도 비용이 거의 없다.

사용: python -m manibot.scripts.collect_scripted n_demos=100
"""

import os
import time

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


def collect(cfg: DictConfig):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from robosuite.controllers import load_composite_controller_config

    from manibot.envs.microwave import MicrowaveTask  # noqa: F401  (레지스트리 등록)
    from manibot.envs.microwave_expert import MicrowaveExpert
    import robosuite as suite

    sim = cfg.task.sim
    cams = list(OmegaConf.to_container(sim.cameras, resolve=True))
    low_dim = list(sim.state_from)
    out = cfg.get("out_path") or cfg.task.hdf5_path
    os.makedirs(os.path.dirname(out), exist_ok=True)

    env = suite.make(
        env_name=sim.env_name, robots=sim.robots,
        controller_configs=load_composite_controller_config(controller="BASIC",
                                                            robot=sim.robots),
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, use_object_obs=True,
        camera_names=cams, camera_heights=sim.image_size, camera_widths=sim.image_size,
        control_freq=cfg.task.fps, horizon=cfg.max_steps + 10, ignore_done=True,
    )
    env.reset()                         # seed 를 주기 전에 한 번 만들어 둔다
    rng = np.random.RandomState(cfg.seed)

    kept, tried, t0 = 0, 0, time.time()
    with h5py.File(out, "w") as f:
        grp = f.create_group("data")
        while kept < cfg.n_demos:
            tried += 1
            env.reset()
            ex = MicrowaveExpert(env)
            obs = env._get_observations()
            traj = {k: [] for k in low_dim + [f"{c}_image" for c in cams]}
            acts, stages = [], []
            stage_ids = {}
            for _ in range(cfg.max_steps):
                a = ex.act()
                for k in low_dim:
                    traj[k].append(np.asarray(obs[k], dtype=np.float32).ravel())
                for c in cams:
                    # robosuite 는 아래위가 뒤집힌 채로 준다 — robomimic 관례대로 세워 저장한다.
                    traj[f"{c}_image"].append(obs[f"{c}_image"][::-1])
                acts.append(np.asarray(a, dtype=np.float32))
                stages.append(stage_ids.setdefault(ex.stage, len(stage_ids)))
                obs, _, _, _ = env.step(a)
                if ex.done:
                    break
            ok = bool(env._check_success())
            if not ok and not cfg.keep_failures:
                print(f"  [{tried:3d}] 실패 — 버림 ({dict(env.progress)})")
                continue
            d = grp.create_group(f"demo_{kept}")
            d.create_dataset("actions", data=np.asarray(acts, dtype=np.float32))
            d.create_dataset("stage", data=np.asarray(stages, dtype=np.int64))
            o = d.create_group("obs")
            for k, v in traj.items():
                arr = np.asarray(v)
                o.create_dataset(k, data=arr, dtype=arr.dtype)
            d.attrs["num_samples"] = len(acts)
            d.attrs["success"] = ok
            kept += 1
            print(f"  [{tried:3d}] demo_{kept-1}: {len(acts):5d} 프레임 · 구간 "
                  f"{len(stage_ids)}개 · 누적 {kept}/{cfg.n_demos} "
                  f"({time.time()-t0:.0f}s)")
        grp.attrs["total"] = int(sum(grp[k].attrs["num_samples"] for k in grp))
        grp.attrs["env_args"] = OmegaConf.to_yaml(sim)
        grp.attrs["stage_names"] = list(stage_ids)
    env.close()
    size = os.path.getsize(out) / 1e9
    print(f"\n{kept} demos / {tried} 시도 · {grp.attrs['total'] if False else ''} "
          f"-> {out} ({size:.2f} GB, {time.time()-t0:.0f}s)")
    return out


@hydra.main(config_path="../configs", config_name="collect_scripted", version_base=None)
def main(cfg: DictConfig):
    collect(cfg)


if __name__ == "__main__":
    main()
