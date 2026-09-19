"""스크립트 정책으로 시연을 모아 robomimic 형식 HDF5 로 쓴다.

**왜 스크립트 수집인가**: 사람이 텔레오퍼레이션으로 모으기 전에, 이 하드웨어·이 배치에서
task 가 성립하는지와 학습 경로 전체(수집 -> 변환 -> 학습 -> rollout)가 도는지를 먼저 확인한다.

출력 형식은 `convert_hdf5_to_zarr.py` 가 그대로 읽을 수 있는 robomimic 관례를 따른다:
  data/demo_{i}/actions            (T, action_dim)
  data/demo_{i}/obs/<key>          low_dim 은 (T, d), 이미지는 (T, H, W, 3) uint8
  data/demo_{i}/stage              (T,) int64 — 스크립트 정책의 구간 라벨
`stage` 를 같이 남기는 이유는 나중에 stage 조건화 실험에서 **정답 분절**이 필요하기
때문이다. 학습에 쓰지 않아도 비용이 거의 없다.

사용: python -m manibot.scripts.collect_scripted task=nero_drawer n_demos=50
"""

import os
import time

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from manibot.utils.seeding import seed_sim_env
from manibot.utils.task_utils import register_sim_modules, resolve, sim_controller_config


def _set_nut_pose(env, x=None, y=None, yaw=None):
    """SquareNut 의 위치·yaw 를 지정한다 (free joint 라 qpos 로 직접 쓴다). None 인 축은 둔다.

    **yaw 를 왜 건드리나**: 배치 초기화가 yaw 를 360° 균등 무작위로 준다. 평가 200판을 각도
    60° 구간으로 나누니 base 성공률이 27%~74% 로 47%p 벌어졌고, 위치 구간 간 편차(8.5%)보다
    방향 구간 간 편차(15.1%)가 2배였다. 게다가 학습이 방향별로 +37%p/-28%p 로 주고받아 합이
    0 이 된다 — 손실을 아홉 가지로 바꿔도 성공률이 안 움직인 이유다 [측정 2026-09-19].
    방향을 고정하면 그 축이 사라져 라운드 효과를 **잴 수 있는** 영역이 된다.

    **위치를 왜 건드리나**: 공개 데이터의 너트 구역이 x 5mm · y 11.5cm 짜리 **띠**다(공개 200개
    전수 확인). 방향을 고정하면 남는 변이가 1차원뿐이라, 너트 긴 축(12.3cm)의 절반씩 x 를
    앞뒤로 넓혀 실물 실험과 같은 2D 직사각 구역으로 만든다.

    ⚠ 원형 너트는 y<0 쪽에 놓이므로 이 구역(y>0)과 안 겹친다 [배치 확인 2026-09-19].
    """
    m = env.sim.model
    adr = m.jnt_qposadr[m.joint_name2id("SquareNut_joint0")]
    off = env.model.mujoco_arena.table_offset
    if x is not None:
        env.sim.data.qpos[adr + 0] = x + off[0]
    if y is not None:
        env.sim.data.qpos[adr + 1] = y + off[1]
    if yaw is not None:
        env.sim.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]  # (w,x,y,z)
    env.sim.forward()


def _kw(node):
    """OmegaConf 노드 -> dict. null 이면 빈 dict."""
    return OmegaConf.to_container(node, resolve=True) if node is not None else {}


def collect(cfg: DictConfig):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import robosuite as suite

    sim = cfg.task.sim
    register_sim_modules(sim)                        # 환경·로봇 레지스트리 등록
    Expert = resolve(sim.expert)

    # 정책 입력 카메라 + 관찰 전용 카메라. 관찰용은 hdf5 에만 남고 zarr 변환에는 안 간다.
    policy_cams = list(OmegaConf.to_container(sim.cameras, resolve=True))
    monitor_cams = list(sim.get("monitor_cameras") or [])
    cams = policy_cams + monitor_cams
    low_dim = list(sim.state_from)
    out = cfg.get("out_path") or cfg.task.hdf5_path
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ctrl = sim_controller_config(sim)
    if ctrl is None:
        from robosuite.controllers import load_composite_controller_config
        ctrl = load_composite_controller_config(controller="BASIC", robot=sim.robots)

    env = suite.make(
        env_name=sim.env_name, robots=sim.robots, controller_configs=ctrl,
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, use_object_obs=True,
        camera_names=cams, camera_heights=sim.image_size, camera_widths=sim.image_size,
        control_freq=cfg.task.fps, horizon=cfg.max_steps + 10, ignore_done=True,
        **_kw(sim.get("env_kwargs")),
    )
    env.reset()                         # seed 를 주기 전에 한 번 만들어 둔다
    seed_sim_env(env, cfg.seed)         # 전역 np.random 만으로는 배치가 안 잡힌다

    kept, tried, t0 = 0, 0, time.time()
    with h5py.File(out, "w") as f:
        grp = f.create_group("data")
        while kept < cfg.n_demos:
            tried += 1
            env.reset()
            # kept 로 색인한다 — 실패해 버려도 같은 격자점을 다시 시도해 커버리지가 정확해진다
            if cfg.yaw_grid:
                _set_nut_pose(env, yaw=-np.pi + 2 * np.pi * (kept + 0.5) / cfg.n_demos)
            if cfg.get("xy_grid") or cfg.get("fixed_yaw_deg") is not None:
                g = _kw(cfg.get("xy_grid")) or None
                gx = gy = None
                if g:
                    gx, gy = g[kept % len(g)]      # 지점을 돌아가며 채운다 (중간에 끊겨도 고르게)
                fy = cfg.get("fixed_yaw_deg")
                _set_nut_pose(env, x=gx, y=gy,
                              yaw=None if fy is None else np.radians(float(fy)))
            ex = Expert(env)
            # 시뮬레이터 예행 동작 — **기록 전에** 한다. 프로세스 안에서 첫 파지·당기기만
            # 결과가 다르다(서랍 0.08cm vs 이후 11.84cm, 결정론적) [실측 2026-09-06].
            if hasattr(ex, "warmup"):
                ex.warmup()
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
                # progress 는 우리가 만든 env 만 갖고 있다 (robomimic 기본 env 엔 없다).
                prog = dict(getattr(env, "progress", {}))
                print(f"  [{tried:3d}] 실패 — 버림 (마지막 구간 {ex.stage}"
                      + (f" · {prog}" if prog else "") + ")")
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
        total = int(sum(grp[k].attrs["num_samples"] for k in grp))
        grp.attrs["total"] = total
        grp.attrs["env_args"] = OmegaConf.to_yaml(sim)
        grp.attrs["stage_names"] = list(stage_ids)
    env.close()
    size = os.path.getsize(out) / 1e9
    print(f"\n{kept} demos / {tried} 시도 · {total} 프레임 "
          f"-> {out} ({size:.2f} GB, {time.time()-t0:.0f}s)")
    return out


@hydra.main(config_path="../configs", config_name="collect_scripted", version_base=None)
def main(cfg: DictConfig):
    collect(cfg)


if __name__ == "__main__":
    main()
