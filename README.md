# manibot

실물 로봇과 시뮬레이션을 **하나의 학습·평가 경로**로 다루기 위한 저장소.
구조는 [LeRobot](https://github.com/huggingface/lerobot)을 따른다.

## 왜 하나의 repo 인가

실물과 시뮬을 각각의 repo 로 두면 같은 실험을 서로 다른 스크립트로 돌리게 되고,
그 차이가 결과에 섞여 들어온다. 이 저장소는 sim/real 의 차이를 **데이터를 만드는
경로에만** 가두고, `LeRobotDataset` 아래로 내려온 뒤부터는 학습·평가 코드를 한 벌만 둔다.

```
실물 (robots/, teleoperators/, cameras/)  ┐
                                          ├─→  LeRobotDataset  ─→  policies/  ─→  rollout/
시뮬 (envs/)                              ┘         (HF Hub)
```

## 구조

LeRobot 의 서브패키지 이름을 그대로 쓴다. **LeRobot 에 자리가 없는 이름은 만들지 않는다.**

| 디렉토리 | 내용 |
|---|---|
| `cameras/` | RealSense · Orbbec |
| `robots/` | 실물 로봇 (Piper · 양팔) |
| `teleoperators/` | 텔레오퍼레이션 장치 |
| `envs/` | 시뮬레이션 환경 (robosuite) |
| `policies/` | ACT · Diffusion Policy · Flow Matching |
| `model/` | 백본·네트워크 |
| `datasets/` | 수집과 LeRobotDataset 변환 |
| `rollout/` | 추론·평가 실행 |
| `scripts/` | record · train · eval 진입점 |
| `configs/` · `utils/` | 설정과 공용 유틸 |

## 설치

```bash
pip install -e .              # 공통
pip install -e '.[hardware]'  # 실물 로봇이 붙은 기기
pip install -e '.[sim]'       # 시뮬레이션
pip install -e '.[train]'     # 학습
```

## 만든 방식

[LeRobot](https://github.com/huggingface/lerobot) 의 구조와 정책 구현을 참고했다. 정리 작업은 Claude Code로 진행했다.

## 시뮬레이션 task

| task config | 로봇 | 내용 |
|---|---|---|
| `robomimic_square` · `square_ph50` · `square_scripted` | Panda | 사각 너트를 봉에 끼우기 (robosuite `NutAssemblySquare`) |
| `robomimic_lift` · `robomimic_transport` | Panda | robomimic 기본 벤치마크 |
| `franka_microwave` | Panda / NERO | 전자레인지 문 열기 → 물건 넣기 → 문 닫기 → start 버튼 |
| `nero_drawer` | NERO | 양팔 2단 서랍 |
| `door_cabinet` | Panda | 문 열기 → 닫기 → 복귀 |

스크립트 시연 정책이 있는 task: `envs/square_expert.py` · `envs/microwave_expert.py` ·
`envs/drawer_expert.py`. task config 의 `sim.expert` 가 어느 것을 쓸지 정한다.

## 사용법

### 1. 시연 수집 → 학습 → 평가

```bash
# 스크립트 시연 (hdf5).  성공한 것만 남는다
python -m manibot.scripts.collect_scripted task=square_scripted n_demos=50 max_steps=600

# 학습용 zarr 로
python -m manibot.scripts.convert_hdf5_to_zarr task=square_scripted

# 학습.  긴 학습은 세션에서 떼어낸다 (setsid nohup ... & disown)
python -m manibot.scripts.train task=square_scripted policy=diffusion \
    train.steps=50000 wandb.project=<이름> wandb.entity=null

# 평가 — 성공률.  view=true 면 창을 띄운다 (view_fps=0 이면 최고 속도)
python -m manibot.scripts.eval task=square_scripted policy=diffusion \
    checkpoint_path=outputs/.../checkpoints/step_0000050000 \
    val.eval_n_episodes=50 view=true view_fps=20
```

### 2. 배포 데이터 수집 (사람이 개입)

정책을 배포해 굴리다 사람이 `i` 로 개입을 트리거하면, 교정 행동은 task 의 스크립트
전문가가 현재 상태에서 이어서 낸다. 저장은 **실물과 같은 LeRobotDataset** 이다.

```bash
python -m manibot.scripts.collect_intervention task=square_ph50 \
    checkpoint_path=outputs/.../checkpoints/step_0000050000 \
    repo_id=<user>/<name> root=data/<name>

# 학습용 zarr 로.  ⚠ --native 를 빼면 84x84 를 240x320 으로 키운다
python src/manibot/scripts/convert.py --local-dir data/<name> --native -o data/<name>_zarr
```

창에 포커스를 두고 조작한다 (Wayland 에서는 전역 키 캡처가 안 된다).

```
i        개입 on/off          →/n  이 에피소드 끝 (저장)
←/r      버리고 다시           q/ESC 정상 종료 (⚠ Ctrl+C 는 파일을 깨뜨린다)
```

라운드 크기는 에피소드가 아니라 **프레임**으로 정한다 — `round_size_ratio`(기본 1.5)가
base policy 가 학습한 시연의 프레임 수에 곱해진다. `n_episodes` 는 안전 상한이다.
중간에 죽으면 **같은 명령을 다시** 주면 이어받는다(`save_every` 마다 finalize 한다).

`eval` 에도 같은 수집기를 붙일 수 있다 — 개입이 없으므로 전부 `action_mode=0` 이다.

```bash
python -m manibot.scripts.eval ... collect.enable=true collect.repo_id=... collect.root=...
```

### 3. 결과가 쌓이는 곳

```
outputs/<task>/<policy>/<학습 세션>/
    checkpoints/step_XXXXXXXX
    eval/step_XXXXXXXX_<시각>/eval_result.json · eval.log · videos/
```
평가 결과는 **평가한 체크포인트 아래**에 쌓인다 — 같은 체크포인트를 여러 번 재면 나란히
남고, 어느 성적인지 경로만 봐도 안다. 배포 수집은 데이터를 `root` 에, 로그를 그 옆
`<root>_collect.log` 에 둔다. 세 경우 모두 빈 세션 폴더를 만들지 않는다.

### 4. 배포·평가 규약

시뮬과 실물이 갈라지지 않게 세 경로(`eval` · `train` 의 온라인 평가 · `collect_intervention`)가
같은 코드를 쓴다.

- **행동 청크는 시각에 앵커한다** (`rollout/merger.py`). `chunk[k]` 는 `t_obs+k` 의 예측이고
  `merger.get_action(step)` 으로 꺼낸다. 추론이 늦게 끝나면 청크 앞부분이 버려질 뿐,
  실행되는 행동은 언제나 "지금"의 예측이다. 청크를 그대로 재생하면 지연만큼 밀린다.
- **자르기는 `rollout/policy_server.make_predict_fn` 한 곳**에서 한다. 정책이 선언한 창
  (`get_observation_indices` / `get_action_indices`)에서 잘라낼 양을 유도하므로, 호출하는
  쪽은 "청크의 첫 칸이 지금"이라는 규약 하나만 안다.
- **라벨은 실물 규약** (`utils/intervention_labels.py`): `action_mode` 0=rollout · 1=intervention.
  APO 의 pre-intervention(=2)은 학습 직전에 `relabel_preintv(mode, k)` 로 만든다.

## 자산 출처

- 전자레인지 `envs/assets/microwave/` — [RoboCasa](https://github.com/robocasa/robocasa)
  의 `Microwave052` (MIT License, © the RoboCasa Team).
- NERO `robots/nero/assets/` — WeGo Robotics 제공 원본 URDF·STL 에서
  `robots/nero/build_assets.py` 가 생성한다.
