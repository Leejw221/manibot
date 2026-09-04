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

[LeRobot](https://github.com/huggingface/lerobot) 의 구조와 정책 구현을 따르고, 기존 저장소
(`manipulation_pipeline` · `manipulation_simulator`)의 코드를 옮겨 구성했다. 정리 작업은
Claude Code 로 진행했다.

라이선스는 `LICENSE-VITA` 를 따른다.
