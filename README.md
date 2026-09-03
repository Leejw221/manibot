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

## 출처와 라이선스

이 저장소는 다음 코드를 이식해 구성한다.

- **VITA / FLARE** (`manipulation_pipeline`) — Copyright © 2024 The Regents of the
  University of California, Davis campus. 비영리 교육·연구 기관의 비상업적 사용만
  허용된다. 원 라이선스 전문은 이식과 함께 `LICENSE-VITA` 로 포함한다.
- **manipulation_simulator** (`mani_sim`) — 개인 저장소.

이식이 끝나기 전까지 이 저장소는 공개하지 않는다.
