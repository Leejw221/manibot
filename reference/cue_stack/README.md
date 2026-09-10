# cue_stack — 색 표식 → 거울상 쌓기 (시뮬)

**출처**: `ljw_workspace/robocasa/scripts/` (2026-08-28 작성, 2026-09-10 이관 후 원본 삭제).
git 이력에도 남아 있다: `git -C ljw_workspace show <커밋>:robocasa/scripts/scene.py`

**왜 남겼나**: 장면 구성에 **실측으로 알아낸 값과 설계 판단**이 박혀 있어 다시 만들면
오래 걸린다. 나머지 분석 스크립트 23개(aliasing·probe·verify·eval)는 다시 쓰면 되므로 지웠다.

| 파일 | 무엇 |
|---|---|
| `scene.py` | 장면 구성. RoboCasa 물체 레지스트리로 못 만드는 블록을 모델 XML 에 직접 주입 |
| `collect_stack2.py` | 수집 스크립트 2판 (궤적 떨림·손목 시야 문제를 고친 판) |
| `armkit.py` | 회전 표현·궤적·팔 제어 공용 도구 |

**scene.py 에 들어 있는 실측·판단**
- `COUNTER_Z = 0.920` — counter 윗면 [실측]
- 원본 free joint(obj1/obj2)가 qpos 맨 끝(109·116 / nq=123)이라 블록 qpos 가 그 뒤에 와서
  패딩이 단순 append 가 된다 [실측 2026-08-28]
- 빨강(목표대)은 **고정 body** — 자유 물체로 두면 쌓는 중 밀려 실패 요인만 는다. 표식도 고정

⚠ **지금 그대로는 안 돈다**: `robocasa`·`robosuite` 를 import 하는데 manibot env 엔 robocasa 가
없다. 다시 구현할 때 읽을 원본이지 실행 파일이 아니다.
