"""개입 시점을 사람이 정하는 키보드 트리거.

**행동은 내지 않는다** — 언제 개입할지만 정한다. 무엇을 할지는 전문가(`envs/*_expert.py`)가
낸다. 그래서 task 가 바뀌어도 이 파일은 그대로다.

**LeRobot 의 키보드 계층을 그대로 쓴다** (`lerobot.utils.keyboard_input`). 다시 구현하지 않는
이유는 그쪽이 이미 백엔드 선택(X11 은 pynput 전역 리스너 · Wayland/헤드리스는 TTY cbreak)과
터미널 복원(atexit)을 처리하기 때문이다. 우리가 더한 건 **개입 토글 키 하나**뿐이다.

키 (LeRobot 규약 + i):
    i          개입 on/off 토글          ← 우리 추가.  TeleopEvents.IS_INTERVENTION
    → 또는 n   에피소드 조기 종료          exit_early
    ← 또는 r   다시 찍기                  rerecord_episode
    s          저장                      save   (lerobot/rollout/configs.py 의 save_key 기본값)
    ESC 또는 q  전체 중단                  stop_recording

문자 키를 같이 두는 것도 LeRobot 관례다 — 느린 SSH/VNC 에서는 방향키의 이스케이프 시퀀스가
쪼개지거나 가로채여 안 먹는 일이 있다.
"""

import logging

from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.keyboard_input import apply_recording_control, create_key_listener

logger = logging.getLogger(__name__)

HELP = "i=개입 토글 · →/n=다음 · ←/r=다시 · s=저장 · ESC/q=중단"


class KeyboardTrigger:
    """개입 토글과 녹화 제어 플래그를 들고 있는 리스너.

    `events` 는 LeRobot 의 recording 이벤트 dict 와 같은 모양이라, 그쪽 루프에 그대로 넣을 수
    있다. 개입 상태만 `TeleopEvents.IS_INTERVENTION` 키로 하나 더 있다.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "save": False,
            TeleopEvents.IS_INTERVENTION.value: False,
        }
        self._listener = create_key_listener(self._on_key, controls_help=HELP)
        if self._listener is None:
            logger.warning("키보드 입력을 못 잡는다 — 개입 없이 정책만 돌게 된다. %s", HELP)

    # ── 리스너 ────────────────────────────────────────────────────────────
    def _on_key(self, name):
        key = name.lower()
        if key == "i":
            self.toggle()
        elif key in ("right", "n"):
            apply_recording_control("right", self.events)
        elif key in ("left", "r"):
            apply_recording_control("left", self.events)
        elif key in ("esc", "q"):
            apply_recording_control("esc", self.events)
        elif key == "s":
            self.events["save"] = True
            if self.verbose:
                print("s: 저장")
        # 그 밖(↑↓ 등)은 무시한다 — LeRobot 도 같다

    def feed(self, name):
        """창(cv2) 등 **다른 입력원**에서 받은 키를 같은 경로로 넣는다.

        Wayland 에서는 pynput 전역 캡처가 안 돼(LeRobot `pynput_can_capture`) 리스너가
        TTY 로 폴백한다 — 그러면 **터미널에 포커스가 있어야만** 키가 먹어서 영상을 보며
        누를 수가 없다. 영상 창이 직접 키를 받아 이리로 넘기면 그 제약이 사라진다.
        """
        if name:
            self._on_key(name)

    # ── 상태 ──────────────────────────────────────────────────────────────
    @property
    def intervening(self):
        return bool(self.events[TeleopEvents.IS_INTERVENTION.value])

    def toggle(self, on=None):
        v = (not self.intervening) if on is None else bool(on)
        self.events[TeleopEvents.IS_INTERVENTION.value] = v
        if self.verbose:
            print(f"i: 개입 {'시작' if v else '해제'}")

    def reset_episode(self):
        """에피소드 경계에서 부른다. `stop_recording` 은 남긴다 — 전체 중단이라서."""
        self.events["exit_early"] = False
        self.events["rerecord_episode"] = False
        self.events["save"] = False
        self.events[TeleopEvents.IS_INTERVENTION.value] = False

    def stop(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
