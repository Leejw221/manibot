"""시뮬 롤아웃을 실시간으로 띄우는 창 — 수집과 평가가 **같은 창**을 쓴다.

따로 만들면 화면·키 처리가 갈라진다. `collect_intervention`(개입 수집)과 `eval`(평가)이
이 모듈 하나를 쓴다.

⚠ 키 코드에 **`& 0xFF` 를 씌우면 안 된다.** Linux 에서 cv2 는 방향키를 X11 keysym 으로
주는데(왼쪽 65361) 하위 바이트만 취하면 81 = 'Q' 가 되어 왼쪽 화살표가 종료 명령이 된다.
그래서 전체 코드를 그대로 보고 표로 푼다. 빌드마다 코드가 달라 세 계열을 모두 넣는다.
"""

import time

import cv2
import numpy as np

CV_KEYS = {
    27: "esc", 13: "enter", 32: "space",
    65361: "left", 65362: "up", 65363: "right", 65364: "down",          # X11 keysym
    2424832: "left", 2490368: "up", 2555904: "right", 2621440: "down",  # 일부 Windows/Qt 빌드
    81: "left", 82: "up", 83: "right", 84: "down",                      # 하위바이트만 오는 빌드
}


class SimViewer:
    """robosuite 카메라를 나란히 띄우고 상태를 겹쳐 그린다.

    `fps > 0` 이면 사람이 볼 수 있게 실시간으로 늦춘다(전역 시계 기준이라 프레임이 밀리지
    않는다). 반환값은 눌린 키 이름 — 쓰는 쪽이 해석한다.
    """

    def __init__(self, cams, res=384, fps=0.0, title="manibot"):
        self.cams = list(cams)
        self.res, self.fps, self.title = res, float(fps), title
        self._t_next = time.perf_counter()

    def show(self, raw, text, highlight=False):
        tiles = [raw.sim.render(width=self.res, height=self.res, camera_name=c)[::-1]
                 for c in self.cams]
        im = np.concatenate(tiles, axis=1)[:, :, ::-1].copy()          # RGB -> BGR
        cv2.rectangle(im, (0, 0), (im.shape[1], 30),
                      (0, 0, 160) if highlight else (40, 40, 40), -1)
        cv2.putText(im, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(self.title, im)
        k = cv2.waitKey(1)
        name = None
        if k >= 0:
            name = CV_KEYS.get(k) or (chr(k) if 32 < k < 127 else None)
        return name

    def pace(self):
        """실시간 재생. show() 뒤 env.step() 전후 어디서 불러도 되지만 한 스텝에 한 번만."""
        if self.fps <= 0:
            return
        self._t_next = max(self._t_next, time.perf_counter()) + 1.0 / self.fps
        time.sleep(max(0.0, self._t_next - time.perf_counter()))

    def reset_clock(self):
        self._t_next = time.perf_counter()

    def close(self):
        cv2.destroyAllWindows()
