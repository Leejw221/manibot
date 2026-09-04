"""비동기 추론 배선 검증 — ObsProvider -> ClientManager -> PolicyServer -> Merger.

정책 없이 predict_fn 만으로 돈다는 것 자체가 이 병합의 요점이다. 그리고
anchor_offset 이 chunk 를 시간축 어디에 놓는지를 확인한다 — 여기가 틀리면
행동이 통째로 밀리는데 예외가 나지 않는다.
"""
import queue
import threading
import time

import numpy as np

from manibot.rollout import ClientManager, ObsProvider, PolicyServer, make_merger

OBS_HORIZON = 2
PRED_HORIZON = 4
ACTION_DIM = 3


def run_once(anchor_offset):
    """t_obs=10 에서 관측 하나를 넣고, chunk 가 어느 스텝에 놓이는지 본다."""
    # chunk[k] 의 모든 성분을 k 로 채운다 — 어느 원소가 어느 스텝에 놓였는지 읽으려고.
    def predict_fn(obs_history):
        assert len(obs_history) == OBS_HORIZON, f"obs_history 길이 {len(obs_history)}"
        assert isinstance(obs_history[0], dict), "raw obs dict 가 아니다"
        return np.stack([np.full(ACTION_DIM, k, dtype=np.float32) for k in range(PRED_HORIZON)])

    obs_provider = ObsProvider()
    obs_q: queue.Queue = queue.Queue(maxsize=1)
    chunk_q: queue.Queue = queue.Queue(maxsize=4)
    merger = make_merger("overwrite")
    lock = threading.Lock()
    stop = threading.Event()

    server = PolicyServer(predict_fn, obs_q, chunk_q, stop)
    client = ClientManager(obs_provider, obs_q, chunk_q, merger, lock,
                           obs_horizon=OBS_HORIZON, anchor_offset=anchor_offset,
                           stop_event=stop)
    server.start(); client.start()

    obs_provider.put(10, {"observation.state": np.zeros(3, dtype=np.float32)})
    deadline = time.time() + 5.0
    while time.time() < deadline:
        with lock:
            if merger.get_action(10) is not None:
                break
        time.sleep(0.01)

    with lock:
        found = {t: merger.get_action(t) for t in range(8, 15)}
    stop.set(); obs_provider.stop()
    server.join(timeout=2); client.join(timeout=2)
    return {t: (None if v is None else int(v[0])) for t, v in found.items()}


# anchor_offset=0 : chunk[0] 이 t_obs(10) 자체 -> 10,11,12,13 에 0,1,2,3
got = run_once(anchor_offset=0)
assert got[10] == 0 and got[13] == 3 and got[9] is None, got
print(f"  ✓ anchor_offset=0  (표준 Diffusion Policy)   step->chunk index {got}")

# anchor_offset=1 : chunk[0] 이 t_obs-1(9) -> 9,10,11,12 에 0,1,2,3
got = run_once(anchor_offset=1)
assert got[9] == 0 and got[10] == 1 and got[12] == 3 and got[13] is None, got
print(f"  ✓ anchor_offset=1  (ACT 류, obs_horizon-1)   step->chunk index {got}")
print("  ✓ PolicyServer 가 정책 없이 predict_fn 만으로 동작")
print("  ✓ ClientManager 가 raw obs dict 를 그대로 버퍼링 (콜드 스타트 복제 포함)")
