"""Middle-layer between ControlLoop (real-time) and PolicyServer (GPU).

For every iteration:
    1. Block on ObsProvider.get() — wait for the next fresh observation
    2. Stack obs_horizon frames (duplicating on cold start)
    3. Ship (t_obs, stacked_obs) to PolicyServer
    4. Block on chunk_queue.get() — wait for the resulting chunk
    5. merger.submit(true_anchor, chunk) under merger_lock, where
       `true_anchor = t_obs - (obs_horizon - 1)` is the timestep predicted by
       chunk[0]. (See "Anchor correction" below.)
    6. Loop immediately

The immediate re-loop is what gives us "continuous inference": as soon
as a chunk lands we capture the latest obs and start the next inference,
without any executed-since-chunk gating.  This is the in-process
analogue of lerobot's RobotClient.receive_actions thread.

Anchor correction
-----------------
A chunk-based policy (Diffusion, ACT, ...) trained with obs_horizon h returns
a full pred_horizon prediction where the first (h-1) entries cover past
timesteps inside the obs window:

    chunk[0]        = action at time (t_obs - (h - 1))   ← past
    chunk[h - 1]    = action at time t_obs                ← obs capture
    chunk[k]        = action at time (t_obs - (h - 1) + k)

Not every policy follows this. A standard Diffusion Policy aligns chunk[0]
with t_obs itself (anchor_offset=0), while ACT-style policies predict from
(t_obs - (h-1)) onward (anchor_offset=h-1). `anchor_offset` therefore has no
default — the caller states the convention.

The policy's own select_action() slices [h-1 : h-1 + action_horizon] before
use.  In async mode we keep the whole chunk and just submit with the
correct anchor — the merger naturally ignores anything <= current_step
(the executed-during-latency portion is never queried), and overlapping
chunks line up on the same timestep grid for temporal ensemble.
"""

from __future__ import annotations

import collections
import queue
import threading
from typing import Callable


class ClientManager(threading.Thread):
    """Buffers observations and ships them to PolicyServer. Knows no policy.

    The buffer holds raw observation dicts, not stacked tensors — turning them
    into a batch is the predict_fn's job (see rollout.policy_server). That keeps
    this thread free of any policy interface.
    """

    def __init__(
        self,
        obs_provider,
        policy_obs_queue: "queue.Queue",
        policy_chunk_queue: "queue.Queue",
        merger,
        merger_lock: threading.Lock,
        obs_horizon: int,
        anchor_offset: int,
        stop_event: threading.Event,
        on_chunk_submitted: Callable[[int, float], None] | None = None,
    ) -> None:
        super().__init__(daemon=True, name="ClientManager")
        self.obs_provider = obs_provider
        self.policy_obs_queue = policy_obs_queue
        self.policy_chunk_queue = policy_chunk_queue
        self.merger = merger
        self.merger_lock = merger_lock
        self.stop_event = stop_event
        self.on_chunk_submitted = on_chunk_submitted

        self.obs_horizon = obs_horizon
        # No default on purpose: getting this wrong shifts every action along the
        # time axis without raising anything, so the caller has to state which
        # convention its policy follows (see the module docstring).
        self._anchor_offset = anchor_offset
        self._obs_buffer: collections.deque = collections.deque(maxlen=obs_horizon)
        self._buffer_lock = threading.Lock()

    def reset_buffer(self) -> None:
        with self._buffer_lock:
            self._obs_buffer.clear()

    def _stack(self, obs) -> list:
        """Push obs into buffer (cold-start by duplicating first frame) and return it."""
        with self._buffer_lock:
            self._obs_buffer.append(obs)
            while len(self._obs_buffer) < self.obs_horizon:
                self._obs_buffer.appendleft(self._obs_buffer[0])
            return list(self._obs_buffer)

    def run(self) -> None:
        while not self.stop_event.is_set():
            item = self.obs_provider.get(timeout=0.5)
            if item is None:
                continue
            t_obs, obs = item

            obs_history = self._stack(obs)

            try:
                self.policy_obs_queue.put((t_obs, obs_history), timeout=0.1)
            except queue.Full:
                continue

            try:
                t_obs_recv, chunk, latency_ms = self.policy_chunk_queue.get(timeout=5.0)
            except queue.Empty:
                continue

            # Anchor correction: submit with the timestep that chunk[0] predicts,
            # so merger.get_action(step) returns the action whose timestep is
            # `step`. Past entries (timestep < step) are simply never queried —
            # the lerobot-style "drop executed, combine from current" behavior
            # falls out for free.
            true_anchor = t_obs_recv - self._anchor_offset
            with self.merger_lock:
                self.merger.submit(true_anchor, chunk)

            if self.on_chunk_submitted is not None:
                # Pass the obs-capture step (not the corrected anchor) so the
                # debug log shows the intuitive "when was this obs taken?" value.
                self.on_chunk_submitted(t_obs_recv, latency_ms)
