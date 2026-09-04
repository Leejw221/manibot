"""Inference worker thread.

Pure inference loop:
    pull (t_obs, obs_history) from input queue
    -> predict_fn(obs_history)
    -> push (t_obs, chunk_np, latency_ms) to output queue

The thread takes a plain ``predict_fn`` rather than a policy object, so it does
not know which policy it serves. Anything that maps an observation history to a
``(pred_horizon, action_dim)`` array works — a manibot policy via
:func:`make_predict_fn`, or any other callable. Keeping the policy interface out
of the thread is what stops a second copy of this file from appearing the next
time a policy with a different interface shows up.

This is the in-process analogue of lerobot's gRPC ``PolicyServer``
(SendObservations + GetActions). Same role, no network — only the queue
boundary would need to change to make it remote.
"""

from __future__ import annotations

import queue
import threading
import time

import torch


def make_predict_fn(policy, cfg, device):
    """manibot 정책(BasePolicy)을 PolicyServer 가 받는 predict_fn 으로 감싼다.

    obs_history 는 raw obs dict 의 리스트(길이 obs_horizon)다. 텐서로 쌓고 device 로
    옮기고 정규화하는 일은 전부 여기서 한다 — ClientManager 는 버퍼링만 하고 정책을
    모른다.
    """
    input_keys = list(cfg.task.image_keys) + [cfg.task.state_key]
    action_key = cfg.task.action_key

    def predict_fn(obs_history):
        batch = {
            k: torch.stack([torch.as_tensor(o[k]) for o in obs_history], dim=0)
            .unsqueeze(0)
            .to(device, non_blocking=True)
            for k in input_keys
        }
        with torch.inference_mode():
            batch = policy.normalize_inputs(batch)
            actions = policy.generate_actions(batch)
            actions = policy.unnormalize_outputs({action_key: actions})[action_key]
        return actions.squeeze(0).cpu().numpy()

    return predict_fn


class PolicyServer(threading.Thread):
    def __init__(
        self,
        predict_fn,
        obs_queue: "queue.Queue",
        chunk_queue: "queue.Queue",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name="PolicyServer")
        self.predict_fn = predict_fn
        self.obs_queue = obs_queue
        self.chunk_queue = chunk_queue
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                t_obs, obs_history = self.obs_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            t_start = time.time()
            chunk_np = self.predict_fn(obs_history)
            latency_ms = (time.time() - t_start) * 1000.0

            try:
                self.chunk_queue.put((t_obs, chunk_np, latency_ms), timeout=0.1)
            except queue.Full:
                # Receiver lagged — drop the chunk. In continuous mode this only
                # happens if ClientManager is stuck; the next chunk supersedes it.
                pass
