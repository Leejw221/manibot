"""Action chunk mergers.

A Merger receives action chunks via `submit(t_obs, chunk)` and returns the
action for a requested global timestep via `get_action(t)`. Timestep-aware:
chunk[k] of a submission with t_obs=N is the prediction for time (N+k).

Three strategies:
    Overwrite           — keep only the latest chunk (no ensembling)
    TemporalEnsembler   — exponential-weighted average over overlapping chunks
                          (ACT-style; coeff controls recency bias)
    OffsetBlend         — latest chunk only, but the position gap at each switch
                          is decayed away instead of jumped (mode-preserving)
"""

from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from .timed_chunk import TimedChunk


class Merger(ABC):
    @abstractmethod
    def submit(self, t_obs: int, chunk: np.ndarray) -> None: ...

    @abstractmethod
    def get_action(self, t: int) -> np.ndarray | None: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def remaining_after(self, t: int) -> int:
        """Number of future timesteps (>= t) for which a prediction exists.
        Used for debug logging and request-scheduling heuristics."""


class Overwrite(Merger):
    """Latest-chunk-only. Old chunks are discarded on each submit."""

    def __init__(self):
        self._latest: TimedChunk | None = None

    def submit(self, t_obs: int, chunk: np.ndarray) -> None:
        self._latest = TimedChunk(t_obs=t_obs, chunk=np.asarray(chunk))

    def get_action(self, t: int) -> np.ndarray | None:
        if self._latest is None:
            return None
        return self._latest.get(t)

    def clear(self) -> None:
        self._latest = None

    def remaining_after(self, t: int) -> int:
        if self._latest is None:
            return 0
        return max(self._latest.last_t() - t + 1, 0)


class TemporalEnsembler(Merger):
    """ACT-style exponential-weighted average across overlapping chunks.

    For each timestep t we collect every stored chunk that predicts t and take
    a weighted average. Weights follow weight[age] = exp(-coeff * age), where
    age=0 is the most-recently-submitted chunk.

        coeff > 0   → recent-dominant
        coeff = 0   → uniform average
        coeff < 0   → old-dominant
    """

    def __init__(self, coeff: float = 0.01, max_chunks: int = 16):
        self.coeff = float(coeff)
        self._chunks: deque[TimedChunk] = deque(maxlen=max_chunks)

    def submit(self, t_obs: int, chunk: np.ndarray) -> None:
        self._chunks.append(TimedChunk(t_obs=t_obs, chunk=np.asarray(chunk)))

    def get_action(self, t: int) -> np.ndarray | None:
        preds: list[np.ndarray] = []
        ages: list[int] = []
        # age = 0 for the newest submission (rightmost in deque)
        for age, tc in enumerate(reversed(self._chunks)):
            a = tc.get(t)
            if a is not None:
                preds.append(a)
                ages.append(age)

        if not preds:
            return None

        weights = np.exp(-self.coeff * np.asarray(ages, dtype=np.float64))
        weights /= weights.sum()
        stacked = np.stack(preds, axis=0)
        return (stacked * weights[:, None]).sum(axis=0).astype(stacked.dtype, copy=False)

    def clear(self) -> None:
        self._chunks.clear()

    def remaining_after(self, t: int) -> int:
        if not self._chunks:
            return 0
        last = max(tc.last_t() for tc in self._chunks)
        return max(last - t + 1, 0)


class OffsetBlend(Merger):
    """Run only the latest chunk, but decay the position gap at each switch.

    Ported from the real-robot code (Cashier_policy-dp `flare/inference/merger.py`, itself
    from manipulation_pipeline 09f12be) so that sim and robot execute chunks the same way
    [사용자 2026-10-01]. Actions inside one chunk come from one diffusion sample and share a
    mode; the next chunk need not. At the seam, Overwrite keeps the mode but jumps in
    position, and TemporalEnsembler removes the jump by averaging two modes into an action
    that is neither. This keeps the new chunk's shape from its first step and only carries
    the position difference, fading it out:

        d = current output(t_switch) - new chunk(t_switch)
        action(t) = new chunk(t) + d * decay(t - t_switch)

    decay is 1 - smoothstep over `span` steps, so its slope is zero at both ends.
    span = min(blend_steps, steps since the previous submission): a residual offset would
    otherwise be re-injected at every switch when submissions come faster than blend_steps.

    `ref` picks t_switch: "first_use" (default) — the first step get_action() is asked for
    after the submission, i.e. where the new chunk first reaches the robot; "anchor" — the
    chunk anchor passed to submit(). "first_use" relies on callers asking for the current
    step before any later one.

    `unblended_dims` take the new chunk's value as is (e.g. gripper dims).
    """

    REFS = ("first_use", "anchor")

    def __init__(self, blend_steps: int = 5, ref: str = "first_use", unblended_dims=()):
        if ref not in self.REFS:
            raise ValueError(f"Unknown blend ref: {ref!r}. Available: {self.REFS}")
        self.blend_steps = max(int(blend_steps), 1)
        self.ref = ref
        self.unblended_dims = np.asarray(sorted({int(i) for i in unblended_dims}), dtype=np.int64)
        self._latest: TimedChunk | None = None
        self._pending: tuple[TimedChunk, int] | None = None   # first_use: submitted, not yet asked for
        self._offset: np.ndarray | None = None
        self._t_switch: int | None = None
        self._span: int = self.blend_steps
        self._last_submit_t: int | None = None

    def _decay(self, dt: int) -> float:
        x = min(max(dt / self._span, 0.0), 1.0)
        return 1.0 - (3.0 * x * x - 2.0 * x * x * x)   # 1 - smoothstep

    def _output(self, t: int) -> np.ndarray | None:
        if self._latest is None:
            return None
        a = self._latest.get(t)
        if a is None:
            return None
        if self._offset is None or self._t_switch is None:
            return a
        w = self._decay(t - self._t_switch)
        return a if w <= 0.0 else a + self._offset * w

    def _switch(self, new: TimedChunk, t_ref: int, span: int) -> None:
        # `before` must still decay the previous offset over the previous span,
        # so the new span is installed only afterwards.
        before = self._output(t_ref)
        after = new.get(t_ref)
        if before is None or after is None:
            self._offset = None
            self._t_switch = None
        else:
            self._offset = before - after
            if self.unblended_dims.size:
                self._offset[self.unblended_dims] = 0.0
            self._t_switch = t_ref
            self._span = span
        self._latest = new

    def submit(self, t_obs: int, chunk: np.ndarray) -> None:
        new = TimedChunk(t_obs=t_obs, chunk=np.asarray(chunk))
        gap = (t_obs - self._last_submit_t) if self._last_submit_t is not None else None
        span = self.blend_steps if gap is None else min(self.blend_steps, max(gap, 1))
        self._last_submit_t = t_obs
        if self.ref == "anchor":
            self._switch(new, t_obs, span)
        else:
            self._pending = (new, span)        # a later submission before any query supersedes this one

    def get_action(self, t: int) -> np.ndarray | None:
        if self._pending is not None:
            (new, span), self._pending = self._pending, None
            self._switch(new, t, span)
        return self._output(t)

    def clear(self) -> None:
        self._latest = None
        self._pending = None
        self._offset = None
        self._t_switch = None
        self._span = self.blend_steps
        self._last_submit_t = None

    def remaining_after(self, t: int) -> int:
        chunk = self._pending[0] if self._pending is not None else self._latest
        if chunk is None:
            return 0
        return max(chunk.last_t() - t + 1, 0)


def make_merger(name: str, te_coeff: float = 0.01, blend_steps: int = 5,
                blend_ref: str = "first_use", unblended_dims=()) -> Merger:
    """Factory used by the eval CLI.

    `name`:
        "overwrite"          — Overwrite
        "temporal_ensemble"  — TemporalEnsembler(coeff=te_coeff)
        "offset_blend"       — OffsetBlend(blend_steps, ref=blend_ref, unblended_dims)
    """
    if name == "overwrite":
        return Overwrite()
    if name == "temporal_ensemble":
        return TemporalEnsembler(coeff=te_coeff)
    if name == "offset_blend":
        return OffsetBlend(blend_steps=blend_steps, ref=blend_ref, unblended_dims=unblended_dims)
    raise ValueError(
        f"Unknown merger: {name!r}. "
        f"Available: 'overwrite', 'temporal_ensemble', 'offset_blend'."
    )
