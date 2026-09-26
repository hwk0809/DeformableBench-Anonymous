"""Chunked action buffer with overlap aggregation + RTC support.

Used by eval_isaac_render.py (sim) and eval_lerobot_real.py (real robot)
to turn a Pi0.5 / LeRobot `predict_action_chunk` output into a step-by-step
action stream with new-chunk / old-chunk blending, and optionally feed Real-Time
Chunking (RTC) guidance back into the next inference call.

Design mirrors Evo-RL's async_inference.robot_client aggregator, simplified for
synchronous/single-process use.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch


AGGREGATE_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "latest_only": lambda old, new: new,
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
    # "temporal_ensemble" handled inline in ingest()
}
_SPECIAL_AGG = {"temporal_ensemble"}


@dataclass
class _TimedAction:
    timestep: int
    action: np.ndarray  # shape (action_dim,)
    ensemble_count: int = 1


class ChunkedActionBuffer:
    """FIFO buffer of per-timestep actions, with overlap aggregation on ingest.

    Typical usage (synchronous):

        buffer = ChunkedActionBuffer(chunk_size=50, aggregate_fn="latest_only")
        step_idx = 0
        while not done:
            if buffer.should_request_new_chunk():
                start_ts = buffer.next_timestep_to_request()
                rtc_kwargs = buffer.get_rtc_kwargs() if rtc_enabled else {}
                t0 = time.perf_counter()
                raw_chunk = policy.predict_action_chunk(batch, **rtc_kwargs)  # (1, K, D)
                buffer.record_inference_time(time.perf_counter() - t0)
                post_chunk = postprocess_chunk(raw_chunk, postprocessor)      # (K, D) np
                buffer.ingest(post_chunk, start_timestep=start_ts, raw_tensor=raw_chunk)
            action, ts = buffer.pop_next()
            robot.send_action(action)
            step_idx += 1
    """

    def __init__(
        self,
        chunk_size: int,
        n_action_steps: Optional[int] = None,
        refill_threshold: float = 0.5,
        aggregate_fn: str = "latest_only",
        gripper_indices: Sequence[int] = (6, 13),
        ensemble_coeff: float = 0.01,
        ensemble_max_steps: int = 200,
        fps: int = 30,
        rtc_execution_horizon: int = 10,
    ):
        if aggregate_fn not in AGGREGATE_FUNCTIONS and aggregate_fn not in _SPECIAL_AGG:
            raise ValueError(
                f"Unknown aggregate_fn: {aggregate_fn!r}. "
                f"Supported: {sorted(set(AGGREGATE_FUNCTIONS) | _SPECIAL_AGG)}"
            )
        if not 0.0 <= refill_threshold <= 1.0:
            raise ValueError(f"refill_threshold must be in [0,1], got {refill_threshold}")
        if n_action_steps is None:
            n_action_steps = chunk_size
        if not 0 < n_action_steps <= chunk_size:
            raise ValueError(
                f"n_action_steps ({n_action_steps}) must be in (0, chunk_size={chunk_size}]"
            )

        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.refill_threshold = refill_threshold
        self.aggregate_fn_name = aggregate_fn
        self.gripper_indices = tuple(int(i) for i in gripper_indices)
        self.ensemble_coeff = ensemble_coeff
        self.fps = fps
        self.rtc_execution_horizon = rtc_execution_horizon

        # Precompute exponential weights for temporal_ensemble
        w = np.array([math.exp(-ensemble_coeff * k) for k in range(ensemble_max_steps)], dtype=np.float64)
        self._ensemble_weights = w
        self._ensemble_cumsum = np.cumsum(w)

        self._queue: collections.deque[_TimedAction] = collections.deque()
        self._last_popped_timestep: int = -1

        # RTC cache
        self._last_raw_chunk: Optional[torch.Tensor] = None  # shape (1, K, D)
        self._last_raw_chunk_start: Optional[int] = None
        self._last_inference_time_s: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self):
        self._queue.clear()
        self._last_popped_timestep = -1
        self._last_raw_chunk = None
        self._last_raw_chunk_start = None
        self._last_inference_time_s = None

    # ------------------------------------------------------------------
    # Consumption-side API
    # ------------------------------------------------------------------
    @property
    def queue_size(self) -> int:
        return len(self._queue)

    def should_request_new_chunk(self) -> bool:
        """True when the queue is below the refill threshold (or empty)."""
        if not self._queue:
            return True
        return self.queue_size / self.n_action_steps <= self.refill_threshold

    def next_timestep_to_request(self) -> int:
        """The timestep the next chunk's first action should correspond to.

        This is the first step that hasn't been popped yet. In steady-state
        with overlap, the new chunk covers this timestep AND beyond, and the
        overlapping portion gets aggregated with the existing queue.
        """
        return self._last_popped_timestep + 1

    def pop_next(self) -> tuple[np.ndarray, int]:
        if not self._queue:
            raise RuntimeError("ChunkedActionBuffer is empty; call ingest() first.")
        ta = self._queue.popleft()
        self._last_popped_timestep = ta.timestep
        return ta.action.copy(), ta.timestep

    # ------------------------------------------------------------------
    # Production-side API
    # ------------------------------------------------------------------
    def record_inference_time(self, dt_seconds: float) -> None:
        """Call right after `predict_action_chunk` returns. Used by RTC to
        estimate how many steps will be consumed during the next inference."""
        self._last_inference_time_s = float(dt_seconds)

    def ingest(
        self,
        new_chunk: np.ndarray,
        start_timestep: int,
        raw_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """Merge a newly-produced chunk with the existing queue.

        Args:
            new_chunk: (chunk_len, action_dim), post-processed (physical units).
                Only the first `n_action_steps` rows are used.
            start_timestep: global timestep index of new_chunk[0].
            raw_tensor: optional (1, K, D) pre-postprocess tensor cached for RTC.
        """
        if new_chunk.ndim != 2:
            raise ValueError(f"new_chunk must be 2D (T, D), got shape {new_chunk.shape}")
        usable_len = min(new_chunk.shape[0], self.n_action_steps)
        new_chunk = new_chunk[:usable_len].astype(np.float32, copy=False)

        # Build a dict of existing future actions keyed by timestep (drop already-popped).
        future: dict[int, _TimedAction] = {}
        for ta in self._queue:
            if ta.timestep > self._last_popped_timestep:
                future[ta.timestep] = ta

        for k in range(usable_len):
            ts = start_timestep + k
            if ts <= self._last_popped_timestep:
                continue  # stale: already executed

            new_a = new_chunk[k].astype(np.float32, copy=False)

            if ts not in future:
                future[ts] = _TimedAction(timestep=ts, action=new_a.copy(), ensemble_count=1)
                continue

            # Overlap: aggregate old + new
            old_ta = future[ts]
            old_a = old_ta.action

            if self.aggregate_fn_name == "temporal_ensemble":
                k_count = old_ta.ensemble_count
                k_clamped = min(k_count, len(self._ensemble_weights) - 1)
                if k_clamped < 1:
                    blended = new_a.copy()
                else:
                    w_k = self._ensemble_weights[k_clamped]
                    cs_prev = self._ensemble_cumsum[k_clamped - 1]
                    cs_new = self._ensemble_cumsum[k_clamped]
                    blended = ((old_a * cs_prev + new_a * w_k) / cs_new).astype(np.float32)
                new_count = k_count + 1
            else:
                fn = AGGREGATE_FUNCTIONS[self.aggregate_fn_name]
                blended = fn(old_a, new_a).astype(np.float32)
                new_count = old_ta.ensemble_count

            # Gripper dims: always take the latest value (no blending of discrete state).
            for gi in self.gripper_indices:
                if 0 <= gi < blended.shape[-1]:
                    blended[gi] = new_a[gi]

            future[ts] = _TimedAction(timestep=ts, action=blended, ensemble_count=new_count)

        self._queue = collections.deque(sorted(future.values(), key=lambda a: a.timestep))

        if raw_tensor is not None:
            # Keep detached CPU tensor to avoid GPU ref buildup; RTC will move to device if needed.
            self._last_raw_chunk = raw_tensor.detach()
            self._last_raw_chunk_start = int(start_timestep)

    # ------------------------------------------------------------------
    # RTC
    # ------------------------------------------------------------------
    def get_rtc_kwargs(self) -> dict:
        """Compute RTC kwargs for the next predict_action_chunk call.

        Returns an empty dict if there is no cached previous chunk, or if the
        previous chunk has been fully consumed (no leftover to align against).
        """
        if self._last_raw_chunk is None or self._last_raw_chunk_start is None:
            return {}

        raw = self._last_raw_chunk  # (1, K_raw, D)
        k_raw = raw.shape[1]
        usable = min(k_raw, self.n_action_steps)

        # "Current timestep" = first step of the incoming new chunk
        current_ts = self.next_timestep_to_request()
        consumed = max(current_ts - self._last_raw_chunk_start, 0)
        consumed = min(consumed, usable)
        tail = usable - consumed
        if tail <= 0:
            return {}

        # Estimate how many steps will be executed during the upcoming inference.
        # Evo-RL fallback: 9 steps (=300ms @ 30Hz) if no measurement yet.
        inference_time_s = self._last_inference_time_s if self._last_inference_time_s is not None else 0.3
        est_latency_steps = max(math.ceil(inference_time_s * self.fps), 1)
        inference_delay = min(est_latency_steps, tail)
        exec_horizon = min(inference_delay + self.rtc_execution_horizon, tail)

        return {
            "prev_chunk_left_over": raw[:, consumed:usable, :],
            "inference_delay": inference_delay,
            "execution_horizon": exec_horizon,
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def setup_rtc_on_policy(
    policy,
    execution_horizon: int = 10,
    max_guidance_weight: float = 10.0,
) -> bool:
    """Enable RTC on a LeRobot policy. Returns True on success, False if unsupported."""
    if not hasattr(policy, "config") or not hasattr(policy.config, "rtc_config"):
        return False
    try:
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
    except ImportError:
        return False
    policy.config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=execution_horizon,
        max_guidance_weight=max_guidance_weight,
    )
    if hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()
    return True


def postprocess_chunk(
    raw_chunk: torch.Tensor,
    postprocessor,
) -> np.ndarray:
    """Apply a LeRobot postprocessor step-by-step to a (1, K, D) chunk tensor.

    Returns (K, D) float32 numpy array in physical units.
    """
    if raw_chunk.dim() != 3 or raw_chunk.shape[0] != 1:
        raise ValueError(f"Expected raw_chunk shape (1, K, D), got {tuple(raw_chunk.shape)}")
    k = raw_chunk.shape[1]
    out_rows = []
    for i in range(k):
        single = raw_chunk[:, i, :]  # (1, D)
        out = postprocessor(single)
        out_rows.append(out)
    stacked = torch.stack(out_rows, dim=1).squeeze(0)  # (K, D)
    return stacked.detach().cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
def _run_self_tests() -> None:
    print("=" * 60)
    print("ChunkedActionBuffer self-tests")
    print("=" * 60)

    # ---- Test 1: latest_only baseline, no overlap ----
    print("\n[1] latest_only, sequential no-overlap chunks")
    buf = ChunkedActionBuffer(chunk_size=4, aggregate_fn="latest_only", gripper_indices=())
    c0 = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)
    buf.ingest(c0, start_timestep=0)
    assert buf.queue_size == 4
    for expected_ts in range(4):
        a, ts = buf.pop_next()
        assert ts == expected_ts, f"ts mismatch: {ts} != {expected_ts}"
        assert a[0] == expected_ts
    assert buf.should_request_new_chunk()
    print("  OK")

    # ---- Test 2: weighted_average on overlap ----
    print("\n[2] weighted_average aggregate (0.3*old + 0.7*new)")
    buf = ChunkedActionBuffer(chunk_size=4, aggregate_fn="weighted_average", gripper_indices=())
    buf.ingest(np.full((4, 2), 10.0, dtype=np.float32), start_timestep=0)
    # Pop 1 (ts=0). Queue now [ts=1,2,3], last_popped=0.
    buf.pop_next()
    # Ingest new chunk at ts=1..4, value=20. Overlap: ts=1,2,3.
    buf.ingest(np.full((4, 2), 20.0, dtype=np.float32), start_timestep=1)
    a1, ts1 = buf.pop_next()
    expected = 0.3 * 10.0 + 0.7 * 20.0  # = 17.0
    assert ts1 == 1
    assert abs(a1[0] - expected) < 1e-5, f"{a1[0]} != {expected}"
    print(f"  blended = {a1[0]:.3f} (expected {expected})  OK")

    # ---- Test 3: gripper indices forced to latest ----
    print("\n[3] gripper_indices=[1] forced to new value even under weighted_average")
    buf = ChunkedActionBuffer(chunk_size=4, aggregate_fn="weighted_average", gripper_indices=(1,))
    buf.ingest(np.array([[10.0, 0.0]] * 4, dtype=np.float32), start_timestep=0)
    buf.pop_next()
    buf.ingest(np.array([[20.0, 1.0]] * 4, dtype=np.float32), start_timestep=1)
    a, ts = buf.pop_next()
    assert ts == 1
    assert abs(a[0] - 17.0) < 1e-5, f"dim0 blended {a[0]}"
    assert abs(a[1] - 1.0) < 1e-5, f"gripper should be latest, got {a[1]}"
    print(f"  dim0={a[0]:.3f} (blended), gripper={a[1]:.3f} (latest)  OK")

    # ---- Test 4: stale timesteps dropped ----
    print("\n[4] stale timesteps dropped")
    buf = ChunkedActionBuffer(chunk_size=4, aggregate_fn="latest_only", gripper_indices=())
    buf.ingest(np.arange(8, dtype=np.float32).reshape(4, 2), start_timestep=0)
    buf.pop_next()  # ts=0
    buf.pop_next()  # ts=1, last_popped=1
    # New chunk at ts=0..3: first 2 should be dropped as stale.
    new = np.full((4, 2), 99.0, dtype=np.float32)
    buf.ingest(new, start_timestep=0)
    # Queue should contain ts=2, ts=3 (overlap w/ new: replaced) only.
    timesteps = [ta.timestep for ta in list(buf._queue)]
    assert timesteps == [2, 3], f"expected [2,3], got {timesteps}"
    a, ts = buf.pop_next()
    assert ts == 2 and a[0] == 99.0
    print(f"  queue timesteps={timesteps}, first popped ts={ts} val={a[0]}  OK")

    # ---- Test 5: temporal_ensemble monotonic blending ----
    print("\n[5] temporal_ensemble: multiple predictions for same ts")
    buf = ChunkedActionBuffer(chunk_size=4, aggregate_fn="temporal_ensemble",
                              gripper_indices=(), ensemble_coeff=0.01)
    # 3 chunks all predict ts=0 with increasing value
    buf.ingest(np.full((4, 1), 1.0, dtype=np.float32), start_timestep=0)
    ta0 = buf._queue[0]
    assert ta0.ensemble_count == 1 and ta0.action[0] == 1.0
    buf.ingest(np.full((4, 1), 2.0, dtype=np.float32), start_timestep=0)
    ta1 = buf._queue[0]
    assert ta1.ensemble_count == 2
    # After k=1: blended = (1*cs[0] + 2*w[1]) / cs[1] = (1 + 2*e^-.01)/(1 + e^-.01)
    w = math.exp(-0.01)
    expected1 = (1.0 * 1.0 + 2.0 * w) / (1.0 + w)
    assert abs(ta1.action[0] - expected1) < 1e-5, f"{ta1.action[0]} != {expected1}"
    buf.ingest(np.full((4, 1), 3.0, dtype=np.float32), start_timestep=0)
    ta2 = buf._queue[0]
    assert ta2.ensemble_count == 3
    print(f"  k=1 blended={ta1.action[0]:.5f} (expected {expected1:.5f})")
    print(f"  k=2 blended={ta2.action[0]:.5f} count={ta2.ensemble_count}  OK")

    # ---- Test 6: should_request_new_chunk threshold ----
    print("\n[6] refill_threshold=0.5 triggers at half-consumed")
    buf = ChunkedActionBuffer(chunk_size=4, refill_threshold=0.5, aggregate_fn="latest_only",
                              gripper_indices=())
    buf.ingest(np.zeros((4, 2), dtype=np.float32), start_timestep=0)
    assert not buf.should_request_new_chunk(), "queue full, should not refill"
    buf.pop_next()  # 3 remain (0.75)
    assert not buf.should_request_new_chunk(), "3/4=0.75 > 0.5"
    buf.pop_next()  # 2 remain (0.5)
    assert buf.should_request_new_chunk(), "2/4=0.5, should refill"
    print("  OK")

    # ---- Test 7: RTC kwargs ----
    print("\n[7] get_rtc_kwargs from cached raw tensor")
    buf = ChunkedActionBuffer(chunk_size=4, aggregate_fn="latest_only",
                              gripper_indices=(), fps=10, rtc_execution_horizon=2)
    raw = torch.arange(4 * 2, dtype=torch.float32).reshape(1, 4, 2)
    post = raw.squeeze(0).numpy()
    buf.ingest(post, start_timestep=0, raw_tensor=raw)
    buf.record_inference_time(0.2)  # 0.2s @ 10fps = 2 steps
    buf.pop_next()  # ts=0, last_popped=0
    buf.pop_next()  # ts=1, last_popped=1
    # next_timestep_to_request=2, consumed=2, tail=2
    kwargs = buf.get_rtc_kwargs()
    assert "prev_chunk_left_over" in kwargs
    leftover = kwargs["prev_chunk_left_over"]
    assert leftover.shape == (1, 2, 2), f"leftover shape {leftover.shape}"
    # leftover[0,0,:] should be raw[0,2,:] = [4, 5]
    assert torch.allclose(leftover[0, 0], torch.tensor([4.0, 5.0]))
    assert kwargs["inference_delay"] == 2  # 0.2s * 10fps = 2
    assert kwargs["execution_horizon"] == 2  # min(2+2, 2) = 2 (clamped by tail)
    print(f"  leftover shape={leftover.shape}, inference_delay={kwargs['inference_delay']}, "
          f"execution_horizon={kwargs['execution_horizon']}  OK")

    # ---- Test 8: RTC empty after full consumption ----
    print("\n[8] get_rtc_kwargs returns {} when previous chunk fully consumed")
    buf.pop_next()  # ts=2
    buf.pop_next()  # ts=3, last_popped=3, next=4, consumed=4, tail=0
    kwargs2 = buf.get_rtc_kwargs()
    assert kwargs2 == {}, f"expected empty dict, got {kwargs2}"
    print("  OK")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    _run_self_tests()
