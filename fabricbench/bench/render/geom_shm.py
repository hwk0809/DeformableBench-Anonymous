"""Tiny python<->python geometry snapshot over shared memory.

The benchmark architecture (``docs/local_docs/sim-architecture-proposal.md`` §4/§7.2)
runs the simulation headless in its own process and exposes it to *decoupled,
read-only, low-frequency* consumers. The Omniverse Kit renderer consumes the
heavyweight ``_shm_fabric`` channel; the web viewer is a much lighter python
consumer, so it gets this purpose-built double-duty snapshot instead of the Kit
protocol.

One snapshot = the only things that change per frame:
  - cloth particle positions  ``state.particle_q``  (N, 3) float32
  - rigid body poses          ``state.body_q``      (B, 7) float32  [x,y,z, qx,qy,qz,qw]

Static geometry (meshes, topology) is NOT sent — the consumer builds the scene
``Model`` once and only injects these positions each frame.

Concurrency: a seqlock (odd seq = write in progress) lets the reader detect and
retry torn frames without blocking the writer. For a viewer the occasional retry
is free; the writer (simulation) is never stalled.

Lifecycle: the consumer ``create()``s the segment (it knows the scene sizes from
its own Model) and ``close(unlink=True)``s it; the producer ``attach()``es by
name and ``close()``s without unlinking.
"""

from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np

_MAGIC = 0x5359_4E47  # "SYNG"
_HDR_LEN = 8          # int64 slots: [magic, n_verts, n_bodies, seq, ...reserved]
_HDR_BYTES = _HDR_LEN * 8


class GeomSnapshotShm:
    """Double-duty (writer or reader) handle to one geometry snapshot segment."""

    def __init__(self, shm: shared_memory.SharedMemory, n_verts: int, n_bodies: int, owner: bool):
        self._shm = shm
        self.n_verts = int(n_verts)
        self.n_bodies = int(n_bodies)
        self._owner = owner
        self._hdr = np.ndarray((_HDR_LEN,), dtype=np.int64, buffer=shm.buf, offset=0)
        off = _HDR_BYTES
        self._part = np.ndarray((self.n_verts, 3), dtype=np.float32, buffer=shm.buf, offset=off)
        off += self.n_verts * 3 * 4
        self._body = np.ndarray((self.n_bodies, 7), dtype=np.float32, buffer=shm.buf, offset=off)

    # -- construction ---------------------------------------------------
    @staticmethod
    def _nbytes(n_verts: int, n_bodies: int) -> int:
        return _HDR_BYTES + n_verts * 3 * 4 + n_bodies * 7 * 4

    @classmethod
    def create(cls, name: str, n_verts: int, n_bodies: int) -> "GeomSnapshotShm":
        """Consumer side: allocate the segment sized for this scene."""
        shm = shared_memory.SharedMemory(
            name=name, create=True, size=cls._nbytes(n_verts, n_bodies))
        self = cls(shm, n_verts, n_bodies, owner=True)
        self._hdr[:4] = (_MAGIC, n_verts, n_bodies, 0)
        return self

    @classmethod
    def attach(cls, name: str) -> "GeomSnapshotShm":
        """Producer side: map an existing segment and read its sizes from the header."""
        shm = shared_memory.SharedMemory(name=name, create=False)
        # The attaching process must not own the segment's lifecycle — the creator
        # (consumer) unlinks it. Otherwise Python's resource_tracker double-unlinks
        # and warns at shutdown (bpo-39959).
        try:
            from multiprocessing import resource_tracker

            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        hdr = np.ndarray((_HDR_LEN,), dtype=np.int64, buffer=shm.buf, offset=0)
        if int(hdr[0]) != _MAGIC:
            shm.close()
            raise ValueError(f"GeomSnapshotShm: bad magic in segment {name!r}")
        return cls(shm, int(hdr[1]), int(hdr[2]), owner=False)

    # -- producer -------------------------------------------------------
    def write(self, particle_q: np.ndarray, body_q: np.ndarray | None) -> None:
        """Publish one frame (seqlock: bump to odd, copy, bump to even)."""
        seq = int(self._hdr[3])
        self._hdr[3] = seq + 1               # odd -> write in progress
        if self.n_verts:
            self._part[:] = np.asarray(particle_q, dtype=np.float32)[: self.n_verts]
        if self.n_bodies and body_q is not None:
            self._body[:] = np.asarray(body_q, dtype=np.float32)[: self.n_bodies]
        self._hdr[3] = seq + 2               # even -> done

    # -- consumer -------------------------------------------------------
    def read(self) -> tuple[np.ndarray, np.ndarray, int] | None:
        """Latest frame as ``(particle_q, body_q, seq)``, or None if never written.

        Seqlock read: retry while a write straddles the copy. Bounded retries so a
        crashed writer can't spin us forever.
        """
        for _ in range(8):
            s0 = int(self._hdr[3])
            if s0 == 0:
                return None
            if s0 & 1:                       # writer mid-frame
                continue
            part = self._part.copy()
            body = self._body.copy() if self.n_bodies else np.empty((0, 7), np.float32)
            s1 = int(self._hdr[3])
            if s0 == s1:
                return part, body, s0 >> 1
        return None

    def close(self, unlink: bool | None = None) -> None:
        # numpy views must drop their reference to buf before SharedMemory.close().
        self._hdr = self._part = self._body = None  # type: ignore[assignment]
        try:
            self._shm.close()
        finally:
            if unlink if unlink is not None else self._owner:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
