"""Concurrency-safe, append-only session layer store.

A session is a directory holding an ordered sequence of *batches*.  A batch is
exactly the set of files uploaded by one HTTP request, in multipart form order.

Guarantees
----------
* **Whole-batch append.**  Uploads are written into a private staging
  directory first.  Only after every file has been stored is the batch moved
  into place and published with a single atomic manifest swap.  A batch that
  fails halfway through is discarded in its entirety; previously committed
  batches are untouched.
* **Submission ordering.**  Every request reserves a monotonically increasing
  sequence number before it starts reading the request body.  Concurrent
  batches are therefore ordered by when they were submitted, even when a
  later batch finishes writing first.  Each batch owns a unique directory, so
  a batch can never overwrite another one.
* **Readers see complete batches only.**  :meth:`SessionStore.snapshot`
  returns the batches recorded in the manifest at the moment it is called.
  Because the manifest is replaced atomically and committed layer files are
  immutable, an extraction started at that point never observes a partially
  written batch.
* **Concurrent reads.**  Staging/uploading takes no shared lock, and the
  manifest read path is lock-free (``os.replace`` gives readers a complete old
  or new file), so downloads keep working while a new batch is being received
  or written.
* **Backwards compatibility.**  Sessions created by older versions (plain
  ``layer_0000.tar`` files in the session root) are adopted lazily as
  committed one-layer batches the first time they are touched.

Cross-process safety is provided by ``flock`` on a per-session lock file in
addition to an in-process ``threading.RLock``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

try:  # pragma: no cover - not available on Windows, which we do not target
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

MANIFEST_NAME = "session.json"
MANIFEST_VERSION = 1
BATCH_DIR = "batches"
STAGING_DIR = ".staging"
LOCK_NAME = ".lock"

# Layers stored by earlier releases live directly in the session directory.
_LEGACY_LAYER_RE = re.compile(r"^layer_(\d+)\.tar$")


class SessionStoreError(Exception):
    """Base error for the session store."""


class SessionNotFound(SessionStoreError):
    """The session directory does not exist."""


class SessionExists(SessionStoreError):
    """The session directory already exists."""


class InvalidManifest(SessionStoreError):
    """The session manifest is unreadable/corrupt."""


@dataclass(frozen=True)
class CommittedBatch:
    seq: int
    dir: str
    files: tuple[str, ...]


def _read_manifest_file(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except OSError as exc:  # pragma: no cover - unusual fs errors
        raise InvalidManifest(f"cannot read {path.name}: {exc}") from exc
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise InvalidManifest(f"cannot parse {path.name}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("batches"), list):
        raise InvalidManifest(f"{path.name} has an unexpected shape")
    return data


class SessionStore:
    """Filesystem-backed ordered layer batches for upload sessions."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks_guard = threading.Lock()
        self._thread_locks: dict[str, threading.RLock] = {}

    # -- paths / lifecycle ---------------------------------------------------

    def session_dir(self, sid: str) -> Path:
        return self.root / sid

    def exists(self, sid: str) -> bool:
        return self.session_dir(sid).is_dir()

    def create(self, sid: str) -> Path:
        """Create a brand new, empty session."""
        sdir = self.session_dir(sid)
        try:
            sdir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise SessionExists(sid) from exc
        try:
            with self._exclusive(sid, sdir):
                self._load_locked(sdir)
        except BaseException:
            shutil.rmtree(sdir, ignore_errors=True)
            raise
        return sdir

    def ensure(self, sid: str) -> Path:
        """Create the session if needed (used by the shared ``default`` stack)."""
        sdir = self.session_dir(sid)
        sdir.mkdir(parents=True, exist_ok=True)
        with self._exclusive(sid, sdir):
            self._load_locked(sdir)
        return sdir

    def delete(self, sid: str) -> None:
        shutil.rmtree(self.session_dir(sid), ignore_errors=True)
        with self._locks_guard:
            self._thread_locks.pop(sid, None)

    def cleanup_stale_staging(self, max_age_seconds: int = 3600) -> None:
        """Remove staging directories left behind by a crashed process."""
        import time

        if not self.root.is_dir():
            return
        now = time.time()
        for sdir in self.root.iterdir():
            staging = sdir / STAGING_DIR
            if not staging.is_dir():
                continue
            for child in staging.iterdir():
                try:
                    if now - child.stat().st_mtime > max_age_seconds:
                        shutil.rmtree(child, ignore_errors=True)
                except OSError:
                    continue

    # -- locking -------------------------------------------------------------

    def _thread_lock(self, sid: str) -> threading.RLock:
        with self._locks_guard:
            lock = self._thread_locks.get(sid)
            if lock is None:
                lock = threading.RLock()
                self._thread_locks[sid] = lock
            return lock

    @contextmanager
    def _exclusive(self, sid: str, sdir: Path) -> Iterator[None]:
        """Serialise manifest mutations for one session (threads + processes)."""
        with self._thread_lock(sid):
            fd = os.open(str(sdir / LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                try:
                    if fcntl is not None:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    # -- manifest ------------------------------------------------------------

    def _write_locked(self, sdir: Path, data: dict) -> None:
        fd, tmp = tempfile.mkstemp(
            dir=str(sdir), prefix=".session-", suffix=".json.tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, sort_keys=True)
                fh.flush()
            # Atomic swap: readers observe either the complete old manifest or
            # the complete new one, never a half-written file.  We deliberately
            # skip fsync here so this stays cheap enough to run on the event
            # loop; layer bytes themselves are fsynced before being published.
            os.replace(tmp, sdir / MANIFEST_NAME)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _load_locked(self, sdir: Path) -> dict:
        """Read the manifest, or adopt a legacy session / create an empty one."""
        path = sdir / MANIFEST_NAME
        try:
            return _read_manifest_file(path)
        except FileNotFoundError:
            pass

        # No manifest: adopt any layers written by the pre-batch implementation.
        legacy: list[tuple[int, str]] = []
        try:
            children = list(sdir.iterdir())
        except OSError as exc:
            raise SessionStoreError(f"cannot list session: {exc}") from exc
        for child in children:
            if not child.is_file():
                continue
            match = _LEGACY_LAYER_RE.match(child.name)
            if match:
                legacy.append((int(match.group(1)), child.name))
        legacy.sort(key=lambda item: item[0])

        data = {
            "version": MANIFEST_VERSION,
            "next_seq": (legacy[-1][0] + 1) if legacy else 0,
            "batches": [
                {"seq": seq, "dir": "", "files": [name]}
                for seq, name in legacy
            ],
        }
        self._write_locked(sdir, data)
        return data

    def _read_manifest_fast(self, sdir: Path) -> dict:
        """Lock-free read: ``os.replace`` guarantees a complete file."""
        path = sdir / MANIFEST_NAME
        try:
            return _read_manifest_file(path)
        except FileNotFoundError:
            with self._exclusive(sdir.name, sdir):
                return self._load_locked(sdir)

    # -- batches -------------------------------------------------------------

    def allocate_seq(self, sid: str) -> int:
        """Reserve the next submission-order slot for a batch.

        Called as soon as a request arrives, before its body is parsed, so
        that concurrently uploaded batches end up ordered by submission rather
        than by completion.
        """
        sdir = self.session_dir(sid)
        if not sdir.is_dir():
            raise SessionNotFound(sid)
        with self._exclusive(sid, sdir):
            data = self._load_locked(sdir)
            seq = int(data.get("next_seq", 0))
            data["next_seq"] = seq + 1
            self._write_locked(sdir, data)
        return seq

    def new_staging(self, sid: str) -> Path:
        """Create a private directory to receive one batch."""
        sdir = self.session_dir(sid)
        if not sdir.is_dir():
            raise SessionNotFound(sid)
        staging_root = sdir / STAGING_DIR
        staging_root.mkdir(parents=True, exist_ok=True)
        staged = staging_root / uuid.uuid4().hex
        staged.mkdir(parents=False, exist_ok=False)
        return staged

    def discard(self, staged: Optional[Path]) -> None:
        """Drop a staging directory (no-op once it has been committed)."""
        if staged is None:
            return
        shutil.rmtree(staged, ignore_errors=True)

    def commit(
        self, sid: str, seq: int, staged: Path, filenames: Sequence[str]
    ) -> CommittedBatch:
        """Publish a fully written staging directory as one atomic batch."""
        sdir = self.session_dir(sid)
        if not sdir.is_dir():
            raise SessionNotFound(sid)

        batch_name = uuid.uuid4().hex
        target = sdir / BATCH_DIR / batch_name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Same filesystem: rename publishes all batch files at once.
        os.replace(str(staged), str(target))

        entry = {
            "seq": int(seq),
            "dir": f"{BATCH_DIR}/{batch_name}",
            "files": list(filenames),
        }
        try:
            with self._exclusive(sid, sdir):
                data = self._load_locked(sdir)
                used = {int(b.get("seq", 0)) for b in data.get("batches", [])}
                if entry["seq"] in used:
                    # Should not happen with the allocator, but never clobber
                    # an existing batch if a stale sequence sneaks in.
                    entry["seq"] = int(data.get("next_seq", 0))
                data.setdefault("batches", []).append(entry)
                data["next_seq"] = max(
                    int(data.get("next_seq", 0)), entry["seq"] + 1
                )
                self._write_locked(sdir, data)
        except BaseException:
            shutil.rmtree(target, ignore_errors=True)
            raise

        return CommittedBatch(
            seq=entry["seq"],
            dir=entry["dir"],
            files=tuple(entry["files"]),
        )

    # -- reading -------------------------------------------------------------

    def snapshot(self, sid: str) -> list[Path]:
        """Return committed layer paths in append order, as of right now.

        The returned list is a point-in-time view: batches committed after
        this call are not included, and every included batch is complete.
        """
        sdir = self.session_dir(sid)
        if not sdir.is_dir():
            return []

        data = self._read_manifest_fast(sdir)
        batches = sorted(
            data.get("batches", []), key=lambda b: int(b.get("seq", 0))
        )
        paths: list[Path] = []
        for batch in batches:
            rel_dir = str(batch.get("dir") or "")
            base = sdir / rel_dir if rel_dir else sdir
            for name in batch.get("files", []):
                paths.append(base / str(name))
        return paths

    def layer_count(self, sid: str) -> int:
        return len(self.snapshot(sid))

    def display_order(self, sid: str) -> list[str]:
        """Stable, client-facing names for the committed layers."""
        return [f"layer_{i:04d}.tar" for i in range(self.layer_count(sid))]
