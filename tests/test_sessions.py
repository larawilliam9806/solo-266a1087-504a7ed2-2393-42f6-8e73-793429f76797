"""Tests for batched, ordered, crash-safe session uploads."""

from __future__ import annotations

import io
import threading
import time
import tarfile

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import STORE, app
from app.sessions import SessionStore


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def tar_bytes(members: list[tuple]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for spec in members:
            kind = spec[0]
            if kind == "dir":
                ti = tarfile.TarInfo(spec[1])
                ti.type = tarfile.DIRTYPE
                ti.mode = 0o755
                tar.addfile(ti)
            elif kind == "file":
                _, name, data = spec
                if isinstance(data, str):
                    data = data.encode()
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def one_file_layer(name: str, data: bytes) -> bytes:
    return tar_bytes([("file", name, data)])


def open_tar(resp) -> tarfile.TarFile:
    assert resp.status_code == 200, resp.text
    return tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:")


def new_session(client: TestClient) -> str:
    r = client.post("/sessions")
    assert r.status_code == 200
    return r.json()["session_id"]


def contents(tf: tarfile.TarFile, name: str) -> bytes:
    f = tf.extractfile(name)
    assert f is not None
    return f.read()


# ---------------------------------------------------------------------------
# Whole-batch append / ordering
# ---------------------------------------------------------------------------

def test_batch_appends_in_form_order(client: TestClient):
    sid = new_session(client)
    l1 = one_file_layer("cfg/a", b"a")
    l2 = one_file_layer("cfg/b", b"b")
    r = client.post(
        f"/sessions/{sid}/layers",
        files=[
            ("layers", ("l1.tar", l1, "application/x-tar")),
            ("layers", ("l2.tar", l2, "application/x-tar")),
        ],
    )
    assert r.status_code == 200
    assert r.json()["added"] == 2

    # The top layer of the next batch overrides the earlier one.
    top = one_file_layer("cfg/a", b"a-top")
    client.post(
        f"/sessions/{sid}/layers",
        files=[("layers", ("top.tar", top, "application/x-tar"))],
    )
    tf = open_tar(client.get(f"/sessions/{sid}/download", params={"path": "/cfg"}))
    assert contents(tf, "cfg/a") == b"a-top"
    assert contents(tf, "cfg/b") == b"b"


def test_multiple_batches_keep_submission_order(client: TestClient):
    sid = new_session(client)
    # bottom -> middle -> top across three separate requests.
    for value in (b"v1", b"v2", b"v3"):
        r = client.post(
            f"/sessions/{sid}/layers",
            files=[("layers", ("l.tar", one_file_layer("cfg/x", value), "application/x-tar"))],
        )
        assert r.status_code == 200
    tf = open_tar(client.get(f"/sessions/{sid}/download", params={"path": "/cfg"}))
    assert contents(tf, "cfg/x") == b"v3"


# ---------------------------------------------------------------------------
# Atomicity: a failed batch is discarded entirely
# ---------------------------------------------------------------------------

def test_failed_batch_is_discarded_and_previous_layers_survive(
    client: TestClient, monkeypatch
):
    sid = new_session(client)
    good = one_file_layer("cfg/keep", b"keep")
    r = client.post(
        f"/sessions/{sid}/layers",
        files=[("layers", ("good.tar", good, "application/x-tar"))],
    )
    assert r.status_code == 200

    # Make the *second* file of the next batch fail to write.
    real_save = main_module._save_upload
    calls = {"n": 0}

    def flaky(upload, dest):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real_save(upload, dest)

    monkeypatch.setattr(main_module, "_save_upload", flaky)

    l_a = one_file_layer("cfg/partial_a", b"a")
    l_b = one_file_layer("cfg/partial_b", b"b")
    failed = client.post(
        f"/sessions/{sid}/layers",
        files=[
            ("layers", ("a.tar", l_a, "application/x-tar")),
            ("layers", ("b.tar", l_b, "application/x-tar")),
        ],
    )
    assert failed.status_code == 500
    assert failed.json()["error"] == "upload_failed"

    # Previously committed configuration is still downloadable and complete.
    tf = open_tar(client.get(f"/sessions/{sid}/download", params={"path": "/cfg"}))
    names = set(tf.getnames())
    assert "cfg/keep" in names
    assert "cfg/partial_a" not in names
    assert "cfg/partial_b" not in names
    assert contents(tf, "cfg/keep") == b"keep"

    # The session is still usable for new appends afterwards.
    monkeypatch.setattr(main_module, "_save_upload", real_save)
    after = one_file_layer("cfg/after", b"after")
    ok = client.post(
        f"/sessions/{sid}/layers",
        files=[("layers", ("after.tar", after, "application/x-tar"))],
    )
    assert ok.status_code == 200
    tf2 = open_tar(client.get(f"/sessions/{sid}/download", params={"path": "/cfg"}))
    assert contents(tf2, "cfg/after") == b"after"


def test_staged_files_are_not_visible_before_commit(client: TestClient, monkeypatch):
    sid = new_session(client)
    client.post(
        f"/sessions/{sid}/layers",
        files=[("layers", ("l.tar", one_file_layer("cfg/a", b"a"), "application/x-tar"))],
    )

    seen_layers: list[int] = []
    real_write_batch = main_module._write_batch

    def slow_write_batch(sid_arg, seq, files):
        # At this point the batch is still in a private staging dir.
        seen_layers.append(len(STORE.snapshot(sid_arg)))
        return real_write_batch(sid_arg, seq, files)

    monkeypatch.setattr(main_module, "_write_batch", slow_write_batch)
    client.post(
        f"/sessions/{sid}/layers",
        files=[("layers", ("l2.tar", one_file_layer("cfg/b", b"b"), "application/x-tar"))],
    )
    assert seen_layers == [1]

    tf = open_tar(client.get(f"/sessions/{sid}/download", params={"path": "/cfg"}))
    assert contents(tf, "cfg/b") == b"b"


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_batches_are_not_lost(client: TestClient):
    """Many simultaneous uploads: none may overwrite another."""
    sid = new_session(client)
    n_batches = 12
    barrier = threading.Barrier(n_batches)
    errors: list[BaseException] = []

    def upload(i: int) -> None:
        try:
            layer = one_file_layer(f"cfg/f{i:02d}", f"val-{i}".encode())
            barrier.wait(timeout=10)
            r = client.post(
                f"/sessions/{sid}/layers",
                files=[("layers", (f"l{i}.tar", layer, "application/x-tar"))],
            )
            assert r.status_code == 200, r.text
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=upload, args=(i,)) for i in range(n_batches)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    assert len(STORE.snapshot(sid)) == n_batches

    tf = open_tar(client.get(f"/sessions/{sid}/download", params={"path": "/cfg"}))
    names = set(tf.getnames())
    for i in range(n_batches):
        assert f"cfg/f{i:02d}" in names
        assert contents(tf, f"cfg/f{i:02d}") == f"val-{i}".encode()


def test_later_finished_batch_does_not_jump_ahead():
    """Submission order is decided before the body is written.

    Batch A reserves sequence 0, then takes a long time to write.  Batch B
    reserves sequence 1 and commits first.  A's layers must still occupy the
    *bottom* of the stack (earlier order), never overwriting B.
    """
    store = SessionStore(main_module.TMP_DIR / "order-test-sessions")
    sid = "ordercheck"
    store.create(sid)

    seq_a = store.allocate_seq(sid)
    time.sleep(0.05)
    seq_b = store.allocate_seq(sid)
    assert (seq_a, seq_b) == (0, 1)

    def stage_and_commit(seq: int, name: str, payload: bytes):
        staged = store.new_staging(sid)
        (staged / "layer_0000.tar").write_bytes(one_file_layer(name, payload))
        return store.commit(sid, seq, staged, ["layer_0000.tar"])

    # B finishes first even though it was submitted later.
    commit_b = stage_and_commit(seq_b, "cfg/order", b"from-B")
    time.sleep(0.05)
    commit_a = stage_and_commit(seq_a, "cfg/order", b"from-A")

    snap = store.snapshot(sid)
    assert len(snap) == 2
    # Snapshot order: A (seq 0) then B (seq 1) -> B is the top layer.
    assert snap[0].read_bytes() == one_file_layer("cfg/order", b"from-A")
    assert snap[1].read_bytes() == one_file_layer("cfg/order", b"from-B")
    assert commit_a.seq == 0 and commit_b.seq == 1


def test_download_works_while_a_batch_is_being_written(client: TestClient, monkeypatch):
    """A download during a slow upload must still return committed layers."""
    sid = new_session(client)
    client.post(
        f"/sessions/{sid}/layers",
        files=[("layers", ("base.tar", one_file_layer("cfg/base", b"base"), "application/x-tar"))],
    )

    started = threading.Event()
    release = threading.Event()
    real_write_batch = main_module._write_batch

    def blocking_write_batch(sid_arg, seq, files):
        started.set()
        release.wait(timeout=10)
        return real_write_batch(sid_arg, seq, files)

    monkeypatch.setattr(main_module, "_write_batch", blocking_write_batch)

    uploader = threading.Thread(
        target=lambda: client.post(
            f"/sessions/{sid}/layers",
            files=[("layers", ("slow.tar", one_file_layer("cfg/slow", b"slow"), "application/x-tar"))],
        )
    )
    uploader.start()
    assert started.wait(timeout=10)
    try:
        # Download must complete without waiting for the in-flight batch.
        tf = open_tar(client.get(f"/sessions/{sid}/download", params={"path": "/cfg"}))
        names = set(tf.getnames())
        assert "cfg/base" in names
        assert "cfg/slow" not in names
    finally:
        release.set()
        uploader.join(timeout=10)


# ---------------------------------------------------------------------------
# Persistence / compatibility
# ---------------------------------------------------------------------------

def test_session_survives_restart_like_reopen(client: TestClient):
    """A new store instance over the same directory sees the same layers."""
    sid = new_session(client)
    client.post(
        f"/sessions/{sid}/layers",
        files=[("layers", ("l.tar", one_file_layer("cfg/p", b"persisted"), "application/x-tar"))],
    )
    reopened = SessionStore(STORE.root)
    paths = reopened.snapshot(sid)
    assert len(paths) == 1
    assert paths[0].read_bytes() == one_file_layer("cfg/p", b"persisted")

    tf = open_tar(client.get(f"/sessions/{sid}/download", params={"path": "/cfg"}))
    assert contents(tf, "cfg/p") == b"persisted"


def test_legacy_flat_session_is_adopted(client: TestClient):
    """Sessions written by the old implementation are still usable."""
    sid = "legacy" + "0" * 6
    sdir = STORE.session_dir(sid)
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "layer_0000.tar").write_bytes(one_file_layer("cfg/legacy_a", b"a"))
    (sdir / "layer_0001.tar").write_bytes(one_file_layer("cfg/legacy_b", b"b"))

    d = client.get(f"/sessions/{sid}/download", params={"path": "/cfg"})
    tf = open_tar(d)
    assert contents(tf, "cfg/legacy_a") == b"a"
    assert contents(tf, "cfg/legacy_b") == b"b"

    # New appends go on top of the adopted layers.
    up = client.post(
        f"/sessions/{sid}/layers",
        files=[("layers", ("n.tar", one_file_layer("cfg/legacy_c", b"c"), "application/x-tar"))],
    )
    assert up.status_code == 200
    tf2 = open_tar(client.get(f"/sessions/{sid}/download", params={"path": "/cfg"}))
    assert contents(tf2, "cfg/legacy_c") == b"c"


def test_interrupted_batch_does_not_break_existing_download(client: TestClient):
    """A crash leaving staging files behind must not affect committed layers."""
    sid = new_session(client)
    client.post(
        f"/sessions/{sid}/layers",
        files=[("layers", ("base.tar", one_file_layer("cfg/base", b"base"), "application/x-tar"))],
    )

    # Simulate a process killed mid-batch: staged files exist but were never
    # committed (no manifest entry).
    staged = STORE.new_staging(sid)
    (staged / "layer_0000.tar").write_bytes(one_file_layer("cfg/half", b"half"))
    # The snapshot must ignore the uncommitted staging directory.
    assert len(STORE.snapshot(sid)) == 1

    tf = open_tar(client.get(f"/sessions/{sid}/download", params={"path": "/cfg"}))
    names = set(tf.getnames())
    assert "cfg/base" in names
    assert "cfg/half" not in names

    # Startup cleanup reclaims the abandoned staging area.
    STORE.cleanup_stale_staging(max_age_seconds=0)
    assert not staged.exists()

    # And new batches still append normally.
    up = client.post(
        f"/sessions/{sid}/layers",
        files=[("layers", ("new.tar", one_file_layer("cfg/new", b"new"), "application/x-tar"))],
    )
    assert up.status_code == 200
    tf2 = open_tar(client.get(f"/sessions/{sid}/download", params={"path": "/cfg"}))
    assert contents(tf2, "cfg/new") == b"new"
    assert contents(tf2, "cfg/base") == b"base"


def test_flat_upload_batches_are_ordered(client: TestClient):
    """Repeated /upload calls append as ordered batches to the default stack."""
    # Use a dedicated explicit session via the header so other tests are safe.
    sid = new_session(client)
    hdrs = {"x-session-id": sid}
    for value in (b"one", b"two", b"three"):
        r = client.post(
            "/upload",
            headers=hdrs,
            files=[("layer", ("l.tar", one_file_layer("cfg/u", value), "application/x-tar"))],
        )
        assert r.status_code == 200
        assert r.json()["session_id"] == sid
    tf = open_tar(client.get(f"/sessions/{sid}/download", params={"path": "/cfg"}))
    assert contents(tf, "cfg/u") == b"three"
