"""The skeleton stage: per-body fault isolation, the failure breaker, ROI resume and the
stale-manifest guard. The machinery is neu-morpho's (``neu_morpho.ops.manifest``); these
check that this stage applies it."""

import importlib
import json
import os

import numpy as np
import pytest

from neu_morpho.config import OutputConfig
from neu_morpho.ops.manifest import FAILED, is_complete
from neu_skel.config import SkeletonConfig
from neu_skel.stage import skeletonize_segments


def _write_seg_zarr(path, vol):
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import zarr3_create_spec

    be = TensorStoreBackend.create(
        zarr3_create_spec("local", path, vol.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(16, 16, 16)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)
    return {"backend": "zarr3", "path": path}


def _three_body_volume():
    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:28, 4:10, 4:10] = 7
    vol[4:28, 14:20, 14:20] = 8
    vol[4:28, 24:30, 24:30] = 9
    return vol


def _two_body_volume():
    vol = np.zeros((64, 32, 32), np.uint64)
    vol[4:28, 12:20, 12:20] = 7      # lives in z-blocks 0-1
    vol[36:60, 12:20, 12:20] = 9     # lives in z-blocks 2-3
    return vol


SKEL_CFG = dict(anisotropy=(8.0, 8.0, 8.0), block_shape=(16, 16, 16), const=30.0,
                dust_threshold=0, postprocess_dust_nm=0.0, postprocess_tick_nm=0.0)


def test_skeleton_fuse_failure_is_isolated_and_retried(tmp_path, monkeypatch):
    mod = importlib.import_module("neu_skel.stage")

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    cfg = SkeletonConfig(**SKEL_CFG)

    real = mod.fuse_body
    calls = {"n": 0}

    def flaky(fragments, cfg_, body_id=None, stats=None):
        if body_id == 8:                       # one poisoned body
            calls["n"] += 1
            raise RuntimeError("synthetic fusion failure")
        return real(fragments, cfg_, body_id=body_id, stats=stats)

    monkeypatch.setattr(mod, "fuse_body", flaky)
    s1 = skeletonize_segments(src, out, cfg, client=None)

    # the run completed, the other bodies are on disk, the failure is recorded
    assert s1["failed_bodies"] == [8]
    assert s1["status_counts"].get(FAILED) == 1
    assert s1["status_counts"].get("written") == 2
    assert os.path.exists(os.path.join(s1["out_dir"], "7"))
    assert os.path.exists(os.path.join(s1["out_dir"], "9"))
    assert not os.path.exists(os.path.join(s1["out_dir"], "8"))

    rows = [json.loads(ln) for ln in open(s1["failures_path"])]
    assert rows[0]["body_id"] == 8 and "synthetic fusion failure" in rows[0]["error"]

    # resume retries ONLY the failed body, and it succeeds once unpoisoned
    monkeypatch.setattr(mod, "fuse_body", real)
    s2 = skeletonize_segments(src, out, cfg, client=None)
    assert s2["num_bodies_fused"] == 1                # not 3 — the others stayed done
    assert s2["failed_bodies"] == []
    assert os.path.exists(os.path.join(s2["out_dir"], "8"))
    assert s2["status_counts"].get(FAILED) is None    # the record was superseded


def test_failed_body_metrics_are_not_written(tmp_path, monkeypatch):
    """A failed body must not leave half-written metrics behind."""
    mod = importlib.import_module("neu_skel.stage")
    from neu_morpho.metrics_db import MetricsDB

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    db_path = str(tmp_path / "m.db")

    def flaky(fragments, cfg_, body_id=None, stats=None):
        raise RuntimeError("all bodies fail")

    monkeypatch.setattr(mod, "fuse_body", flaky)
    skeletonize_segments(src, out, SkeletonConfig(**SKEL_CFG), db_path=db_path, client=None)

    db = MetricsDB(db_path)
    rows = db.con.execute("SELECT COUNT(*) FROM bodies WHERE cable_length_nm IS NOT NULL")
    assert rows.fetchone()[0] == 0
    db.close()


def test_stage_aborts_after_consecutive_failures_but_keeps_diagnostics(tmp_path, monkeypatch):
    """An abort still leaves the failure records, and they stay retryable."""
    mod = importlib.import_module("neu_skel.stage")
    from neu_morpho.ops.manifest import StageAborted

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))

    def always_fail(*a, **k):
        raise ValueError("everything is broken")

    monkeypatch.setattr(mod, "fuse_body", always_fail)
    with pytest.raises(StageAborted, match="2 consecutive"):
        skeletonize_segments(src, out, SkeletonConfig(**SKEL_CFG),
                             max_consecutive_failures=2, client=None)

    # written in `finally`, so the abort still leaves the tracebacks behind
    path = os.path.join(str(tmp_path / "out"), "failures.skel.jsonl")
    rows = [json.loads(ln) for ln in open(path)]
    assert len(rows) == 2 and "everything is broken" in rows[0]["error"]

    # and the failures are recorded as retryable, not as done
    from blockrun import Manifest
    m = Manifest(os.path.join(str(tmp_path / "out"), "progress.skel.jsonl")).load()
    assert not is_complete(m, "skel-fuse", rows[0]["body_id"])


def test_systemic_error_aborts_on_the_first_body(tmp_path, monkeypatch):
    mod = importlib.import_module("neu_skel.stage")
    from neu_morpho.ops.manifest import StageAborted

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    seen = {"n": 0}

    def oom(*a, **k):
        seen["n"] += 1
        raise MemoryError()

    monkeypatch.setattr(mod, "fuse_body", oom)
    with pytest.raises(StageAborted, match="systemic"):
        skeletonize_segments(src, out, SkeletonConfig(**SKEL_CFG),
                             max_consecutive_failures=1000, client=None)
    assert seen["n"] == 1          # stopped at the first body, not all three


def test_successes_in_the_aborting_batch_still_get_their_metrics(tmp_path, monkeypatch):
    """An abort leaves no skeleton on disk without its metrics in the DB."""
    mod = importlib.import_module("neu_skel.stage")
    from neu_morpho.metrics_db import MetricsDB
    from neu_morpho.ops.manifest import StageAborted

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    db_path = str(tmp_path / "m.db")
    real = mod.fuse_body

    def fail_first(fragments, cfg_, body_id=None, stats=None):
        if body_id == 7:
            raise ValueError("first body fails")
        return real(fragments, cfg_, body_id=body_id, stats=stats)

    monkeypatch.setattr(mod, "fuse_body", fail_first)
    # trips on the very first failure, but bodies 8 and 9 already succeeded
    with pytest.raises(StageAborted):
        skeletonize_segments(src, out, SkeletonConfig(**SKEL_CFG), db_path=db_path,
                             max_consecutive_failures=1, client=None)

    db = MetricsDB(db_path)
    written = {r[0] for r in db.con.execute(
        "SELECT body_id FROM bodies WHERE cable_length_nm IS NOT NULL")}
    db.close()
    skel_dir = os.path.join(out.volume_dir(), out.skeleton_dir)   # inside the volume
    on_disk = {int(f) for f in os.listdir(skel_dir) if f.isdigit()}
    assert on_disk == written, "a skeleton on disk without metrics in the DB"


def test_stage1_block_failure_aborts(tmp_path, monkeypatch):
    """A block failure is loud: a silently skipped block yields truncated bodies."""
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))

    def boom(*a, **k):
        raise RuntimeError("block failure")

    monkeypatch.setattr(importlib.import_module("neu_skel.stage"), "skeletonize_block", boom)
    with pytest.raises(RuntimeError, match="block failure"):
        skeletonize_segments(src, out, SkeletonConfig(**SKEL_CFG), stages=("skel-chunk",),
                             client=None)


def test_stage1_progress_before_an_abort_is_durable(tmp_path, monkeypatch):
    """Blocks recorded before a crash are not redone on resume."""
    mod = importlib.import_module("neu_skel.stage")

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _three_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    cfg = SkeletonConfig(**SKEL_CFG)

    real = mod.skeletonize_block
    seen = {"n": 0}

    def boom_after_three(*a, **k):
        seen["n"] += 1
        if seen["n"] > 3:
            raise RuntimeError("dies partway")
        return real(*a, **k)

    monkeypatch.setattr(mod, "skeletonize_block", boom_after_three)
    with pytest.raises(RuntimeError):
        skeletonize_segments(src, out, cfg, stages=("skel-chunk",), client=None)

    assert seen["n"] == 4          # 3 succeeded and were recorded, the 4th raised

    retried = {"n": 0}

    def counting(*a, **k):
        retried["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mod, "skeletonize_block", counting)
    resumed = skeletonize_segments(src, out, cfg, stages=("skel-chunk",), client=None)
    # 8 blocks total; the 3 recorded before the crash are not redone
    assert retried["n"] == 5
    assert sum(resumed["chunk_counts"].values()) == 8


def test_roi_restricts_skeletonization_then_extends_on_resume(tmp_path):
    """Widening the ROI later reuses the first run's blocks."""
    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _two_body_volume())
    out = OutputConfig(dst=str(tmp_path / "out" / "segmentation"), work_dir=str(tmp_path / "out"))
    cfg = SkeletonConfig(**SKEL_CFG)

    # grid is 4x2x2 = 16 blocks; the ROI covers z-blocks 0-1 -> 2*2*2 = 8
    first = skeletonize_segments(src, out, cfg, roi="0,0,0,32,32,32", client=None)
    assert first["num_blocks"] == 8
    assert os.path.exists(os.path.join(first["out_dir"], "7"))
    assert not os.path.exists(os.path.join(first["out_dir"], "9"))

    # widen to the whole volume: the first 8 blocks are already done, 8 remain
    done_after_first = set(first["chunk_counts"])
    assert sum(first["chunk_counts"].values()) == 8
    second = skeletonize_segments(src, out, cfg, client=None)
    assert second["num_blocks"] == 16
    assert sum(second["chunk_counts"].values()) == 16     # 8 reused + 8 new
    assert os.path.exists(os.path.join(second["out_dir"], "9"))


def test_a_manifest_outliving_its_output_is_refused_not_silently_skipped(tmp_path):
    """Manifests live in the work dir, not in dst, so clearing dst leaves a manifest that
    claims everything is done. Resuming must refuse rather than write nothing."""
    import shutil

    from neu_morpho.ops.manifest import StaleManifest

    src = _write_seg_zarr(str(tmp_path / "seg.zarr"), _two_body_volume())
    work, dst = str(tmp_path / "out"), str(tmp_path / "out" / "segmentation")
    out = OutputConfig(dst=dst, work_dir=work)
    cfg = SkeletonConfig(**SKEL_CFG)

    s1 = skeletonize_segments(src, out, cfg, client=None)
    path = os.path.abspath(s1["progress_path"])
    assert os.path.commonpath([path, os.path.abspath(work)]) == os.path.abspath(work)
    assert not path.startswith(os.path.abspath(dst) + os.sep), "manifest inside the volume"

    shutil.rmtree(dst)                        # the data is cleared; the manifest survives
    with pytest.raises(StaleManifest, match="no 'info' at"):
        skeletonize_segments(src, out, cfg, client=None)

    # resume=False is an explicit fresh start, so it proceeds and rewrites
    s2 = skeletonize_segments(src, out, cfg, client=None, resume=False)
    assert s2["num_bodies_fused"] > 0
    assert os.path.exists(os.path.join(s2["out_dir"], "7"))
