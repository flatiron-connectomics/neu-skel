"""Stage-1 tracer selection: units, parity of shape, and the isotropy guard."""
from __future__ import annotations

import os

import numpy as np
import pytest

from neu_skel import skeletonize as skeleton
from neu_skel.config import SkeletonConfig


def test_edge_memory_guard_raises_instead_of_oom():
    """The 'edge' cost's per-component graph is capped, and raises rather than letting a
    dask worker die of OOM. The shipped cap cannot fire at a 256^3 block."""
    from neu_skel import neutu_trace

    m = _two_body_block(n=32)
    tiny = SkeletonConfig(anisotropy=(32.0,) * 3, dust_threshold=0, tracer="neutu",
                          neutu_cost="edge", neutu_edge_max_gb=1e-9)
    with pytest.raises(MemoryError, match="neutu_edge_max_gb|edge"):
        skeleton.skeletonize_block(m, (0, 0, 0), tiny)

    assert 256 ** 3 * neutu_trace.EDGE_BYTES_PER_VOXEL / 1e9 < SkeletonConfig().neutu_edge_max_gb


def test_default_tracer_is_neutu_in_the_config_and_the_cli():
    """The default tracer is a decision, and the CLI passes --tracer explicitly, so the
    two defaults could drift into producing different skeletons."""
    from neu_skel import cli

    args = cli._parse_args(["run", "--src", "x", "--dst", "y", "--work-dir", "z"])
    assert args.tracer == SkeletonConfig().tracer == "neutu"
    assert args.neutu_cost == SkeletonConfig().neutu_cost == "edge"


def _two_body_block(n=48, r=4):
    """One block holding two labelled tubes."""
    m = np.zeros((n, n, n), dtype=np.uint64)
    zz, yy, xx = np.indices(m.shape)
    m[(zz >= 4) & (zz < n - 4) & ((yy - 12) ** 2 + (xx - 12) ** 2 <= r * r)] = 7
    m[(zz >= 4) & (zz < n - 4) & ((yy - 34) ** 2 + (xx - 34) ** 2 <= r * r)] = 9
    return np.asfortranarray(m)


def _cfg(**kw):
    return SkeletonConfig(anisotropy=(32.0, 32.0, 32.0), dust_threshold=0, **kw)


@pytest.mark.parametrize("tracer", ["kimimaro", "neutu"])
def test_both_tracers_return_both_bodies_in_nm(tracer):
    m = _two_body_block()
    out = skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer=tracer))
    assert set(out) == {7, 9}
    for body_id, skel in out.items():
        v = np.asarray(skel.vertices, float)
        # nm, not voxels: a 48-voxel block at 32 nm spans ~1536 nm
        assert v[:, 0].max() > 200.0, f"{tracer}: vertices look like voxels, not nm"
        assert v.max() <= 48 * 32.0 + 1e-3


def test_neutu_radii_are_in_nm_not_voxels():
    """neutu_trace works in voxels, so radii must be scaled to nm; positions alone would
    not reveal radii 32x too small."""
    m = _two_body_block(r=4)
    out = skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer="neutu"))
    radii = np.concatenate([np.asarray(s.radii, float) for s in out.values()])
    # a radius-4-voxel tube at 32 nm/voxel is ~128 nm, nowhere near 4
    assert radii.max() > 60.0, f"radii look like voxels: max {radii.max():.2f}"
    assert radii.max() < 32.0 * 8


@pytest.mark.parametrize("tracer", ["kimimaro", "neutu"])
def test_block_origin_is_applied(tracer):
    m = _two_body_block()
    origin = (10, 20, 30)
    at0 = skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer=tracer))
    off = skeleton.skeletonize_block(m, origin, _cfg(tracer=tracer))
    d0 = np.asarray(at0[7].vertices, float).min(axis=0)
    d1 = np.asarray(off[7].vertices, float).min(axis=0)
    assert np.allclose(d1 - d0, np.asarray(origin, float) * 32.0, atol=1e-3)


def test_neutu_tracer_rejects_anisotropic_voxels():
    m = _two_body_block()
    cfg = SkeletonConfig(tracer="neutu", anisotropy=(40.0, 32.0, 32.0))
    with pytest.raises(ValueError, match="isotropic"):
        skeleton.skeletonize_block(m, (0, 0, 0), cfg)


def test_unknown_tracer_raises():
    m = _two_body_block()
    with pytest.raises(ValueError, match="unknown tracer"):
        skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer="nope"))


@pytest.mark.parametrize("cost", ["voxel", "edge"])
def test_neutu_cost_mode_reaches_the_tracer(cost):
    """Both cost modes reach the tracer and trace."""
    m = _two_body_block()
    out = skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer="neutu",
                                                        neutu_cost=cost))
    assert set(out) == {7, 9}
    for skel in out.values():
        v = np.asarray(skel.vertices, float)
        assert len(v) and v[:, 0].max() > 200.0        # still nm, still traced


def test_neutu_cost_is_validated_not_silently_ignored():
    m = _two_body_block()
    with pytest.raises(ValueError, match="voxel|edge"):
        skeleton.skeletonize_block(m, (0, 0, 0), _cfg(tracer="neutu",
                                                      neutu_cost="nonsense"))


# Config resolution itself is tested in blockrun; here, --config reaches it and fails early.
