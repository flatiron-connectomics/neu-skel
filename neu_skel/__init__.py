"""neu-skel: TEASAR skeletons per segment, a stage added to a neu-morpho run.

Two tracers: ``neutu_trace`` (default), a reimplementation of NeuTu's TEASAR, and
``kimimaro``, required for anisotropic voxels. Skeletons are traced per block and fused
per body, written as neuroglancer-precomputed skeletons inside the volume neu-morpho
publishes. GPL-3.0-or-later, because both tracers build on ``kimimaro`` and
``dijkstra3d``; everything that reads and measures the result is in neu-morpho.

From Python::

    from neu_morpho import pipeline
    from neu_skel import run_stage

    p = pipeline.plan(src, dst, work_dir=work, stages=("skel",), extra_stages={"skel": 2})
    pipeline.run(p, runners={"skel": run_stage})
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import SkeletonConfig
from .skeletonize import fuse_body, fusion_stats_summary, skeletonize_block
from .stage import run_stage, skeletonize_segments

__all__ = ["__version__", "SkeletonConfig", "fuse_body", "fusion_stats_summary",
           "run_stage", "skeletonize_block", "skeletonize_segments"]
