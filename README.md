# neu-skel

TEASAR **skeletons** per segment of a segmentation volume, written as
neuroglancer-precomputed skeletons. It adds a `skel` stage to a
[neu-morpho](https://github.com/flatiron-connectomics/neu-suite/tree/main/neu-morpho) run,
so skeletons land in the same volume, metrics DB and run summary as the meshes.

- **Tracers**: `neu_skel.neutu_trace` by default, a reimplementation of NeuTu's TEASAR
  that matches it to sub-voxel centreline agreement with ~2.6× fewer nodes than
  [`kimimaro`], which remains available as `--tracer kimimaro` and is **required for
  anisotropic voxels**. Why each constant is what it is:
  [`docs/skeletonization.md`](docs/skeletonization.md).
- **Block-first**: each block is traced into per-body fragments, then each body's
  fragments are fused into one skeleton. A block is a bounded array, where a large body's
  bounding box is not.

**Why a separate package.** Both tracers build on `kimimaro` and `dijkstra3d`, which are
GPL-3.0-or-later, so this package is too. neu-morpho and the rest of
[neu-suite](https://github.com/flatiron-connectomics/neu-suite) are Apache-2.0. The
dependency runs one way: neu-skel imports neu-morpho, and nothing in neu-suite imports
neu-skel. Reading, viewing and measuring skeletons needs only neu-suite:
`neu_vol.readback`, `neu-morpho measure skeletons`, neu-draw and neu-glance all work
on skeletons however they were made.

## Install

neu-skel needs the neu-suite packages, which are not yet on PyPI. In an environment set
up from neu-suite's `environment.yml`:

```bash
git clone https://github.com/flatiron-connectomics/neu-suite
git clone https://github.com/flatiron-connectomics/neu-skel
pip install --no-deps -e ./neu-suite/neu-lib -e ./neu-suite/blockrun \
    -e ./neu-suite/neu-vol -e ./neu-suite/neu-morpho
pip install kimimaro dijkstra3d
pip install --no-deps -e ./neu-skel
python -m pytest -q neu-skel
```

## Running it

`neu-skel run` runs only the `skel` stage. Index and mesh are `neu-morpho run`'s, and
both commands take the same `--src`, `--dst` and `--work-dir`:

```bash
neu-morpho run --src /path/to/seg \
    --dst /path/to/morpho/segmentation --work-dir /path/to/morpho \
    --serial --roi 0,0,0,512,2048,2048 --roi-scale 2 --stages index,mesh --min-voxels 1000
neu-skel run --src /path/to/seg \
    --dst /path/to/morpho/segmentation --work-dir /path/to/morpho \
    --serial --roi 0,0,0,512,2048,2048 --roi-scale 2
```

Its options are those of `neu-morpho run` that apply to skeletons (ROI, block, body
selection, occupancy, dask), plus `--skel-scale`, `--tracer` and `--neutu-cost`. Bodies
are chosen as in neu-morpho:

- `--min-voxels` / `--limit-bodies` select from the index's voxel counts, so the work
  dir must hold an index. The allowlist they write there is reused by a later run that
  passes no selection, so the example's `neu-skel run` skeletonizes the meshed bodies.
- `--allowlist FILE` needs no index, and is not copied into the work dir: pass it to
  both commands.
- `--meshed-bodies` takes the bodies that have a mesh at `--dst`, by listing the mesh
  directory, so skeletons cover what was meshed however it was selected. Unsharded
  meshes only; for a sharded source, pass `--allowlist`.
- With none of these and no allowlist in the work dir, every body is skeletonized.

A body that fails is logged, listed in `<work-dir>/failures.skel.jsonl` and retried by
the next run; the run then exits 1, or 0 with `--allow-body-failures`, for a pipeline that
should carry on. An aborted stage always exits non-zero.

`neu-morpho progress <work-dir>` and `neu-morpho run-report <work-dir>` report the skel
stage beside the others. From Python, the stage is a runner for `neu_morpho.pipeline`:

```python
from neu_morpho import pipeline
from neu_skel import run_stage

p = pipeline.plan(src, dst, work_dir=work, stages=("skel",), extra_stages={"skel": 2})
summary = pipeline.run(p, runners={"skel": run_stage})
```

**A possible later option**: `neu-morpho run` could discover an installed neu-skel (a
plugin entry point) and offer the `skel` stage itself, so one command does everything.
It is not done, because that would load GPL code into neu-morpho's process, which is the
licensing question this split exists to avoid.

[`kimimaro`]: https://github.com/seung-lab/kimimaro

## License

©2026 The Simons Foundation, Inc.

Licensed under the [GNU General Public License v3.0 or later](LICENSE). See
[CONTRIBUTING.md](CONTRIBUTING.md) for how contributions are licensed.
