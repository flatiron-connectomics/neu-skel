"""neu-skel: skeletons per body, added to a neu-morpho run, locally or on SLURM.

    neu-skel run ...      # the skel stage, in a neu-morpho work dir

``run`` traces skeletons per body and writes them into the volume at --dst, beside the
meshes, with their cable metrics in the work dir's metrics DB. Index and mesh are
``neu-morpho run``'s; give both commands the same --src, --dst and --work-dir, and the
same --roi if there is one:

    neu-morpho run --src S --dst D --work-dir W --stages index,mesh --min-voxels 1000
    neu-skel run   --src S --dst D --work-dir W

--min-voxels and --limit-bodies select bodies from the index's voxel counts, so they need
the index run, and the allowlist they write in W is reused by any later run passing none
of the three. An explicit --allowlist is not copied into W: pass it to both commands, or use
--meshed-bodies, which takes the bodies that have a mesh at --dst.
``neu-morpho progress W`` and ``neu-morpho run-report W`` cover the skel stage too.

The skel stage has two phases, fragments per (body, block) and then one skeleton fused
per body, with its own manifest; a re-run reuses the fragments already on disk.
"""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger("neu-skel")


def build_parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="neu-skel", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="skeletons per body, in a neu-morpho work dir",
                       description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="segmentation volume (path or s3://...)")
    p.add_argument("--dst", help="the published precomputed volume the skeletons go "
                                 "into, as given to neu-morpho run")
    p.add_argument("--work-dir", help="the neu-morpho run directory. Defaults to the "
                                      "PARENT of a local --dst; required when --dst is "
                                      "remote")
    p.add_argument("--describe", action="store_true",
                   help="print the source pyramid and exit (no cluster, no writes)")

    p.add_argument("--skel-scale", type=int, default=2,
                   help="scale to skeletonize at (default: 2)")
    p.add_argument("--tracer", default="neutu", choices=("kimimaro", "neutu"),
                   help="stage-1 skeletonizer (default neutu). 'neutu' is "
                        "neu_skel.neutu_trace, which reproduces NeuTu's skeletons "
                        "with fewer vertices and requires isotropic voxels; 'kimimaro' "
                        "is the older tracer, and the one to use on an anisotropic "
                        "pyramid (see docs/skeletonization.md)")
    p.add_argument("--neutu-cost", default="edge", choices=("voxel", "edge"),
                   help="--tracer neutu only (default edge). 'edge' is NeuTu's own "
                        "symmetric cost and matches it more closely. 'voxel' is "
                        "cheaper in memory, which matters only if a component exceeds "
                        "SkeletonConfig.neutu_edge_max_gb (see docs/skeletonization.md)")

    p.add_argument("--roi", help="z0,y0,x0,z1,y1,x1 voxels, at --roi-scale")
    p.add_argument("--roi-scale", type=int, default=None,
                   help="scale the --roi values are given in; required with --roi")
    p.add_argument("--block", default="256,256,256", help="block shape (z,y,x) voxels")
    p.add_argument("--min-voxels", type=int, default=0,
                   help="only bodies with at least this many voxels, from the work "
                        "dir's index")
    p.add_argument("--limit-bodies", type=int, default=None,
                   help="only the N largest bodies, from the work dir's index")
    choose = p.add_mutually_exclusive_group()
    choose.add_argument("--allowlist", help="explicit body-id file (overrides --min-voxels)")
    choose.add_argument("--meshed-bodies", action="store_true",
                        help="the bodies with a mesh in --dst, listed into "
                             "<work-dir>/allowlist.meshed.csv. Unsharded meshes only")
    p.add_argument("--no-metrics-db", action="store_true",
                   help="do not write cable metrics to the work dir's metrics DB")
    p.add_argument("--occupancy-scale", type=int, default=5,
                   help="coarse scale used to skip empty blocks; -1 disables")
    p.add_argument("--occupancy-dilate", type=int, default=1,
                   help="grow the occupied set by N blocks; keep >=1, since a coarse "
                        "scale misses sparse blocks")

    p.add_argument("--config", action="append", default=None, metavar="NAME_OR_PATH",
                   help="dask config: a blockrun template (dask-local, "
                        "dask-slurm-example) or a YAML path. Repeatable, deep-merged "
                        "left to right. Default: dask-local")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--serial", action="store_true",
                   help="no dask at all — run in this process")
    p.add_argument("--no-resume", action="store_true", help="start over, ignoring manifests")
    p.add_argument("--max-consecutive-failures", type=int, default=10,
                   help="stop after this many per-body failures in a row (0 disables)")
    p.add_argument("--dry-run", action="store_true",
                   help="report the plan (scale, ROI, block count) and exit")
    p.add_argument("--store-logs", action="store_true",
                   help="keep TensorStore's benign S3 credential-chain logging")
    p.set_defaults(func=_run)
    return top


def _configs(args) -> list[str]:
    return args.config or ["dask-local"]


def _parse_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.meshed_bodies and (args.min_voxels or args.limit_bodies):
        raise SystemExit("--meshed-bodies chooses the bodies itself; drop --min-voxels "
                         "and --limit-bodies")
    if not args.serial:                       # a mistyped --config fails before any read
        from blockrun import load_config

        load_config(_configs(args))
    return args


def _run(args) -> int:
    import functools

    from neu_morpho import pipeline
    from neu_morpho.roi import parse_roi

    from .stage import run_stage

    if args.describe:
        from neu_vol import describe_scales

        print(describe_scales(args.src))
        return 0
    if not args.dst:
        raise SystemExit("--dst is required unless --describe")
    try:
        plan = pipeline.plan(
            args.src, args.dst, work_dir=args.work_dir, stages=("skel",),
            extra_stages={"skel": args.skel_scale},
            block=tuple(int(v) for v in args.block.split(",")),
            roi=parse_roi(args.roi), roi_scale=args.roi_scale,
            occupancy_scale=args.occupancy_scale, occupancy_dilate=args.occupancy_dilate,
            allowlist=args.allowlist, min_voxels=args.min_voxels,
            limit_bodies=args.limit_bodies, metrics_db=not args.no_metrics_db,
            max_consecutive_failures=args.max_consecutive_failures,
            resume=not args.no_resume,
            extra={"tracer": args.tracer, "neutu_cost": args.neutu_cost})
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if args.meshed_bodies:
        import dataclasses

        meshed = _meshed_allowlist(plan, write=not args.dry_run)
        plan = dataclasses.replace(plan, allowlist=meshed)
    if args.dry_run:
        log.info("--dry-run: nothing executed")
        return 0

    runners = {"skel": functools.partial(run_stage, tracer=args.tracer,
                                         neutu_cost=args.neutu_cost)}
    if args.serial:
        summary = pipeline.run(plan, runners=runners)
    else:
        from blockrun import start_dask

        with start_dask(args.workers, _configs(args), label="neu-skel",
                        effective_config_path=plan.work_dir) as client:
            summary = pipeline.run(plan, client=client, runners=runners)
    # Per-body failures do not stop the run; exit non-zero so a script notices them.
    return 1 if summary["n_failed_bodies"] else 0


def _meshed_allowlist(plan, *, write: bool) -> str:
    """List the bodies meshed at ``plan.dst`` into the work dir; returns the file's path."""
    import os

    from neu_vol import read_json
    from neu_vol.readback import UnsupportedSubresource, mesh_body_ids

    mesh_dir = (read_json(plan.dst, "info") or {}).get("mesh", "mesh")
    try:
        ids = mesh_body_ids(plan.dst, mesh_dir)
    except UnsupportedSubresource as e:
        raise SystemExit(f"--meshed-bodies: {e} Use --allowlist.") from None
    if not ids:
        raise SystemExit(f"--meshed-bodies: no meshes under {plan.dst}/{mesh_dir}")
    path = f"{plan.work_dir}/allowlist.meshed.csv"
    log.info("--meshed-bodies: %d bodies with a mesh -> %s", len(ids), path)
    if write:
        os.makedirs(plan.work_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write("body_id\n")
            f.writelines(f"{i}\n" for i in ids)
    return path


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    from neu_vol.logs import quiet_store_logs

    with quiet_store_logs(not args.store_logs):
        return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
