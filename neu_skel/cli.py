"""neu-skel: skeletons per body, added to a neu-morpho run, locally or on SLURM.

    neu-skel run ...      # the skel stage, in a neu-morpho work dir

``run`` traces skeletons per body and writes them into the volume at --dst, beside the
meshes, with their cable metrics in the work dir's metrics DB. Index and mesh are
``neu-morpho run``'s; point both commands at the same --src, --dst and --work-dir:

    neu-morpho run --src S --dst D --work-dir W --stages index,mesh --min-voxels 1000
    neu-skel run   --src S --dst D --work-dir W

--min-voxels and --limit-bodies select bodies from the index's voxel counts, so they need
the index run, and the allowlist they write in W is reused by any later run passing none
of the three. An explicit --allowlist is not copied into W: pass it to both commands.
``neu-morpho progress W`` and ``neu-morpho run-report W`` cover the skel stage too.

The skel stage has two phases, fragments per (body, block) and then one skeleton fused
per body, with its own manifest; a re-run reuses the fragments already on disk.
"""

from __future__ import annotations

import argparse
import logging

from neu_morpho.cli import add_run_args, check_run_args, run_main


def build_parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="neu-skel", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="skeletons per body, in a neu-morpho work dir",
                       description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p, stages=("skel",))
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
    p.set_defaults(func=_run)
    top._run_parser = p
    return top


def _parse_args(argv=None):
    top = build_parser()
    args = top.parse_args(argv)
    check_run_args(top._run_parser, args)
    return args


def _run(args) -> int:
    import functools

    from .stage import run_stage

    runner = functools.partial(run_stage, tracer=args.tracer, neutu_cost=args.neutu_cost)
    return run_main(args, extra_stages={"skel": args.skel_scale}, runners={"skel": runner},
                    extra={"tracer": args.tracer, "neutu_cost": args.neutu_cost},
                    label="neu-skel")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    from neu_vol.logs import quiet_store_logs

    with quiet_store_logs(not args.store_logs):
        return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
