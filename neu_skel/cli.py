"""neu-skel: neu-morpho's pipeline with a skeleton stage, locally or on SLURM.

    neu-skel run ...      # neu-morpho run, plus the skel stage

``run`` takes every argument of ``neu-morpho run`` (see ``neu-morpho help run``) and adds
``skel``: TEASAR skeletons per body, written into the same volume, metrics DB and run
summary as the meshes. Progress and reports come from neu-morpho:
``neu-morpho progress <work-dir>`` and ``neu-morpho run-report <work-dir>`` cover the
skel stage too.

    neu-skel run --src ... --dst /path/to/morpho/segmentation --work-dir /path/to/morpho \\
        --serial --roi 0,0,0,512,2048,2048 --roi-scale 2 --stages index,mesh,skel
"""

from __future__ import annotations

import argparse
import logging

from neu_morpho.cli import add_run_args, check_run_args, run_main, stages_help
from neu_morpho.pipeline import STAGE_DOC as MORPHO_STAGE_DOC

STAGE_DOC = {
    **MORPHO_STAGE_DOC,
    "skel": "skeletons, per body (the neutu tracer by default). Two phases — fragments "
            "per (body, block), then one skeleton fused per body — with its own manifest; "
            "a re-run with only this stage reuses the fragments.",
}


def build_parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="neu-skel", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="neu-morpho's pipeline, plus the skel stage",
                       description=f"{__doc__}\n{stages_help(stage_doc=STAGE_DOC)}",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_args(p, stages=tuple(STAGE_DOC), default_stages="index,mesh,skel")
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
