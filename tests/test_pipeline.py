"""`neu-skel run`: the skel stage on neu-morpho's pipeline, end to end."""

import json

import numpy as np
import pytest

from neu_morpho import pipeline
from neu_skel import run_stage


def _source(tmp_path):
    """A two-level precomputed segmentation: a rod across blocks and a blob."""
    from neu_vol import convert
    from neu_vol.backends.tensorstore import TensorStoreBackend
    from neu_vol.profiles import zarr3_create_spec

    vol = np.zeros((64, 64, 64), np.uint64)
    vol[4:60, 30:36, 30:36] = 7
    vol[10:22, 8:20, 8:20] = 9
    raw = str(tmp_path / "raw.zarr")
    be = TensorStoreBackend.create(
        zarr3_create_spec("local", raw, vol.shape, "uint64",
                          dimension_names=("z", "y", "x"), chunk=(16, 16, 16)),
        delete_existing=True)
    be.write_region(tuple(slice(0, s) for s in vol.shape), vol)
    src = str(tmp_path / "seg")
    convert(raw, src, voxel_size=(8, 8, 8), kind="segmentation",
            profile="local-neuroglancer", chunk=(16, 16, 16), factors=[(2, 2, 2)],
            min_dim=8, delete_existing=True)
    return src


def test_run_adds_skeletons_to_the_volume_and_the_db(tmp_path):
    src = _source(tmp_path)
    work = tmp_path / "out"
    p = pipeline.plan(src, str(work / "seg"), stages=("seg", "index", "skel"),
                      index_scale=0, mesh_scale=0, extra_stages={"skel": 0},
                      block=(32, 32, 32), occupancy_scale=-1, extra={"tracer": "neutu"})
    summary = pipeline.run(p, runners={"skel": run_stage}, command="test")

    assert summary["n_failed_bodies"] == 0
    assert set(summary["stages"]) == {"seg", "index", "skel"}
    assert set(summary["linked_subresources"]) == {"skeletons"}
    assert "skeletons" in json.loads((work / "seg" / "info").read_text())
    plan = json.loads((work / "run_plan.json").read_text())
    assert plan["stages"]["skel"]["tracer"] == "neutu" and plan["stages"]["skel"]["scale"] == 0

    from neu_morpho.metrics_db import MetricsDB

    db = MetricsDB(str(work / "metrics.db"), read_only=True)
    rows = dict(db.con.execute("SELECT body_id, cable_length_nm FROM bodies"))
    db.close()
    assert set(rows) == {7, 9} and rows[7] > 300         # the rod is ~56 voxels of 8 nm


def test_the_db_columns_are_neu_morphos_definition(tmp_path):
    """The skel stage writes the columns `neu-morpho measure` reads back, by one function."""
    from neu_morpho.measure.skeletons import _measure_one
    from neu_morpho.precomputed import write_body_skeleton, write_skeleton_info
    from osteoid import Skeleton

    from neu_skel.stage import _db_columns

    v = np.array([[i * 100.0, 0, 0] for i in range(5)], np.float32)
    e = np.array([[i, i + 1] for i in range(4)], np.uint32)
    skel = Skeleton(v, e, radii=np.array([20.0, 90.0, 30.0, 10.0, 40.0], np.float32),
                    segid=100)
    vol = str(tmp_path / "vol")
    write_skeleton_info(f"{vol}/skeleton")
    write_body_skeleton(f"{vol}/skeleton", 100, skel)
    _, status, measured = _measure_one(100, volume=vol, skeleton_dir="skeleton",
                                       require_radii=True)
    assert status == "written"
    assert _db_columns(skel) == pytest.approx(measured)
    assert measured["max_radius_nm"] == pytest.approx(90.0)


def test_the_cli_plans_the_skel_stage(tmp_path, capsys):
    from neu_skel import cli

    src = _source(tmp_path)
    assert cli.main(["run", "--src", src, "--dst", str(tmp_path / "out" / "seg"),
                     "--serial", "--dry-run", "--skel-scale", "1", "--block", "32,32,32",
                     "--occupancy-scale", "-1"]) == 0


@pytest.mark.parametrize("flag", ["--stages", "--mesh-scale", "--index-scale"])
def test_the_cli_runs_only_the_skel_stage(flag):
    from neu_skel import cli

    with pytest.raises(SystemExit):
        cli._parse_args(["run", "--src", "x", "--dst", "y", flag, "0"])


def test_skeletons_follow_a_neu_morpho_index_in_the_same_work_dir(tmp_path):
    """The two-command workflow: the skel run reuses the index run's size filter."""
    from neu_morpho import cli as morpho_cli

    from neu_skel import cli

    src = _source(tmp_path)
    work = tmp_path / "out"
    common = ["--src", src, "--dst", str(work / "seg"), "--serial", "--block", "32,32,32",
              "--occupancy-scale", "-1"]
    assert morpho_cli.main(["run", *common, "--stages", "seg,index",
                            "--index-scale", "0", "--mesh-scale", "0",
                            "--min-voxels", "2000"]) == 0
    assert cli.main(["run", *common, "--skel-scale", "0"]) == 0

    summary = json.loads((work / "run_summary.json").read_text())
    assert list(summary["stages"]) == ["seg", "index", "skel"]
    assert summary["linked_subresources"] == {"skeletons": "skeleton"}
    # 9 (1728 voxels) is filtered out; unfiltered, both bodies are fused
    assert summary["stages"]["skel"]["num_bodies_fused"] == 1

    from neu_morpho.metrics_db import MetricsDB

    db = MetricsDB(str(work / "metrics.db"), read_only=True)
    cable = dict(db.con.execute("SELECT body_id, cable_length_nm FROM bodies"))
    db.close()
    assert cable[7] > 300


def test_a_size_filter_without_an_index_is_refused(tmp_path):
    from neu_skel import cli

    src = _source(tmp_path)
    with pytest.raises(SystemExit, match="metrics.db does not exist"):
        cli.main(["run", "--src", src, "--dst", str(tmp_path / "out" / "seg"),
                  "--serial", "--min-voxels", "10", "--occupancy-scale", "-1"])


def test_the_cli_defaults_are_neu_morphos():
    """The shared options are declared here too; their defaults must not drift apart."""
    import inspect

    from neu_skel import cli

    plan = {k: v.default for k, v in inspect.signature(pipeline.plan).parameters.items()}
    args = cli._parse_args(["run", "--src", "x", "--serial"])
    ours = {k: getattr(args, k) for k in ("occupancy_scale", "occupancy_dilate", "min_voxels",
                                          "limit_bodies", "max_consecutive_failures",
                                          "allowlist", "roi_scale")}
    ours.update(block=tuple(int(v) for v in args.block.split(",")),
                metrics_db=not args.no_metrics_db, resume=not args.no_resume)
    assert ours == {k: plan[k] for k in ours}


def test_meshed_bodies_skeletonizes_what_neu_morpho_meshed(tmp_path):
    """A mesh run with an explicit allowlist, then --meshed-bodies: the same bodies."""
    pytest.importorskip("vol2mesh")
    from neu_morpho import cli as morpho_cli

    from neu_skel import cli

    src = _source(tmp_path)
    work = tmp_path / "out"
    common = ["--src", src, "--dst", str(work / "seg"), "--serial", "--block", "32,32,32",
              "--occupancy-scale", "-1"]
    only = tmp_path / "only.csv"
    only.write_text("body_id\n7\n")
    assert morpho_cli.main(["run", *common, "--stages", "seg,mesh", "--mesh-scale", "0",
                            "--allowlist", str(only)]) == 0
    assert cli.main(["run", *common, "--skel-scale", "0", "--meshed-bodies"]) == 0

    assert (work / "allowlist.meshed.csv").read_text() == "body_id\n7\n"
    skel = json.loads((work / "run_summary.json").read_text())["stages"]["skel"]
    assert skel["num_bodies_fused"] == 1                 # unselected, 9 is fused too


def test_skeletons_for_new_bodies_in_a_finished_work_dir_are_refused(tmp_path):
    """Stage 1 traced only the listed bodies, so a resume cannot add one."""
    from neu_morpho.ops.manifest import SelectionChanged

    from neu_skel import cli

    src = _source(tmp_path)
    work = tmp_path / "out"
    common = ["--src", src, "--dst", str(work / "seg"), "--serial", "--block", "32,32,32",
              "--occupancy-scale", "-1", "--skel-scale", "0", "--no-metrics-db"]
    only = tmp_path / "only.csv"
    only.write_text("7\n")
    assert cli.main(["run", *common, "--allowlist", str(only)]) == 0
    only.write_text("9\n")
    with pytest.raises(SelectionChanged, match="e.g. 9"):
        cli.main(["run", *common, "--allowlist", str(only)])
    # what the message says to do instead
    assert cli.main(["run", *common, "--allowlist", str(only),
                     "--work-dir", str(tmp_path / "other")]) == 0
    other = json.loads((tmp_path / "other" / "run_summary.json").read_text())
    assert other["stages"]["skel"]["num_bodies_fused"] == 1


@pytest.mark.parametrize("extra, match", [
    (["--allowlist", "a.csv"], None),
    (["--min-voxels", "5"], "drop --min-voxels"),
])
def test_meshed_bodies_excludes_other_selections(extra, match):
    from neu_skel import cli

    with pytest.raises(SystemExit, match=match):
        cli._parse_args(["run", "--src", "x", "--dst", "y", "--serial", "--meshed-bodies",
                         *extra])


def test_meshed_bodies_without_meshes_is_refused(tmp_path):
    from neu_skel import cli

    src = _source(tmp_path)
    with pytest.raises(SystemExit, match="no meshes under"):
        cli.main(["run", "--src", src, "--dst", str(tmp_path / "out" / "seg"), "--serial",
                  "--meshed-bodies", "--skel-scale", "0", "--occupancy-scale", "-1"])


@pytest.mark.parametrize("failed, flag, code", [
    (0, [], 0), (2, [], 1), (2, ["--allow-body-failures"], 0),
])
def test_failed_bodies_exit_1_unless_allowed(monkeypatch, failed, flag, code):
    from neu_skel import cli

    monkeypatch.setattr(pipeline, "plan", lambda *a, **k: object())
    monkeypatch.setattr(pipeline, "run", lambda p, **k: {"n_failed_bodies": failed})
    assert cli.main(["run", "--src", "x", "--dst", "y", "--serial", *flag]) == code
