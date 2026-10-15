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
    assert plan["tracer"] == "neutu" and plan["scales"]["skel"] == 0

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

    from neu_morpho.metrics_db import MetricsDB

    db = MetricsDB(str(work / "metrics.db"), read_only=True)
    cable = dict(db.con.execute("SELECT body_id, cable_length_nm FROM bodies"))
    db.close()
    assert cable[7] > 300 and cable[9] is None          # 9 (1728 voxels) is filtered out


def test_a_size_filter_without_an_index_is_refused(tmp_path):
    from neu_skel import cli

    src = _source(tmp_path)
    with pytest.raises(SystemExit, match="metrics.db does not exist"):
        cli.main(["run", "--src", src, "--dst", str(tmp_path / "out" / "seg"),
                  "--serial", "--min-voxels", "10", "--occupancy-scale", "-1"])
