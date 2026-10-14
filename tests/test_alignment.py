"""Skeletons land in the meshes' physical-nm space, before and after encoding."""

import numpy as np
import pytest

from neu_morpho.coords import crop_origin_nm, physical_box, skeleton_to_physical


def test_mesh_and_skeleton_coincide():
    """A body at a global offset: its skeleton lies inside its mesh bbox. Dropping the
    crop origin would leave the skeleton at local coordinates."""
    pytest.importorskip("vol2mesh")
    import kimimaro
    from vol2mesh import Mesh

    vs = (8, 8, 8)
    origin_vox = (10, 10, 10)                                  # crop starts at global voxel 10
    crop = np.zeros((16, 6, 6), np.uint64)
    crop[:, 2:5, 2:5] = 7                                      # 3x3 rod along z

    # mesh: region is the crop's global extent -> nm box
    region = (slice(10, 26), slice(10, 16), slice(10, 16))
    box = physical_box(region, vs)                            # z:[80,208] y,x:[80,128]
    mesh = Mesh.from_label_volume(crop, box, labels=[7], ensure_halo=True, progress=False)[7]
    m_lo, m_hi = mesh.vertices_zyx.min(0), mesh.vertices_zyx.max(0)

    # skeleton: crop-local, then shifted to global nm by the crop origin
    skels = kimimaro.skeletonize(crop, anisotropy=vs, object_ids=[7],
                                 teasar_params={"scale": 1.5, "const": 30, "pdrf_scale": 100000,
                                                "pdrf_exponent": 4}, dust_threshold=0, progress=False)
    sv = skeleton_to_physical(skels[7].vertices, crop_origin_nm(origin_vox, vs))
    s_lo, s_hi = sv.min(0), sv.max(0)

    # skeleton lies within the mesh bbox (a small margin for surface vs centerline)
    assert (s_lo >= m_lo - 8).all() and (s_hi <= m_hi + 8).all(), (m_lo, m_hi, s_lo, s_hi)
    # z extent is the global rod (~80..208), not the local one (~0..128)
    assert s_lo[0] >= 72 and s_hi[0] > 180


def test_mesh_and_skeleton_agree_after_encoding():
    """After encoding to stored xyz, a skeleton still lands inside its mesh bbox."""
    pytest.importorskip("vol2mesh")
    from osteoid import Skeleton
    from vol2mesh import Mesh

    from neu_morpho.precomputed import encode_skeleton
    from neu_skel.config import SkeletonConfig
    from neu_skel.skeletonize import skeletonize_block

    vs = (8.0, 8.0, 8.0)
    # asymmetric placement, so a z<->x swap cannot pass by accident
    block = np.zeros((24, 16, 40), np.uint64)
    block[4:20, 6:10, 30:34] = 7

    box = physical_box((slice(0, 24), slice(0, 16), slice(0, 40)), vs)
    mesh = Mesh.from_label_volume(block, box, labels=[7], ensure_halo=True, progress=False)[7]
    m_lo_xyz = mesh.vertices_zyx.min(0)[::-1]
    m_hi_xyz = mesh.vertices_zyx.max(0)[::-1]

    cfg = SkeletonConfig(anisotropy=vs, const=30.0, dust_threshold=0)
    skel = skeletonize_block(block, (0, 0, 0), cfg)[7]
    sv = Skeleton.from_precomputed(encode_skeleton(skel)).vertices

    assert (sv.min(0) >= m_lo_xyz - 8).all() and (sv.max(0) <= m_hi_xyz + 8).all(), \
        (m_lo_xyz, m_hi_xyz, sv.min(0), sv.max(0))
    # x is the narrow axis (~240..272 nm) and z the long one (~32..160): a flip bug swaps them
    assert sv[:, 0].min() > 200 and sv[:, 2].max() < 200
