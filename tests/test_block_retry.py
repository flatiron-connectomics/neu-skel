"""The skel stage-1 worker retries a transient store error instead of failing the stage."""

import numpy as np

from blockrun import iter_blocks

#: What a worker saw when DNS failed mid-run (curl_code=6), trimmed.
TRANSIENT = ('UNAVAILABLE: Error opening "neuroglancer_precomputed" driver: CURL error '
             'Could not resolve hostname [curl_code=\'6\']')


def test_the_skel_worker_retries_a_transient_read(monkeypatch, tmp_path):
    import neu_vol.backends.base as base
    import neu_vol.retry as retry

    from neu_skel.config import SkeletonConfig
    from neu_skel.stage import _chunk_block

    vol = np.zeros((32, 32, 32), np.uint64)
    vol[4:28, 14:18, 14:18] = 7
    reads = []

    class Flaky:
        def read_region(self, region):
            reads.append(region)
            if len(reads) == 1:
                raise ValueError(TRANSIENT)
            return vol[region]

    monkeypatch.setattr(retry.time, "sleep", lambda s: None)
    monkeypatch.setattr(base, "open_backend", lambda spec: Flaky())
    (block,) = iter_blocks(vol.shape, (32, 32, 32))
    index, status = _chunk_block(block, seg_spec={}, chunked_dir=str(tmp_path),
                                 cfg=SkeletonConfig(block_shape=(32, 32, 32)), allow=None)
    assert len(reads) == 2 and status == "written"
