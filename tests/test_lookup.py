#!/usr/bin/env python3

from datetime import datetime, timezone

import numpy as np

import fanc


def test_lookup():
    print('fanc.lookup: Start tests')
    pts = np.array([[48848, 114737, 2690],
                    [49198, 114622, 2690]])
    assert fanc.lookup.svids_from_pts(list(pts[0, :])) == [73679924787396631]
    assert fanc.lookup.svids_from_pts(tuple(pts[0, :])) == [73679924787396631]
    assert fanc.lookup.svids_from_pts(pts) == [73679924787396631, 73750224812092331]

    jan2022 = datetime(2022, 1, 1, tzinfo=timezone.utc)
    assert all(fanc.lookup.segids_from_pts(pts, timestamp=jan2022) == np.array([648518346494405175, 648518346502740211], dtype=np.uint64))
    assert all(fanc.lookup.segids_from_pts(pts) != np.array([648518346494405175, 648518346502740211], dtype=np.uint64))
    print('fanc.lookup: PASS')


def test_false():
    assert 0 == 1


if __name__ == '__main__':
    test_lookup()
    #test_false()
    print('All tests passed')

