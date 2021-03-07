#!/usr/bin/env python3

import numpy as np


def flip_xyz_zyx_convention(array):
    """
    Given an Nx6 array, swap values in columns 1 and 3 and swap values in
    columns 4 and 6. This converts xyz-ordered point pairs to be zyx-ordered,
    or zyx-order point pairs to be xyz-ordered.
    """
    assert array.shape[1] == 6
    array[:, 0:3] = array[:, 2::-1]
    array[:, 3:6] = array[:, 5:2:-1]


def flip_pre_post_order(array):
    """
    Given an Nx6 array, swap columns 1-3 with columns 4-6.
    """
    assert array.shape[1] == 6
    tmp = array[:, 0:3].copy()
    array[:, 0:3] = array[:, 3:6]
    array[:, 3:6] = tmp


def load(fn, verbose=True):
    """
    Given a filename pointing to a file containing synaptic links, load them
    and return them as an Nx6 numpy array.
    """
    if fn.endswith('.npy'):
        if verbose: print('Mode 1: npy')
        # For opening .npy files saved from np.save
        links = np.load(fn)

        # The .npy files Jasper generated on Feb 8 were saved in zyx, so flip them to xyz
        if True:
            flip_xyz_zyx_convention(links)
        # The .npy files Jasper generated on Feb 8 were saved in post-pre order
        if True:
            flip_pre_post_order(links)

        # The .npy files Jasper generated on Feb 8 are saved in nm, so convert to
        # units of voxels at (4, 4, 40) nm voxel size for easier entering into ng
        links = links / (4, 4, 40, 4, 4, 40)

        #links is now pre-post, xyz, in units of voxels at (4, 4, 40)nm

    elif fn.endswith('.csv'):
        if verbose: print('Mode 2: csv')
        # For opening ground truth annotation files
        links = np.genfromtxt(fn, delimiter=',', skip_header=1, dtype=np.uint16)

        # Ground truth annotations were saved in zyx, so flip them to xyz
        if True:
            flip_xyz_zyx_convention(links)

        # Ground truth annotations were saved in nm, so convert to units of voxels
        # at (4, 4, 40) nm voxel size for easier entering into ng
        links = links / (4, 4, 40, 4, 4, 40)

        #links is now pre-post, xyz, in units of voxels at (4, 4, 40)nm

    else:
        if verbose: print('Mode 3: binary')
        # For opening binary files saved by ../detection/worker.py
        links = np.fromfile(fn, dtype=np.int32).reshape(-1, 6)

        # The Feb 7 predictions were saved in post-pre order
        if True:
            flip_pre_post_order(links)

        # The Feb 7 predictions are in units of mip1 voxels ((8.6, 8.6, 45) nm)
        # so convert to mip0 voxels for easier entering into ng
        links = links * (2, 2, 1, 2, 2, 1) + np.array([1,1,0,1,1,0])

        #links is now pre-post, xyz, in units of voxels at (4.3, 4.3, 45)nm

    return links
