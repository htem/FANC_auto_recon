#!/usr/bin/env python3

import os

import numpy as np
import requests
import tqdm


def fanc4_to_3(points, scale=2, return_dict=False):
    '''
    Convert from realigned dataset to original coordinate space.
    Args:
        points:      an nx3 array of mip0 voxel coordinates
        scale:       selects the granularity of the field being used, but
                         does not change the units.
        return_dict: If false, returns an nx3 numpy array. If false,
                         returns a dict containing x, y, z, dx, and dy values.
    
    Returns:
        (default) an nx3 numpy array containing the transformed points
        (return_dict=True) a dictionary of transformed x/y/z values and the
            dx/dy values.
        Returned coordinates are floats (i.e. can have fractional values)
    '''
             
    base = 'https://spine.janelia.org/app/transform-service/dataset/fanc_v4_to_v3/s/{}/'.format(scale)

    n = len(np.shape(points))
                      
    if n == 1:
        full_url = (base + 'z/{}/'.format(str(int(points[2])))
                         + 'x/{}/'.format(str(int(points[0])))
                         + 'y/{}/'.format(str(int(points[1]))))
        r = requests.get(full_url)
    else:
        full_url = base + 'values_array'
        points_dict = {'x': list(points[:,0].astype(str)),
                       'y': list(points[:,1].astype(str)),
                       'z': list(points[:,2].astype(str))}
        r = requests.post(full_url, json=points_dict)

    try:
        r = r.json()
        if return_dict:
            return r
        else:
            if n == 1:
                return np.array((r['x'], r['y'], r['z']))
            else:
                return np.array((r['x'], r['y'], r['z'])).T
    except:
        return r


def fanc3_to_4(pts, mode='descent', precision=1.5, verbose=False):
    """
    Args:
        pts:       nx3 numpy array of xyz-ordered points
        mode:      'descent' or 'inversefield'
                   Use 'descent' if the v3 to v4 alignment field isn't available.
                   Use 'inversefield' if it is and you want faster (but
                       potentially less accurate?) mappings.
        precision: when in 'descent' mode, how accurate of an inverse must be
                   found before descent stops. In units of mip0 pixels.

    Returns:
        An nx3 numpy array containing the transformed points. Returned
        coordinates are ints. (Subpixel interpolation could be implemented for
        slightly more precision but hasn't been done yet.)
    """
    if isinstance(pts, (list, tuple)):
        pts = np.array(pts)

    if mode is 'descent':
        # Use this mode if the v3 to v4 alignment field hasn't been calculated yet.

        def mag(v):
            """Magnitude of a vector"""
            return np.sum(v**2)**0.5

        def find_inverse_by_descent(pt, precision, max_iterations=10):
            def report():
                print(i, pt, inv, fanc4_to_3(inv), error_vec)

            inv = pt.copy()
            error_vec = fanc4_to_3(inv) - pt
            i = 0
            if verbose: report()
            while mag(error_vec) > precision and i < max_iterations:
                inv = inv - error_vec
                error_vec = fanc4_to_3(inv) - pt
                i += 1
                if verbose: report()
            if i >= max_iterations:
                print('WARNING: Max iterations ({}) reached.'.format(max_iterations))

            return inv.astype(np.uint32)

        if len(pts.shape) == 1:  # Vector
            return find_inverse_by_descent(pts, precision=precision)
        elif len(pts.shape) == 2:  # Matrix
            # TODO TODO TODO write a version of find_inverse_by_descent that
            # parallelizes iterations for all given points to reduce http
            # requests. Currently the function is applied serially to each
            # point, which is slow.
            return np.apply_along_axis(find_inverse_by_descent, 1, pts, precision)

    elif mode is 'inversefield':
        # Use if the inverse field (v3->v4) has been calculated.
        # (This has not been done yet as of Apr 21, 2021)
        raise NotImplementedError

    else:
        raise ValueError("mode must be 'descent' or 'inversefield' but was {}".format(mode))


# Some example points to test
test_pts_v3 = np.array([
    [5207, 98758, 3028],
    [23540, 116496, 1228]
], dtype=np.uint32)
test_pts_v4 = np.array([
    [5286, 98834, 3028],
    [23622, 116520, 1228]
], dtype=np.uint32)

def test_fanc3_to_4(verbose=True):
    pts3to4 = fanc3_to_4(test_pts_v3, verbose=verbose)
    error = pts3to4 - test_pts_v4

    print('True v4 points:')
    print(test_pts_v4)
    print('Predicted v4 points:')
    print(pts3to4)
    print('Error:')
    print(error)

    # TODO add some assertion about the error being small, and a True/False
    # return value based on that


def test_fanc4_to_3():
    pts4to3 = fanc4_to_3(test_pts_v4)
    error = pts4to3 - test_pts_v3

    print('True v3 points:')
    print(test_pts_v3)
    print('Predicted v3 points:')
    print(pts4to3)
    print('Error:')
    print(error)

    # TODO add some assertion about the error being small, and a True/False
    # return value based on that


def test(verbose=True):
    pts_3_to_4_to_3 = fanc4_to_3(fanc3_to_4(test_pts_v3))
    error343 = np.subtract(pts_3_to_4_to_3, test_pts_v3, dtype=float)
    print('Error upon v3 -> v4 -> v3')
    print(error343)

    pts_4_to_3_to_4 = fanc3_to_4(fanc4_to_3(test_pts_v4))
    error434 = np.subtract(pts_4_to_3_to_4, test_pts_v4, dtype=float)
    print('Error upon v4 -> v3 -> v4')
    print(error434)
