#!/usr/bin/env python3

# Jasper Phelps
# Initial commit Feb 10, 2021

# Provide utilities related to the 9 synapse-containing cubes of image data
# (that had all synapses manually annotated) and 11 synapse-free cubes of image
# data that were used for training and evaluation of automated synapse
# detection networks using the method from Buhmann et al. bioRxiv 2020
# (https://www.biorxiv.org/content/10.1101/2019.12.12.874172v2) for FANC.

# NOTE THAT ALL COORDINATES ARE SPECIFIED IN ZYX ORDER AND IN UNITS OF
# NANOMETERS, ASSUMING A VOXEL SIZE OF (40, 4, 4) nm.

# First functions in this script are related to getting the names of the
# different cutouts and their ground truth synaptic link annotations.

# Subsequent functions are related to checking whether certain coordinates are
# within certain regions of the FANC synapse ground truth cutouts.
#
# Each cutout contains a central region, usually the region 2 microns
# (sometimes 1 micron) away from all edges of the cutout, where synapses were
# annotated. This is the 'annotated' or 'full' region. Within the annotated
# region, the lower 75% of slices (smaller z coordinates) were used for
# training the synapse prediction network, and the upper 25% of the slices
# (larger z coordinates) were held out for validation.
#
# The 'train50' (first 50%) and 'test' (50%-75%)regions were not used, but are
# included anyway in this script in case they're ever needed.


import sys
import os
import json

import numpy as np


_synapse_cutouts = [f'synapse_cutout{i}' for i in [2, 3, 4, 6, 7, 8, 9, 10, 11]]
_no_synapse_cutouts = [f'no_synapse_cutout{i}' for i in [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12]]
cutout_names = _synapse_cutouts + _no_synapse_cutouts


def print_cutout_names():
    print(cutout_names)

def get_cutout_names():
    return cutout_names

def get_empty_cutout_names():
    return _no_synapse_cutouts

def get_annotated_cutout_names():
    return _synapse_cutouts


#default_base_path = (
#    '/n/groups/htem/temcagt/datasets/vnc1_r066/'
#    'synapsePrediction+templateAlignment/0_synapsePrediction/'
#    'synapse_ground_truth')
default_base_path = os.path.join(os.path.dirname(__file__),
                                 'ground_truth_link_annotations')
default_csv_name_format = '{}_link_annotations.csv'
def load_annotations(cutout_name,
                     validation_region='full',
                     validation_target='postsynapse',
                     base_path=default_base_path,
                     csv_name_format=default_csv_name_format):
    """
    Load the ground truth synaptic links annotated in a requested cutout.
    Returns: Nx6 numpy array representing N synaptic link annotations, with
    6 columns representing pre_z, pre_y, pre_x, post_z, post_y, post_x
    coordinates in nanometers (assuming a voxel size of [40, 4, 4] nm).
    """
    if cutout_name in _no_synapse_cutouts:
        print('WARNING: You requested annotations for a synapse-free cutout. Was this intended?')
        return np.array([], dtype=np.uint16)
    if isinstance(cutout_name, int):
        cutout_name = 'synapse_cutout{}'.format(cutout_name)
    assert cutout_name in _synapse_cutouts, cutout_name + ' is not a valid cutout name.'

    fn = os.path.join(base_path, csv_name_format.format(cutout_name))
    assert os.path.exists(fn), 'No file found at ' + fn
    print('Loading annotations for {}'.format(cutout_name))
    links = np.genfromtxt(fn, delimiter=',', skip_header=1, dtype=np.uint16)
    if validation_region is not None:
        if validation_target == 'postsynapse':
            valid = in_region(links[:, 3:6], cutout_name, validation_region)
        elif validation_target == 'presynapse':
            valid = in_region(links[:, 0:3], cutout_name, validation_region)
        elif validation_target == 'both':
            valid_post = in_region(links[:, 3:6], cutout_name, validation_region)
            valid_pre  = in_region(links[:, 0:3], cutout_name, validation_region)
            valid = np.logical_and(valid_pre, valid_post)
        else:
            raise ValueError("validation_target must be either 'postsynapse'"
                             " or 'presynapse' but was " + validation_target)
        print(f'Found {sum(valid)} in-region points out of {len(links)}'
              f' (validation_region: {validation_region}, validation_target:'
              f' {validation_target})')
        #print('Invalid points:')
        #print(links[~valid, 5:2:-1])  # Print in xyz order for pasting into catmaid
        # I checked the invalid points in CATMAID to make sure that they
        # actually corresponded to annotations that were slightly outside the
        # annotated ROI, and indeed they all did. Couldn't confirm that every
        # invalid point was found but I saw invalid points on low and high x,
        # low and high y, and low and high z, so pretty sure this in_region
        # function works as intended.
        links = links[valid, :]
    return links

def load_all_annotations(**kwargs):
    """
    See load_annotations() for kwargs options
    """
    return {cutout: load_annotations(cutout, **kwargs)
            for cutout in _synapse_cutouts}


def load_segmentation(cutout_name):
    """
    Returns the segmentation for a synapse cutout.
    Segmentation is returned as a dict with 3 entries:
        'data': numpy array containing segment IDs for each voxel.
        'voxel_size': list of 3 values representing the voxel size.
        'voxel_offset': list of 3 values representing the voxel offset.
    """
    if cutout_name in _no_synapse_cutouts:
        raise ValueError('Only synapse-containing cutouts have segmentations'
                         f' but you asked for {cutout_name}')
    if isinstance(cutout_name, int):
        cutout_name = 'synapse_cutout{}'.format(cutout_name)
    assert cutout_name in _synapse_cutouts, cutout_name + ' is not a valid cutout name.'

    from cloudvolume import CloudVolume
    path = 'gs://zetta_lee_fly_vnc_001_synapse_cutout/{}/seg_medium_cube'
    print('Downloading segmentation for ' + cutout_name)
    vol = CloudVolume(path.format(cutout_name), use_https=True)
    mip0_info = vol.info['scales'][0]
    #print(mip0_info)
    seg = vol[:]
    seg = seg.squeeze().T  # CloudVolumes are xyzc. Want zyx for this script
    return {
        'data': seg,
        'voxel_size': mip0_info['resolution'][::-1],
        'voxel_offset': mip0_info['voxel_offset'][::-1]
    }


def load_all_segmentations():
    return {name: load_segmentation(name)
            for name in _synapse_cutouts}


def in_annotated_region(pts: np.array, cutout: str):
    """
    Given an Nx3 numpy array representing zyx-ordered points in units of nm,
    test which points are within the full annotated region of the named cutout.
    Call 'print_cutout_names()' for a list of valid cutout names.
    Returns: length N np.array of booleans
    """
    start = rois[cutout]['full']['start']
    end = rois[cutout]['full']['end']
    return in_roi(pts, start, end)


def in_training_region(pts: np.array, cutout: str):
    """
    Given an Nx3 numpy array representing zyx-ordered points in units of nm,
    test which points are within the training region of the named cutout (lower
    75% of z slices).
    Call 'print_cutout_names()' for a list of valid cutout names.
    Returns: length N np.array of booleans
    """
    start = rois[cutout]['train75']['start']
    end = rois[cutout]['train75']['end']
    return in_roi(pts, start, end)


def in_validation_region(pts: np.array, cutout: str):
    """
    Given an Nx3 numpy array representing zyx-ordered points in units of nm,
    test which points are within the validation region of the named cutout
    (upper 25% of the z slices).
    Call 'print_cutout_names()' for a list of valid cutout names.
    Returns: length N np.array of booleans
    """
    start = rois[cutout]['validation']['start']
    end = rois[cutout]['validation']['end']
    return in_roi(pts, start, end)

def in_region(pts: np.array, cutout: str, region_name: str):
    """
    Given an Nx3 numpy array representing zyx-ordered points in units of nm,
    test which points are within the given region of the named cutout.
    Call 'print_cutout_names()' for a list of valid cutout names.
    Returns: length N np.array of booleans
    """
    assert region_name in rois[cutout], (
        region_name + ' is not a valid region name for ' + cutout
        + '. Options are ' + str(list(rois[cutout].keys()))
    )
    start = rois[cutout][region_name]['start']
    end = rois[cutout][region_name]['end']
    return in_roi(pts, start, end)


def in_roi(pts: np.array, start: tuple, end: tuple) -> np.array:
    """
    Given an Nx3 numpy array representing zyx-ordered points in units of nm,
    test which points are within a box whose start (lower-left) and end
    (upper-right) points are given.
    Returns: length N np.array of booleans
    """
    pts = np.array(pts)
    if len(pts.shape) == 1:
        pts = pts[np.newaxis, :]
    #problems = pts[pts[:, 0] == 5000, :]
    #print('problems:', problems)
    #print(problems < end)
    return np.logical_and((pts >= start).all(axis=1), (pts < end).all(axis=1))


rois_fn = os.path.join(os.path.dirname(__file__),
                       'synapse_cutout_rois.json')
def generate_rois(save_path=rois_fn):
    import daisy  #github.com/funkelab/daisy, use 0.3-dev branch
    rois = {}
    special_case_dealt_with = False
    for cutout in cutout_names:
        rois[cutout] = {}

        config_f = 'cutout_configs/' + cutout + '_config.json'
        assert os.path.exists(config_f), f'{config_f} does not exist'
        with open(config_f) as f:
            config = json.load(f)

        voxel_size = daisy.Coordinate(config['CatmaidIn']['voxel_size'])
        annotated_shape = daisy.Coordinate(config['CatmaidIn']['roi_shape_nm'])
        annotated_shape_context = daisy.Coordinate(config['CatmaidIn']['roi_context_nm'])
        annotated_shape_full = annotated_shape + annotated_shape_context * 2#+ annotated_shape_context
        roi = daisy.Roi((0, 0, 0), annotated_shape_full)

        try:
            roi_context = daisy.Coordinate(tuple(config["CatmaidIn"]["effective_roi_context_nm"]))
        except:
            roi_context = daisy.Coordinate(tuple(config["CatmaidIn"]["roi_context_nm"]))
        roi = roi.grow(-roi_context, -roi_context)
        full_shape = roi.get_shape()

        if cutout == 'synapse_cutout11':  # Synapses were only annotated in first half of slices for this cube
            special_case_dealt_with = True
            full_shape = daisy.Coordinate((full_shape[0]/2+20, full_shape[1], full_shape[2]))
            assert full_shape == full_shape / voxel_size * voxel_size
            #print('full_shape changed to {} for cutout11'.format(full_shape))
        full_offset = roi.get_offset()

        masks = {
            'full':       (0, 1),

            'train75':    (0, 0.75),

            'train50':    (0, 0.5),
            'test':       (0.5, 0.75),

            'validation': (0.75, 1),
        }
        for mask_name in masks:
            mask_z_range = masks[mask_name]
            mask_shape = daisy.coordinate.Coordinate((full_shape[0]*(mask_z_range[1] - mask_z_range[0]),
                                                      full_shape[1],
                                                      full_shape[2]))
            mask_offset = daisy.coordinate.Coordinate((full_offset[0] + full_shape[0]*mask_z_range[0],
                                                       full_offset[1],
                                                       full_offset[2]))
            # Rounding to a multiple of voxel size
            mask_top_corner_rounded = (mask_shape + mask_offset) / voxel_size * voxel_size 
            mask_offset_rounded = mask_offset / voxel_size * voxel_size
            mask_shape_rounded = mask_top_corner_rounded - mask_offset_rounded

            mask_roi = daisy.Roi(offset=mask_offset_rounded, shape=mask_shape_rounded)
            rois[cutout][mask_name] = {}
            rois[cutout][mask_name]['start'] = tuple(mask_roi.get_offset())
            rois[cutout][mask_name]['shape'] = tuple(mask_roi.get_shape())
            rois[cutout][mask_name]['end'] = tuple(mask_roi.get_end())

    assert special_case_dealt_with
    #print(rois)
    with open(save_path, 'w') as f:
        json.dump(rois, f, indent=2)


if not os.path.exists(rois_fn):
    try:
        generate_rois(save_path=rois_fn)
    except ModuleNotFoundError:
        print(f'ROIs could not be loaded from {rois_fn}, and generate_rois() '
              'failed. Install github.com/funkelab/daisy/tree/0.3-dev or get '
              'the ROIs file from github.com/htem/FANC_auto_recon/blob/master'
              '/synapses/ground_truth/synapse_cutout_rois.json.')
        raise

with open(rois_fn, 'r') as f:
    rois = json.load(f)
