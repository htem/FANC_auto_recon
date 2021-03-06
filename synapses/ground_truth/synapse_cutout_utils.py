#!/usr/bin/env python3

# Jasper Phelps
# Initial commit Feb 10, 2021

# Provide utilities related to the 9 synapse-containing cubes of image data
# that had all synapses manually annotated and 11 synapse-free cubes of image
# data that were used for training and evaluation of automated synapse
# detection networks based on Buhmann et al. bioRxiv 2020
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
#


import sys
import os
import json
from typing import List

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

# TODO some functions that return paths related to each cutout?

#default_base_path = (
#    '/n/groups/htem/temcagt/datasets/vnc1_r066/'
#    'synapsePrediction+templateAlignment/0_synapsePrediction/'
#    'synapse_ground_truth')
default_base_path = os.path.join(os.path.dirname(__file__),
                                 'ground_truth_link_annotations')
default_csv_name_format = '{}_link_annotations.csv'
def load_annotations(cutout_name: str,
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
    assert cutout_name in _synapse_cutouts, cutout_name + ' is not a valid cutout name.'

    fn = os.path.join(base_path, csv_name_format.format(cutout_name))
    assert os.path.exists(fn), 'No file found at ' + fn
    links = np.genfromtxt(fn, delimiter=',', skip_header=1, dtype=np.uint16)
    return links


def in_annotated_region(pts: 'np.array', cutout: str):
    """
    Given an n by 3 numpy array representing zyx-ordered points in units of nm,
    test which points are within the full annotated region of the named cutout.
    Call 'print_cutout_names()' for a list of valid cutout names.
    """
    start = rois[cutout]['full']['start']
    end = rois[cutout]['full']['end']
    return in_roi(pts, start, end)


def in_training_region(pts: 'np.array', cutout: str):
    """
    Given an n by 3 numpy array representing zyx-ordered points in units of nm,
    test which points are within the training region of the named cutout (lower
    75% of z slices).
    Call 'print_cutout_names()' for a list of valid cutout names.
    """
    start = rois[cutout]['train75']['start']
    end = rois[cutout]['train75']['end']
    return in_roi(pts, start, end)


def in_validation_region(pts: 'np.array', cutout: str):
    """
    Given an n by 3 numpy array representing zyx-ordered points in units of nm,
    test which points are within the validation region of the named cutout
    (upper 25% of the z slices).
    Call 'print_cutout_names()' for a list of valid cutout names.
    """
    start = rois[cutout]['validation']['start']
    end = rois[cutout]['validation']['end']
    return in_roi(pts, start, end)


def in_roi(pts: 'np.array', start: tuple, end: tuple):
    """
    Given an n by 3 numpy array representing zyx-ordered points in units of nm,
    test which points are within a box whose start (lower-left) and end
    (upper-right) points are given.
    """
    pts = np.array(pts)
    if len(pts.shape) == 1:
        pts = pts[np.newaxis, :]
    return (pts >= start).all(axis=1) and (pts < end).all(axis=1)


rois_fn = 'synapse_cutout_rois.json'
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
