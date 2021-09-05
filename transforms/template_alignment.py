#!/usr/bin/env python3

# Originally published in https://www.lee.hms.harvard.edu/phelps-hildebrand-graham-et-al-2021
# and made available in that paper's repository located at:
# https://github.com/htem/GridTape_VNC_paper/tree/main/template_registration_pipeline/register_EM_dataset_to_template
# Development of this script is continuing here, not in the GridTape_VNC_paper repository


# TODO
# - Implement input_units='voxels' and output_units='voxels' assuming a
#    template voxel size of [400, 400, 400] nm and a FANC voxel size of [4.3,
#    4.3, 45] nm.
# - Assert that input_units and output_units are one of 'nm', 'nanometer',
#    'nanometers', 'um', 'µm', 'micron', 'microns', 'pixels', 'voxels'
# - Implement an interactive __main__ function where users can manually input
#    the coordinates of a single point they want to warp.
# - Test that reflect=True/False in warp_points_template_to_FANC now works for
#    single-point arguments

import os
import subprocess
import numpy as np

import transformix  # https://github.com/jasper-tms/pytransformix


template_plane_of_symmetry_x_voxel = 329
template_plane_of_symmetry_x_microns = 329 * 0.400

def warp_points_FANC_to_template(points,
                                 input_units='nm',
                                 output_units='microns',
                                 reflect=False):
    points = np.array(points, dtype=np.float64)
    if len(points.shape) == 1:
        return warp_points_FANC_to_template(np.expand_dims(points, 0),
                                            input_units, output_units)[0]
    if input_units == 'nm' and (points < 1000).all():
        resp = input('Your points appear to be in microns, not nm. Want to'
                     ' change input_units from nm to microns? [y/n] ')
        if resp.lower() == 'y':
            input_units = 'microns'
    if input_units in ['um', 'microns']:
        points *= 1000  # Convert microns to nm

    points -= (533.2, 533.2, 945)  # (1.24, 1.24, 2.1) vox at (430, 430, 450)nm/vox
    points /= (430, 430, 450)
    points *= (300, 300, 400)
    points[:, 2] = 435*400 - points[:, 2]  # z flipping a stack with 436 slices
    points /= 1000  # Convert nm to microns

    transform_params = os.path.join(
        os.path.dirname(__file__),
        'transform_parameters',
        'TransformParameters.FixedFANC.txt'
    )
    points = transformix.transform_points(points, transform_params)

    if not reflect:
        points[:, 0] = template_plane_of_symmetry_x_microns * 2 - points[:, 0]
    if output_units == 'nm':
        points *= 1000  # Convert microns to nm

    return points


def warp_points_template_to_FANC(points,
                                 input_units='nm',
                                 output_units='microns',
                                 reflect=False):
    points = np.array(points)
    if len(points.shape) == 1:
        return warp_points_template_to_FANC(np.expand_dims(points, 0),
                                            input_units=input_units,
                                            output_units=output_units,
                                            reflect=reflect)[0]
    if input_units == 'nm' and (points < 1000).all():
        resp = input('Your points appear to be in microns, not nm. Want to'
                     ' change input_units from nm to microns? [y/n] ')
        if resp.lower() == 'y':
            input_units = 'microns'
    if input_units == 'nm':
        points /= 1000  # Convert nm to microns

    if not reflect:
        points[:, 0] = template_plane_of_symmetry_x_microns * 2 - points[:, 0]
    transform_params = os.path.join(
        os.path.dirname(__file__),
        'transform_parameters',
        'TransformParameters.FixedTemplate.Bspline.txt'
    )
    points = transformix.transform_points(points, transform_params)

    points *= 1000  # Convert microns to nm
    points[:, 2] = 435*400- points[:, 2]  # z flipping a stack with 436 slices
    points /= (300, 300, 400)
    points *= (430, 430, 450)
    points += (533.2, 533.2, 945)  # (1.24, 1.24, 2.1) vox at (430, 430, 450)nm/vox

    if output_units in ['um', 'microns']:
        points /= 1000  # Convert nm to microns
    return points


if __name__ == "__main__":
    pass
