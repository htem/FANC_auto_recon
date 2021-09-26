#!/usr/bin/env python3

# Warp mesh files from JRC2018_VNC template space to FANC space
# Currently only works with stl files, need to write support for obj

import sys
import os

import stl  # pip install numpy-stl

# Modify the line below if you keep your repositories on a different path
sys.path.append(os.path.expanduser('~/repos/FANC_auto_recon/transforms'))
from template_alignment import warp_points_template_to_FANC


def show_help():
    print('Run via: ./warp_mesh_to_FANC.py some_mesh.stl folder_to_put_output_in/')


def main():
    if len(sys.argv) == 1:
        show_help()
        return

    # Argument validation
    try:
        input_filename = sys.argv[1]
        if not os.path.exists(input_filename):
            print(show_help)
            raise FileNotFoundError(input_filename)

        output_folder = sys.argv[2]
        if not os.path.exists(output_folder):
            print(show_help)
            raise FileNotFoundError(output_folder)
    except:
        show_help()
        raise

    if not input_filename.endswith('.stl'):
        raise NotImplementedError('Currently I only know how to open .stl format meshes')

    # Load
    mesh = stl.mesh.Mesh.from_file(input_filename)
    # Do the warping
    mesh.v0 = warp_points_template_to_FANC(mesh.v0,
                                           input_units='nm',
                                           output_units='nm')
    mesh.v1 = warp_points_template_to_FANC(mesh.v1,
                                           input_units='nm',
                                           output_units='nm')
    mesh.v2 = warp_points_template_to_FANC(mesh.v2,
                                           input_units='nm',
                                           output_units='nm')
    # Save
    mesh.save(output_folder + '/' + os.path.basename(input_filename))


if __name__ == "__main__":
    main()
