#!/usr/bin/env python3

import sys
import os

import numpy as np
import tqdm

import navis
import flybrains
from caveclient import CAVEclient
from meshparty import trimesh_io
from FANC_auto_recon.transforms.template_alignment import warp_points_FANC_to_template
import npimage # pip install git+https://github.com/jasper-tms/npimage
import npimage.graphics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Make sure template_spaces' folder is on path
from template_spaces import template_info, get_nrrd_metadata


show_help = """\
    Render a neuron that has been reconstructed in FANC into an image stack aligned to a VNC template.
    The rendered neuron stack can be used for generation of a depth-colored MIP for use in mask searching.

    Usage:
      ./render_neuron_into_template_space.py seg_id [name_of_template_space]

    seg_id must be a segment ID from the FANC production segmentation.
    name_of_template_space must be the name of a template space. The options
      for this can be found in FANC_auto_recon/lm_em_comparisons/template_spaces.py
      If omitted, a default of 'JRC2018_VNC_UNISEX_461' will be used.

    Example usage:
      ./render_neuron_into_template_space.py 648518346494405175
    Output file will be named segid648518346494405175_in_JRC2018_VNC_UNISEX_461.nrrd

    WARNING: This script can take 10+ minutes to run on large neurons, as
      large neurons can have meshes with many millions of faces.
      A small neuron (~100k faces) that can be used for testing is 648518346516214999
"""


def render_mesh_into_template_space(seg_id: int, target_space: str, header_pixels=0):
    if target_space not in template_info.keys():
        raise ValueError(
            'target_space was {} but must be one of: '
            '{}'.format(target_space, list(template_info.keys()))
        )
    target_info = template_info[target_space]

    # Setup
    client = CAVEclient('fanc_production_mar2021')
    mm = trimesh_io.MeshMeta(
        cv_path=client.info.segmentation_source(),
        disk_cache_path=os.path.expanduser('~/.meshes'),
        map_gs_to_https=True
    )

    print('Downloading mesh')
    my_mesh = mm.mesh(
        seg_id=seg_id,
        remove_duplicate_vertices=True,
        merge_large_components=False  # False is faster but probably worse quality
    )

    # Remove any mesh faces in the neck connective or brain,
    # since those can't be warped to the VNC template
    y_cutoff = 322500 + 1e-4  # y=75000vox * 4.3nm/vox, plus a small epsilon
    # Find row numbers of vertices that are out of bounds
    out_of_bounds_vertices = (my_mesh.vertices[:, 1] < y_cutoff).nonzero()[0]
    in_bounds_faces = np.isin(my_mesh.faces,
                              out_of_bounds_vertices,
                              invert=True).all(axis=1)
    my_mesh.faces = my_mesh.faces[in_bounds_faces]

    if target_space.startswith('JRC2018_VNC_FEMALE'):
        print('Warping into alignment with JRC2018_VNC_FEMALE')
        my_mesh.vertices = navis.xform_brain(my_mesh.vertices, source='FANC', target='JRCVNC2018F')
    elif target_space.startswith('JRC2018_VNC_UNISEX'):
        print('Warping into alignment with JRC2018_VNC_UNISEX')
        my_mesh.vertices = navis.xform_brain(my_mesh.vertices, source='FANC', target='JRCVNC2018U')
    elif target_space.startswith('JRC2018_VNC_MALE'):
        print('Warping into alignment with JRC2018_VNC_MALE')
        my_mesh.vertices = navis.xform_brain(my_mesh.vertices, source='FANC', target='JRCVNC2018M')
    else:
        raise ValueError('Could not determine target from: {}'.format(target_space))

    # Convert from microns to pixels in the target space
    my_mesh.vertices = my_mesh.vertices / target_info['voxel size']

    # Render into a target-space-sized numpy array
    print('Rendering mesh faces')
    rendered_image = np.zeros(target_info['stack dimensions'], dtype=np.uint8)
    for face in tqdm.tqdm(my_mesh.faces):
        npimage.graphics.drawtriangle(
            rendered_image,
            my_mesh.vertices[face[0]],
            my_mesh.vertices[face[1]],
            my_mesh.vertices[face[2]],
            255,
            fill_value=255
        )

    if header_pixels:
        dims = rendered_image.shape
        header = np.zeros((dims[0], header_pixels, dims[2]), dtype=np.uint8)
        rendered_image = np.concatenate((header, rendered_image), axis=1)

    npimage.save(rendered_image,
                 'segid{}_in_{}.nrrd'.format(seg_id, target_space),
                 metadata=get_nrrd_metadata(target_space),
                 dim_order='xyz')


def main():
    if len(sys.argv) == 1:
        print(show_help)
        return

    seg_id = int(sys.argv[1])

    if len(sys.argv) >= 3:
        template_space_name = sys.argv[2]
    else:
        template_space_name = 'JRC2018_VNC_UNISEX_461'

    render_mesh_into_template_space(seg_id, template_space_name)


if __name__ == '__main__':
    main()
