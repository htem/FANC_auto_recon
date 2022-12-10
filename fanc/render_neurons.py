#!/usr/bin/env python3

import sys
import os

import numpy as np
import tqdm
import navis
import flybrains
from meshparty import trimesh_io
import npimage
import npimage.graphics

from . import auth
from .transforms.template_alignment import warp_points_FANC_to_template
from .template_spaces import template_info, get_nrrd_metadata


def render_neuron_into_template_space(seg_id: int, target_space: str,
                                      header_pixels=0, skeletonize=False):
    """
    1. Create an image canvas aligned to a VNC template space (target_space)
    2. Render a template-aligned version of a neuron onto that canvas
    3. Save the resulting image volume as a .nrrd file

    seg_id :  The segment ID to render
    target_space :  See template_spaces.py
    header_pixels :  Add this many empty rows to the top of the image

    If skeletonize is False, the neuron's mesh will be rendered. This is slow
    (may involve rendering many millions of triangles) but gives the most
    accurate rendering.
    If skeletonize is True, the neuron's mesh will be skeletonized and then the
    skeleton will be rendered.
    """
    if target_space not in template_info.keys():
        raise ValueError(
            'target_space was {} but must be one of: '
            '{}'.format(target_space, list(template_info.keys()))
        )
    target_info = template_info[target_space]

    # Setup
    client = auth.get_caveclient()
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
