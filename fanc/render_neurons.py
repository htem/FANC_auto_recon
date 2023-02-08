#!/usr/bin/env python3

import sys
import os

import numpy as np
import tqdm
import navis
import flybrains
import npimage
import npimage.graphics

from . import auth
from .template_spaces import template_info, get_nrrd_metadata
from .transforms import template_alignment


def render_neuron_into_template_space(seg_id: int,
                                      target_space: str,
                                      skeletonize=False,
                                      compress=True):
    """
    Create an image volume in .nrrd format with dimensions matching a
    specified VNC template space, containing a rendering of a neuron
    from FANC aligned to that VNC template space.

    Arguments
    ---------
    seg_id : int
        The segment ID from the FANC segmentation to render
    target_space: str
       See template_spaces.py for a list of template spaces that can be
       provided for this argument
    skeletonize: bool (default False)
        If True, the skeletonized version of the neuron will be rendered.
        If False, the neuron's mesh will be rendered. This is slow because it
        may involve rendering many millions of triangles, but gives the most
        accurate rendering.
    compress: bool (default True)
        If True, save the .nrrd with gzip encoding. If False, save with raw
        encoding. (Because the image volume created here is all black except for
        a small percent of the pixels that are white to represent the neuron,
        compression gives file sizes <1MB where raw gives file sizes >100MB.)
    """
    if target_space not in template_info.keys():
        raise ValueError(
            'target_space was {} but must be one of: '
            '{}'.format(target_space, list(template_info.keys()))
        )
    target_info = template_info[target_space]

    # Setup
    client = auth.get_caveclient()
    meshmanager = auth.get_meshmanager()

    if skeletonize:
        raise NotImplementedError

    print('Downloading mesh')
    my_mesh = meshmanager.mesh(
        seg_id=seg_id,
        remove_duplicate_vertices=True,
        merge_large_components=False  # False is faster but probably worse quality
    )

    template_alignment.align_mesh(my_mesh, target_space)

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

    #if header_pixels:
    #    dims = rendered_image.shape
    #    header = np.zeros((dims[0], header_pixels, dims[2]), dtype=np.uint8)
    #    rendered_image = np.concatenate((header, rendered_image), axis=1)

    npimage.save(rendered_image,
                 'segid{}_in_{}.nrrd'.format(seg_id, target_space),
                 metadata=get_nrrd_metadata(target_space),
                 compress=compress,
                 dim_order='xyz')
