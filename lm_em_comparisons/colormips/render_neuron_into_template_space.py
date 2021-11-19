#!/usr/bin/env python3

import sys
import os

import numpy as np
import tqdm

from caveclient import CAVEclient
from meshparty import trimesh_io
from FANC_auto_recon.transforms.template_alignment import warp_points_FANC_to_template
import npimage # pip install git+https://github.com/jasper-tms/npimage
import npimage.graphics


def show_help():
    m = ('Render a neuron that has been reconstructed in FANC into an image'
         ' stack aligned to the JRC2018_VNC_FEMALE template.\n'
         'The rendered neuron stack can be used for a number of things,'
         ' including generation of a depth-colored MIP for use in mask'
         ' searching.\n'
         'Example usage:\n'
         './render_segid_in_template_space.py 648518346494405175\n'
         'Output would be named 648518346494405175.nrrd\n'
         'WARNING: Meshes often have millions of faces, so this can take a few'
         ' minutes per neuron.')
    print(m)



def main():
    if len(sys.argv) == 1:
        show_help()
        return

    seg_id = int(sys.argv[1])

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

    print('Warping to template')
    my_mesh.vertices = warp_points_FANC_to_template(my_mesh.vertices, input_units='nanometers', output_units='microns')


    # Render into template-sized numpy array

    # JRC2018_VNC_FEMALE.nrrd is 660x1342x358  (in xyz order) at 0.4 micron
    # voxel size>

    # TODO figure out the image size, voxel size, and offset used in the images
    # made available in the Janelia MCFO collections, and render with those
    # settings.

    print('Rendering mesh faces')
    template_vol = np.zeros((660, 1342,  358), dtype=np.uint8)
    voxel_size = 0.4  # 0.4 microns
    my_mesh.vertices = my_mesh.vertices / voxel_size
    n = my_mesh.n_faces
    #for i, face in enumerate(my_mesh.faces)):
    for face in tqdm.tqdm(my_mesh.faces):
        #print(i, '/', n)
        npimage.graphics.drawtriangle(
            template_vol,
            my_mesh.vertices[face[0]],
            my_mesh.vertices[face[1]],
            my_mesh.vertices[face[2]],
            255,
            fill_value=255
            #voxel_size=0.4)
        )

    npimage.save(template_vol, '{}.nrrd'.format(seg_id), dim_order='xyz')

if __name__ == '__main__':
    main()
