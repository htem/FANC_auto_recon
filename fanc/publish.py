#!/usr/bin/env python3

import os

import cloudvolume

from . import auth, transforms

PUBLISHED_MESHES_CLOUDPATH = ('gs://lee-lab_female-adult-nerve-cord/'
                              'VNC_templates/{}/FANC_neurons')
VALID_TEMPLATE_SPACES = ('JRC2018_VNC_FEMALE', 'JRC2018_VNC_UNISEX',
                         'JRC2018_VNC_MALE')


def publish_mesh_to_gcloud(segids,
                           template_space='JRC2018_VNC_FEMALE',
                           gcloud_path=PUBLISHED_MESHES_CLOUDPATH):
    """
    Download the mesh for a neuron, warp it into alignment with the specified
    VNC template, and upload it to a public google cloud storage bucket.
    Neurons uploaded to the public project will have the same ID as the source
    neuron they came from in FANC.

    Currently only admins like Jasper and Wei have the necessary permissions
    for the upload to succeed. Please ask one of them to run this function for
    you when you have a list of neurons you're ready to make public.

    The complete list of public neurons can be seen at:
    https://console.cloud.google.com/storage/browser/lee-lab_female-adult-nerve-cord/VNC_templates/JRC2018_VNC_FEMALE/FANC_neurons/meshes

    Public neurons can be viewed by entering their ID into this neuroglancer state:
    https://neuromancer-seung-import.appspot.com/?json_url=https://global.daf-apis.com/nglstate/api/v1/5850904959909888
    """
    try:
        iter(segids)
    except:
        segids = [segids]

    if template_space not in VALID_TEMPLATE_SPACES:
        raise ValueError('{} not in {}'.format(template_space,
                                               VALID_TEMPLATE_SPACES))
    
    fancneurons_cloudvolume = cloudvolume.CloudVolume(gcloud_path.format(template_space))

    mm = auth.get_meshmanager()
    for segid in segids:
        mesh = mm.mesh(seg_id=segid)
        transforms.template_alignment.align_mesh(mesh, target_space=template_space)
        mesh.vertices *= 1000  # TODO delete this after adding nm/um to align_mesh
        mesh = cloudvolume.mesh.Mesh(mesh.vertices, mesh.faces, segid=segid)
        fancneurons_cloudvolume.mesh.put(mesh)


def publish_skeleton_to_catmaid(segids,
                                catmaid_instance=None):
    """
    Download the mesh for a neuron, skeletonize it, upload it to one
    catmaid project, warp the skeleton into alignment with the female VNC
    template, and upload that to a different catmaid project
    """
    raise NotImplementedError


def _configure_template_cloudvolumes(template_space='JRC2018_VNC_FEMALE'):
    """
    This function was run once to configure some cloudvolumes to hold published
    neurons. Only admins like Jasper and Wei have the necessary permissions on
    the google cloud project for this function to complete successfully.
    The code is included here in case we need to do something similar again.
    """
    print('Uploading {}_4iso.nrrd image data'.format(template_space))
    import npimage  # pip install numpyimage
    import npimage.operations
    # Load VNC template from local file (you must have this file in this folder)
    # Use xyz order because that's what cloudvolume wants
    template_image_data = npimage.load('{}_4iso.nrrd'.format(template_space),
                                       dim_order='xyz')
    template_image_data_8bit = npimage.operations.to_8bit(template_image_data)

    # Open a CloudVolume
    gcloud_path = ('gs://lee-lab_female-adult-nerve-cord/VNC_templates/'
                   '{}/image/'.format(template_space))
    info = cloudvolume.CloudVolume.create_new_info(
        num_channels = 1,
        layer_type = 'image', # 'image' or 'segmentation'
        data_type = 'uint8', # can pick any popular uint
        encoding = 'jpeg', # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
        resolution = [400, 400, 400], # X,Y,Z values in nanometers
        voxel_offset = [0, 0, 0], # values X,Y,Z values in voxels
        chunk_size = [660, 1342, 1], # rechunk of image X,Y,Z in voxels
        volume_size = [660, 1342, 358], # X,Y,Z size in voxels
    )
    template_image_cloudvolume = cloudvolume.CloudVolume(gcloud_path, info=info)
    template_image_cloudvolume.provenance.description = (
        '{}_4iso.nrrd from https://www.janelia.org/open-science/jrc-2018-brain-templates'
        ' converted from 16-bit to 8-bit.'.format(template_space)
    )
    template_image_cloudvolume.provenance.owners = ['jasper.s.phelps@gmail.com']
    template_image_cloudvolume.commit_info()
    template_image_cloudvolume.commit_provenance()

    # Upload image data
    template_image_cloudvolume[:] = template_image_data_8bit

    print('Making cloudvolumes to hold meshes')
    volumes_cloudpath = ('gs://lee-lab_female-adult-nerve-cord/VNC_templates/'
                         '{}/volume_meshes/'.format(template_space))
    info = cloudvolume.CloudVolume.create_new_info(
        num_channels = 1,
        layer_type = 'segmentation', # 'image' or 'segmentation'
        mesh = 'meshes',
        data_type = 'uint64', # can pick any popular uint
        encoding = 'raw', # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
        resolution = [400, 400, 400], # X,Y,Z values in nanometers
        voxel_offset = [0, 0, 0], # values X,Y,Z values in voxels
        chunk_size = [660, 1342, 1], # rechunk of image X,Y,Z in voxels
        volume_size = [660, 1342, 358], # X,Y,Z size in voxels
    )
    volumes_cv = cloudvolume.CloudVolume(volumes_cloudpath, info=info)
    volumes_cv.provenance.description = (
        'Meshes encompasing various regions of the VNC, including the whole '
        'VNC, the the whole VNC neuropil, and the different VNC neuropil '
        'regions defined in Court et al. 2020 Neuron. All are in the template '
        'space {}.'.format(template_space)
    )
    volumes_cv.provenance.owners = ['jasper.s.phelps@gmail.com']
    volumes_cv.commit_info()
    volumes_cv.commit_provenance()

    fancneurons_path = ('gs://lee-lab_female-adult-nerve-cord/VNC_templates/'
                        '{}/FANC_neurons/'.format(template_space))
    info = cloudvolume.CloudVolume.create_new_info(
        num_channels = 1,
        layer_type = 'segmentation', # 'image' or 'segmentation'
        mesh = 'meshes',
        data_type = 'uint64', # can pick any popular uint
        encoding = 'raw', # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
        resolution = [400, 400, 400], # X,Y,Z values in nanometers
        voxel_offset = [0, 0, 0], # values X,Y,Z values in voxels
        chunk_size = [660, 1342, 1], # rechunk of image X,Y,Z in voxels
        volume_size = [660, 1342, 358], # X,Y,Z size in voxels
    )
    fancneurons_cloudvolume = cloudvolume.CloudVolume(fancneurons_path, info=info)
    fancneurons_cloudvolume.provenance.description = (
        'Meshes of neurons reconstructed in FANC and warped into alignment with '
        '{}'.format(template_space)
    )
    fancneurons_cloudvolume.provenance.owners = ['jasper.s.phelps@gmail.com']
    fancneurons_cloudvolume.commit_info()
    fancneurons_cloudvolume.commit_provenance()


    print('Uploading volume outline meshes')
    meshes_folder = '../data/volume_meshes/{}'.format(template_space)
    volume_meshes = [
        ('tissueOutline_Aug2019.stl', 1),
        ('VNC_neuropil_Aug2020.stl', 2)
    ]
    # TODO add neuropil region meshes

    import trimesh.exchange
    for filename, segid in volume_meshes:
        with open(os.path.join(meshes_folder, filename), 'r') as f:
            mesh = trimesh.exchange.stl.load_stl(f)
        mesh = cloudvolume.mesh.Mesh(
            mesh['vertices'],
            mesh['faces'],
            segid=segid
        )
        volumes_cv.mesh.put(mesh)
