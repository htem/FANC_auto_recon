#!/usr/bin/env python3

import os
from datetime import datetime, timezone
from xml.etree import ElementTree

import numpy as np
import requests
import cloudvolume

from . import auth, lookup, transforms

PUBLISHED_MESHES_CLOUDVOLUME = ('gs://lee-lab_female-adult-nerve-cord/'
                                'meshes/{}/FANC_neurons')
PUBLISHED_MESHES_GCLOUDPROJECT = 'prime-sunset-531'
VALID_TEMPLATE_SPACES = ('FANC', 'JRC2018_VNC_FEMALE',
                         'JRC2018_VNC_UNISEX', 'JRC2018_VNC_MALE')


def list_public_segment_ids(template_space='JRC2018_VNC_FEMALE',
                            cloudvolume_path=PUBLISHED_MESHES_CLOUDVOLUME):
    """
    List the segment IDs of all neurons that have been published to the
    specified template space.
    """
    if template_space not in VALID_TEMPLATE_SPACES:
        raise ValueError('{} not in {}'.format(template_space,
                                               VALID_TEMPLATE_SPACES))
    cloudvolume_path = cloudvolume_path.format(template_space)
    # Implementation aided by GPT-4
    url = f'https://storage.googleapis.com/{cloudvolume_path.split("/")[2]}'
    params = {'prefix': '/'.join(cloudvolume_path.split('/')[3:])}

    segids = []
    while True:
        response = requests.get(url, params=params)
        tree = ElementTree.fromstring(response.text)

        # namespaces cause trouble, so find them and remove them
        namespaces = {'ns': 'http://doc.s3.amazonaws.com/2006-03-01'}

        for elem in tree.findall('ns:Contents/ns:Key', namespaces):
            filename = elem.text.split('/')[-1]
            if filename.endswith(':0'):
                segids.append(np.int64(filename[:-2]))

        # Each query can only get 500 neurons, so we need to check whether
        # there are more results to fetch
        marker_elem = tree.find('ns:NextMarker', namespaces)
        if marker_elem is None:
            break
        # If there are, add the marker to the params and allow the while loop
        # to continue, making a request for the next batch
        params['marker'] = marker_elem.text

    return segids


def publish_mesh_to_gcloud(segids,
                           template_space='JRC2018_VNC_FEMALE',
                           cloudvolume_path=PUBLISHED_MESHES_CLOUDVOLUME,
                           link_to_cellid=True,
                           timestamp='now'):
    """
    Download the mesh for a neuron, warp it into alignment with the specified
    VNC template (optional), and upload it to a public google cloud storage bucket.
    Neurons uploaded to the public project will have the same ID as the source
    neuron they came from in FANC.

    Currently only admins like Jasper and Wei have the necessary permissions
    for the upload to succeed. Please ask one of them to run this function for
    you when you have a list of neurons you're ready to make public.

    Get a complete list of IDs of public neurons by running the function
    `list_public_segment_ids()`, or by opening one of the the following
    links in your browser, depending on which space you want to see neurons in:
    https://console.cloud.google.com/storage/browser/lee-lab_female-adult-nerve-cord/meshes/FANC/FANC_neurons/meshes
    https://console.cloud.google.com/storage/browser/lee-lab_female-adult-nerve-cord/meshes/JRC2018_VNC_FEMALE/FANC_neurons/meshes
    Not implemented yet: https://console.cloud.google.com/storage/browser/lee-lab_female-adult-nerve-cord/meshes/JRC2018_VNC_UNISEX/FANC_neurons/meshes
    Not implemented yet: https://console.cloud.google.com/storage/browser/lee-lab_female-adult-nerve-cord/meshes/JRC2018_VNC_MALE/FANC_neurons/meshes

    These neurons can be viewed by entering their ID(s) into one of the following neuroglancer links:
    FANC: https://ng.fanc.community/published-neurons-viewer
    JRC2018_VNC_FEMALE: https://ng.fanc.community/published-neurons-aligned-to-template-viewer
    JRC2018_VNC_UNISEX: Not implemented yet
    JRC2018_VNC_MALE: Not implemented yet

    Parameters
    ----------
    segids: int or iterable of ints
      The segment ID(s) of the neuron(s) to be published.

    template_space: str
      The name of the template space to which the neurons should be aligned
      before being published. Must be 'JR2018_VNC_FEMALE' or 'FANC'.

    cloudvolume_path: str, default set by publish.PUBLISHED_MESHES_CLOUDVOLUME
      A path to a CloudVolume (typically on google cloud storage) where
      the published neurons should be uploaded. The path should contain
      a placeholder {} for the template_space.

    link_to_cellid: bool, default True
      Whether to add an alias to the mesh in the cloudvolume that allows
      the mesh to be loaded by the cell ID of the neuron it represents.

    timestamp: 'now' (default) OR datetime OR None
      The timestamp at which to query the segment's publication
      status and cell ID.
      If 'now', use the current time.
      If datetime, use the time specified by the user.
      If None, use the timestamp of the latest materialization.
    """
    try:
        iter(segids)
    except:
        segids = [segids]

    if template_space not in VALID_TEMPLATE_SPACES:
        raise ValueError('{} not in {}'.format(template_space,
                                               VALID_TEMPLATE_SPACES))

    already_published_ids = list_public_segment_ids(template_space=template_space,
                                                    cloudvolume_path=cloudvolume_path)
    already_published_ids = set(already_published_ids)

    fancneurons_cloudvolume = cloudvolume.CloudVolume(cloudvolume_path.format(template_space))

    if timestamp == 'now':
        timestamp = datetime.now(timezone.utc)
    mm = auth.get_meshmanager()
    for segid in segids:
        if segid in already_published_ids:
            print(f'Segment {segid} already published in {template_space}-space, skipping.')
            continue
        print(f'Publishing segment {segid} to {template_space}-space.')
        mesh = mm.mesh(seg_id=segid)
        if template_space != 'FANC':
            transforms.template_alignment.align_mesh(mesh, target_space=template_space)
            mesh.vertices *= 1000  # TODO delete this after adding nm/um to align_mesh
        mesh = cloudvolume.mesh.Mesh(mesh.vertices, mesh.faces, segid=segid)
        fancneurons_cloudvolume.mesh.put(mesh)
        del mesh
        if link_to_cellid:
            add_cellid_alias(segid,
                             template_space=template_space,
                             cloudvolume_path=cloudvolume_path,
                             force=True,
                             timestamp=timestamp)


def add_cellid_alias(segid,
                     template_space='JRC2018_VNC_FEMALE',
                     cloudvolume_path=PUBLISHED_MESHES_CLOUDVOLUME,
                     force=False,
                     timestamp='now'):
    return add_mesh_alias(
        segid,
        lookup.cellid_from_segid(segid, timestamp=timestamp),
        cloudvolume_path.format(template_space) + '/meshes',
        force=force
    )


def add_mesh_alias(meshid,
                   alias,
                   gcloud_mesh_folder,
                   gcloud_project=PUBLISHED_MESHES_GCLOUDPROJECT,
                   force=False):
    """
    Add an alias to a mesh in the google cloud storage bucket.

    Parameters
    ---------
    meshid: int or iterable of ints
      The mesh ID(s) for which the alias should be added.

    alias: int or iterable of ints
      The alias(es) to be added to the mesh(es).
      (Technically this can be a string or any object that can be
      converted to a string, but neuroglancer only knows how to load
      meshes by integer IDs.)

    gcloud_mesh_folder: str
      The path to the folder in the google cloud storage bucket where
      the mesh(es) are stored.

    force: bool
      Whether to overwrite the alias if it already exists.
    """
    from google.cloud import storage
    from google.cloud.exceptions import NotFound
    gcs_client = storage.Client(project=gcloud_project)
    bucket = gcs_client.get_bucket(gcloud_mesh_folder.replace('gs://', '').split('/')[0])
    mesh_folder = '/'.join(gcloud_mesh_folder.replace('gs://', '').split('/')[1:])

    try:
        iter(meshid)
        meshids = meshid
    except TypeError:
        meshids = [meshid]
    try:
        iter(alias)
        if isinstance(alias, str):
            raise TypeError
        aliases = alias
    except TypeError:
        aliases = [alias]

    for meshid, alias in zip(meshids, aliases):
        blob = bucket.blob(f'{mesh_folder}/{alias}:0')
        try:
            if force:
                # Pretend the file doesn't exist
                raise NotFound('')
            blob.reload()
            print(f'Blob {blob.name} already exists, skipping.')
        except NotFound:
            try:
                bucket.blob(f'{mesh_folder}/{meshid}:0:1').reload()
                print(f'Mesh {meshid} exists. Adding alias named {alias}.')
                blob.upload_from_string(
                    f'{{"fragments":["{meshid}:0:1"]}}',
                    content_type='application/json'
                )
            except NotFound:
                print(f'Mesh {meshid} does not exist. Cannot add alias named {alias}.')


def publish_all_meshes(published_tag='publication',
                       tag_location=('neuron_information', 'tag2'),
                       template_space='JRC2018_VNC_FEMALE',
                       cloudvolume_path=PUBLISHED_MESHES_CLOUDVOLUME,
                       timestamp='now',
                       n=None):
    """
    Copy to a public location the meshes of all neurons marked as
    published.

    Parameters
    ---------
    published_tag: str
      The tag used to mark neurons as published.

    tag_location: tuple
      A 2-tuple of (CAVE table name, column name) indicating the annotation
      table column where the published_tag should be looked for.

    template_space: str
      The name of the template space to which the neurons should be aligned
      before being published. Must be 'JR2018_VNC_FEMALE' or 'FANC'.

    cloudvolume_path: str
      A path to a CloudVolume (typically on google cloud storage) where
      the published neurons should be uploaded. The path should contain
      a placeholder {} for the template_space.

    timestamp: 'now' (default) OR datetime OR None
      The timestamp at which to query the segment's publication
      status and cell ID.
      If 'now', use the current time.
      If datetime, use the time specified by the user.
      If None, use the timestamp of the latest materialization.
    """
    segids_with_published_annotation = lookup.cells_annotated_with(
        published_tag,
        source_tables=[tag_location],
        timestamp=timestamp
    )
    publish_mesh_to_gcloud(segids_with_published_annotation[:n],
                           template_space=template_space,
                           cloudvolume_path=cloudvolume_path,
                           timestamp=timestamp)


def publish_skeleton_to_catmaid(segids,
                                catmaid_instance=None):
    """
    Download the mesh for a neuron, skeletonize it, upload it to one
    catmaid project, warp the skeleton into alignment with the female VNC
    template, and upload that to a different catmaid project
    """
    raise NotImplementedError


def publish_to_bcio(cave_token):
    """
    Rerun the export procedure on BrainCircuits.io to export published root_ids to
    the the `fruitfly_fanc_public` project. The function may take several minutes
    to complete. It should return `Export successful` message.

    Exported files are accessible at:
    https://api.braincircuits.io/data/fruitfly_fanc_public/
    """
    import requests
    return requests.get(f'https://api.braincircuits.io/publish/dataset?project=fruitfly_fanc_public&cave_token={cave_token}')


def _configure_template_cloudvolumes(template_space='JRC2018_VNC_FEMALE'):
    """
    This function was run once to configure some cloudvolumes to hold published
    neurons. Only admins like Jasper and Wei have the necessary permissions on
    the google cloud project for this function to complete successfully.
    The code is included here in case we need to do something similar again.
    """
    print('Uploading {}_4iso.nrrd image data'.format(template_space))
    import npimage  # pip install numpyimage
    # Load VNC template from local file (you must have this file in this folder)
    # Use xyz order because that's what cloudvolume wants
    template_image_data = npimage.load('{}_4iso.nrrd'.format(template_space),
                                       dim_order='xyz')
    template_image_data_8bit = npimage.to_8bit(template_image_data)

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
