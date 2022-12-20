#!/usr/bin/env python3

import os
import json

import numpy as np
import pandas as pd
from matplotlib import cm, colors
import vtk
from meshparty import trimesh_vtk, trimesh_io, meshwork
try:
    from trimesh import exchange
except ImportError:
    from trimesh import io as exchange
import pymaid
from cloudvolume import CloudVolume
from cloudvolume.frontends.precomputed import CloudVolumePrecomputed
from nglui.statebuilder import *

from . import auth, catmaid, rootID_lookup
from .transforms import realignment


def skel2scene(skid, project=13, segment_threshold=10, node_threshold=None, return_as='url', dataset='production'):
    CI = catmaid.catmaid_login('FANC', project)
    try:
        n = pymaid.get_neurons(skid)
    except:
        return 'No matching skeleton ID in project {}'.format(project)

    n.downsample(inplace=True)

    target_volume = auth.get_cloudvolume(dataset=dataset)
    seg_ids, points = skel2seg(n, target_volume, transform=True)

    neuron_df, skeleton_df = fragment_dataframes(seg_ids,
                                                 points,
                                                 segment_threshold=segment_threshold,
                                                 node_threshold=node_threshold)
    annotations = [{'name': 'skeleton coords',
                    'type': 'points',
                    'data': skeleton_df}]
    return render_scene(neurons=neuron_df, annotations=annotations, return_as=return_as)


def skel2seg(neuron,
             target_volume,
             transform=True):
    nodes = neuron.nodes[['x', 'y', 'z']].values / np.array([4.3, 4.3, 45])

    if transform is True:
        points = realignment.fanc3_to_4(nodes)
    else:
        points = nodes

    return rootID_lookup.segIDs_from_pts_service(points, cv=target_volume), points


def fragment_dataframes(seg_ids, coords, segment_threshold=20, node_threshold=None):
    ''' Generate dataframes for skeleton nodes, and subsequent neuron fragments
        seg_ids: list,array      List of rootIDs associated with the skeleton
        coords:  nx3 array       skeleton coords in voxel space
        segment_threshold: int   if not None, the number of segments to include in the dataframe. Will include the most overlapping segment IDs
        node_threshold: int      if not None, the number of nodes required for a segment ID to be included'''

    ids, counts = np.unique(seg_ids, return_counts=True)
    value_counts = np.array(list(zip(ids, counts)), dtype=int)
    value_counts = value_counts[value_counts[:, 0] != 0, :]

    primary_neuron = value_counts[value_counts[:, 1] == max(value_counts[:, 1]), 0]
    fragments = value_counts[value_counts[:, 0] != primary_neuron, :]

    if segment_threshold and not node_threshold:
        ids_to_use = fragments[np.argsort(-fragments[:, 1])[0:segment_threshold], 0]
    elif node_threshold and not segment_threshold:
        ids_to_use = fragments[fragments[:, 1] > node_threshold][:, 0]
    elif node_threshold and segment_threshold:
        print('Warning: cannot use segment and node threshold concurrently,defaulting to segment threshold')
        ids_to_use = fragments[np.argsort(-fragments[:, 1])]
    else:
        ids_to_use = seg_ids

    skeleton_df = pd.DataFrame(columns=['pt_root_id', 'pt_position'])
    skeleton_df.pt_position = [i for i in coords]
    cmap = cm.get_cmap('Blues_r', len(ids_to_use))
    sk_colors = [colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    neuron_df = pd.DataFrame(columns=['pt_root_id', 'pt_position', 'color'])
    neuron_df.pt_root_id = ids_to_use

    for i in range(len(ids_to_use)):
        idx = seg_ids == ids_to_use[i]
        neuron_df.loc[neuron_df.pt_root_id == ids_to_use[i], 'color'] = sk_colors[i]

    neuron_df = neuron_df.append({'pt_root_id': primary_neuron[0], 'pt_position': None, 'color': "#ff0000"}, ignore_index=True)
    return neuron_df, skeleton_df


def render_scene(neurons=None,
                 annotations=None,
                 synapses_layer=True,
                 nuclei_layer=False,
                 volumes_layer=True,
                 return_as='url',
                 **kwargs):
    """
    Render a neuroglancer scene with an arbitrary number of annotation layers

    ---Arguments---
    neurons: int or list or DataFrame or np.array or string
        A IDs or locations of the neurons that you want to display in
        the scene. This can be provided in a few ways:
        - A int specifying a single segment ID
        - A list containing segment IDs
        - A pd.DataFrame containing a segmentID column and optionally a
          color column
        - A np.array with shape (N, 3) containing the pt_position coordinates of
          N points, each of which indicates the location of a neuron
        - A string specifying the name of a CAVE table from which to
          pull neurons

    annotations: list of dicts
        A list of dictionaries specifying annotation dataframes. Format
        must be [{'name':str,'type':'points','data': DataFrame}]
        where DataFrame is formatted appropriately for the annotation
        type. Currently only points is implemented.

    synapses_layer: bool (default True)
        Whether to include the postsynaptic blobs layer in the state
    nuclei_layer: bool (default False)
        Whether to include the nuclei layer in the state
    volumes_layer: bool (default True)
        Whether to include the volume outlines in the state

    return_as: string
        Must be 'json' or 'url' (default). Specifies whether to return a
        json representation of the desired neuroglancer state, or a
        neuroglancer link (after uploading the JSON state to a
        neuroglancer state server).

    ---Other kwargs---
    client: CAVEclient
        Override the default CAVEclient
    img_source: str
        Override the default url for the image layer
    seg_source: str
        Override the default url for the segmentation layer
    state_server: str
        Override the default url for the json state server
    bg_color: str
        Set the background color. Must be 'w'/'white' or a hex color code
    
    ---Returns---
    Neuroglancer state (as a json or a url depending on 'return_as')
    """
    # This import is delayed because it triggers creation of a CAVEclient,
    # which I don't want to do until this function is called
    from . import ngl_info

    # Process kwargs
    if 'client' in kwargs:
        client = kwargs['client']
    else:
        client = auth.get_caveclient()

    if 'materialization_version' in kwargs:
        materialization_version = kwargs['materialization_version']
    else:
        materialization_version = client.materialize.most_recent_version()

    # Handle 'neurons' argument
    if neurons is None:
        # Default to showing the 'homepage' FANC neuron
        neurons = np.array([[48848, 114737, 2690]])
    elif isinstance(neurons, int):
        neurons = [neurons]
    elif isinstance(neurons, str):
        neurons = list(client.materialize.query_table(neurons, materialization_version = materialization_version).pt_root_id.values)

    if isinstance(neurons, np.ndarray) and np.any(neurons < 10000000000000000): # convert only when np array has coordinates.
        neurons = list(rootID_lookup.segIDs_from_pts_service(neurons))
 
    if isinstance(neurons, list):
        cmap = cm.get_cmap('Set1', len(neurons))
        neurons_df = pd.DataFrame(columns=['pt_root_id', 'color'])
        neurons_df['pt_root_id'] = neurons
        neurons_df['color'] = [colors.rgb2hex(cmap(i)) for i in range(cmap.N)]


    # Process kwargs
    if 'img_source' in kwargs:
        ngl_info.im['path'] = kwargs['img_source']
    if 'seg_source' in kwargs:
        ngl_info.seg['path'] = kwargs['seg_source']
    if 'state_server' in kwargs:
        ngl_info.other_options['jsonStateServer'] = kwargs['state_server']
    if 'bg_color' in kwargs:
        if kwargs['bg_color'].lower() in ['white', 'w']:
            kwargs['bg_color'] == '#ffffff'
        ngl_info.other_options['perspectiveViewBackgroundColor'] = kwargs['bg_color']

    # Make layers
    img_config = ImageLayerConfig(
        name=ngl_info.im['name'],
        source=ngl_info.im['path']
    )
    seg_config = SegmentationLayerConfig(
        name=ngl_info.seg['name'],
        source=ngl_info.seg['path'],
        selected_ids_column='pt_root_id',
        color_column='color',
        fixed_ids=None,
        active=True
    )
    synapses_config = ImageLayerConfig(
        name=ngl_info.syn['name'],
        source=ngl_info.syn['path']
    )
    nuclei_config = SegmentationLayerConfig(
        name=ngl_info.nuclei['name'],
        source=ngl_info.nuclei['path']
    )

    # Annotation layer(s)
    annotation_states = []
    annotation_data = []
    if annotations is not None:
        for i in annotations:

            if 'pt_root_id' in i['data'].columns:
                linked_segmentation_column = 'pt_root_id'
            else:
                linked_segmentation_column = None

            if i['type'] == 'points':
                anno_mapper = PointMapper(point_column='pt_position', linked_segmentation_column = linked_segmentation_column)
            elif i['type'] == 'spheres':
                anno_mapper = SphereMapper(center_column='pt_position', radius_column='radius', linked_segmentation_column = linked_segmentation_column)
            else:
                raise NotImplementedError

            anno_layer = AnnotationLayerConfig(name=i['name'], mapping_rules=anno_mapper)
            annotation_states.append(
                StateBuilder(layers=[anno_layer], resolution=ngl_info.voxel_size)
            )
            annotation_data.append(i['data'])

    # Build a state with the requested layers
    layers = [img_config, seg_config]
    if synapses_layer:
        layers.append(synapses_config)
    if nuclei_layer:
        layers.append(nuclei_config)

    view_options = ngl_info.view_options.copy()
    zoom_2d = view_options.pop('zoom_2d')

    standard_state = StateBuilder(
        layers=layers,
        resolution=ngl_info.voxel_size,
        view_kws=view_options
    )
    chained_sb = ChainedStateBuilder([standard_state] + annotation_states)

    # Turn state into a dict, then add some last settings manually
    state = chained_sb.render_state([neurons_df] + annotation_data, return_as='dict')
    if synapses_layer:
        # User must make synapse layer visible manually if they want to see postsynaptic blobs
        state['layers'][2]['visible'] = False
    if volumes_layer:
        state['layers'].append(ngl_info.volume_meshes)
    state['navigation']['zoomFactor'] = zoom_2d
    state.update(ngl_info.other_options)

    if return_as == 'json':
        return state
    elif return_as == 'url':
        json_id = client.state.upload_state_json(state)
        return client.state.build_neuroglancer_url(json_id, ngl_info.ngl_app_url)
    else:
        raise ValueError('"return_as" must be "json" or "url" but was {}'.format(return_as))
