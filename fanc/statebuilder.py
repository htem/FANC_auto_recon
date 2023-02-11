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

from . import auth, catmaid, lookup
from .transforms import realignment


def skel2scene(skid, project=13, segment_threshold=10, node_threshold=None, return_as='url', dataset='production'):
    catmaid.connect(project_id=project)
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

    return lookup.segids_from_pts(points, cv=target_volume), points


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
                 annotation_units='voxels',
                 outlines_layer=True,
                 nuclei_layer=True,
                 synapses_layer=False,
                 return_as='url',
                 **kwargs):
    """
    Render a neuroglancer scene with an arbitrary number of annotation layers

    ---Arguments---
    neurons:
        Some specification of which neurons you want to be displayed in the
        scene. This argument is flexible and can be provided in a few ways:
        - A int specifying a single segment ID
        - A list or pd.Series containing segment IDs
        - A pd.DataFrame with a column named pt_root_id containing
          segment IDs, and optionally with a column named color
        - A np.array with shape (N, 3) containing the xyz coordinates of
          N points, each of which indicates the location of a neuron you
          want to display. Coordinates should be in units of voxels
        - A string specifying the name of a CAVE table from which to
          pull neurons

    annotations: Nx3 numpy array OR dict OR list of dicts
        Data (often point coordinates) you want displayed in an annotation layer.
        If Nx3 numpy array, each row must specify a point coordinate (xyz order).
        If dict, format must be
          {'name': str,
           'type': 'points' OR 'spheres',
           'data': numpy array OR DataFrame}
        where data is formatted appropriately for the specified type.
        Currently supported types and their corresponding data:
        - 'points': data must be an Nx3 numpy array or a DataFrame with a
                    column named 'pt_position'
        - 'spheres': data must be a DataFrame with columns 'pt_position'
                     and 'radius'
        If list of dicts, each dict must have the format above, and each one
        will be displayed as its own annotation layer.

    annotation_units: 'voxel' (Default) or 'nm'
        Whether annotation data is provided in units of voxels or nanometers.
        If in nanometers, data will be divided by `fanc.ngl_info.voxel_size` to
        convert to voxels.

    synapses_layer: bool (default True)
        Whether to include the postsynaptic blobs layer in the state
    nuclei_layer: bool (default False)
        Whether to include the nuclei layer in the state
    outlines_layer: bool (default True)
        Whether to include the region outlines in the state

    return_as: string
        Must be 'json' or 'url' (default). Specifies whether to return a
        json representation of the desired neuroglancer state, or a
        neuroglancer link (after uploading the JSON state to a
        neuroglancer state server).

    ---Other kwargs---
    client: CAVEclient
        Override the default CAVEclient
    materialization_version: int
        A materialization version for querying CAVEclient
    img_source: str
        Override the default url for the image layer
    seg_source: str
        Override the default url for the segmentation layer
    state_server: str
        Override the default url for the json state server
    bg_color: str
        Set the background color. Must be 'w'/'white' or a hex color code
    nuclei: int or list or DataFrame or np.array
        Nucleus IDs to visualize specific nuclei. Set nuclei_layer=True when using.
    
    ---Returns---
    Neuroglancer state (as a json or a url depending on 'return_as')
    """
    # This import is delayed because it triggers creation of a CAVEclient,
    # which I don't want to do until this function is called
    from . import ngl_info

    # Process some kwargs here
    if 'client' in kwargs:
        client = kwargs['client']
    else:
        client = auth.get_caveclient()

    if 'materialization_version' in kwargs:
        materialization_version = kwargs['materialization_version']
    else:
        materialization_version = client.materialize.most_recent_version()

    if annotation_units not in ['nm', 'nanometer', 'nanometers', 'vox', 'voxel', 'voxels']:
        raise ValueError(f"annotation_units must be 'nm' or 'voxel' but was {annotation_units}")

    # Build a DataFrame containing rootIDs starting from whatever type is given
    if neurons is None:
        # None -> np.array
        # Default to showing the 'homepage' FANC neuron
        neurons = np.array([[48848, 114737, 2690]])
    elif isinstance(neurons, int):
        # int -> list
        neurons = [neurons]
    elif isinstance(neurons, str):
        # str -> DataFrame
        neurons = client.materialize.query_table(neurons, materialization_version=materialization_version)
    if isinstance(neurons, pd.Series):
        # pd.Series -> pd.DataFrame or list
        try:
            # If Series contains point coordinates instead of rootIDs, lookup rootIDs
            iter(neurons[0])
            neurons = lookup.segids_from_pts(neurons.values)
        except:
            neurons = pd.DataFrame(neurons)
    if isinstance(neurons, np.ndarray):
        # np.array -> list
        if np.any(neurons < 10000000000000000):
            # If array contains point coordinates instead of rootIDs, lookup rootIDs
            neurons = lookup.segids_from_pts(neurons)
        neurons = list(neurons)
    if isinstance(neurons, list):
        # list -> pd.DataFrame
        neurons = pd.DataFrame({'pt_root_id': neurons})

    if not isinstance(neurons, pd.DataFrame):
        raise TypeError('Could not determine how to handle neurons argument')

    # Add a color column
    cmap = cm.get_cmap('Set1', len(neurons))
    neurons['color'] = [colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    # Process the rest of kwargs
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

    # Additional layer(s)
    additional_states = []
    additional_data = []
    if annotations is not None:
        if isinstance(annotations, np.ndarray):
            annotations = {
                'name': 'points',
                'type': 'points',
                'data': pd.DataFrame({'pt_position': [pt for pt in annotations]})
            }
        if isinstance(annotations, dict):
            annotations = [annotations]

        for i in annotations:
            if isinstance(i['data'], np.ndarray):
                i['data'] = pd.DataFrame({'pt_position': [pt for pt in i['data']]})

            if 'pt_root_id' in i['data'].columns:
                segid_column = 'pt_root_id'
            else:
                segid_column = None

            if annotation_units in ['nm', 'nanometer', 'nanometers']:
                i['data'].pt_position = [row for row in
                                         np.vstack(i['data'].pt_position) / ngl_info.voxel_size]

            if i['type'] == 'points':
                anno_mapper = PointMapper(point_column='pt_position',
                                          linked_segmentation_column=segid_column)
            elif i['type'] == 'spheres':
                anno_mapper = SphereMapper(center_column='pt_position',
                                           radius_column='radius',
                                           linked_segmentation_column=segid_column)
            else:
                raise NotImplementedError(f"Unrecognized annotation type: '{i['type']}'")

            anno_layer = AnnotationLayerConfig(name=i['name'], mapping_rules=anno_mapper)
            additional_states.append(
                StateBuilder(layers=[anno_layer], resolution=ngl_info.voxel_size)
            )
            additional_data.append(i['data'])
    if nuclei_layer:
        nuclei_config = SegmentationLayerConfig(name=ngl_info.nuclei['name'],
                                                source=ngl_info.nuclei['path'],
                                                selected_ids_column='nucleus_id')
        if 'nuclei' in kwargs:
            try:
                iter(kwargs['nuclei'])
                nucleus_ids = kwargs['nuclei']
            except:
                nucleus_ids = [kwargs['nuclei']]
            nuclei_df = pd.DataFrame(columns=['nucleus_id'])
            nuclei_df['nucleus_id'] = nucleus_ids
            additional_data.append(nuclei_df)
        else:
            additional_data.append(None)
        additional_states.append(StateBuilder(layers=[nuclei_config],
                                              resolution=ngl_info.voxel_size)
        )
    if synapses_layer:
        synapses_config = ImageLayerConfig(name=ngl_info.syn['name'],
                                           source=ngl_info.syn['path'])
        additional_states.append(StateBuilder(layers=[synapses_config],
                                              resolution=ngl_info.voxel_size))
        additional_data.append(None)


    # Build a state with the requested layers
    standard_state = StateBuilder(layers=[img_config, seg_config],
                                  resolution=ngl_info.voxel_size,
                                  view_kws=ngl_info.view_options)
    chained_sb = ChainedStateBuilder([standard_state] + additional_states)

    # Turn state into a dict, then add some last settings manually
    state = chained_sb.render_state([neurons] + additional_data, return_as='dict')
    if outlines_layer:
        state['layers'].insert(2, ngl_info.outlines_layer)
    ngl_info.final_json_tweaks(state)

    if return_as == 'json':
        return state
    elif return_as == 'url':
        json_id = client.state.upload_state_json(state)
        return client.state.build_neuroglancer_url(json_id, ngl_info.ngl_app_url)
    else:
        raise ValueError('"return_as" must be "json" or "url" but was {}'.format(return_as))
