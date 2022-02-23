#!/usr/bin/env python3

from ..transforms import realignment
from ..segmentation import authentication_utils, rootID_lookup
from ..skeletonization import catmaid_utilities
from ..synapses import connectivity_utils
from nglui.statebuilder import *
import numpy as np
import pandas as pd
import pymaid
import json
from matplotlib import cm, colors
from meshparty import trimesh_vtk, trimesh_io, meshwork


def skel2scene(skid, project=13, segment_threshold=10, node_threshold=None, return_as='url', dataset='production'):
    CI = catmaid_utilities.catmaid_login('FANC', project)
    try:
        n = pymaid.get_neurons(skid)
    except:
        return 'No matching skeleton ID in project {}'.format(project)

    n.downsample(inplace=True)

    target_volume = authentication_utils.get_cloudvolume(dataset=dataset)
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

    skeleton_df = pd.DataFrame(columns=['segment_id', 'xyz'])
    skeleton_df.xyz = [i for i in coords]
    cmap = cm.get_cmap('Blues_r', len(ids_to_use))
    sk_colors = [colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    neuron_df = pd.DataFrame(columns=['segment_id', 'xyz', 'color'])
    neuron_df.segment_id = ids_to_use

    for i in range(len(ids_to_use)):
        idx = seg_ids == ids_to_use[i]
        neuron_df.loc[neuron_df.segment_id == ids_to_use[i], 'color'] = sk_colors[i]

    neuron_df = neuron_df.append({'segment_id': primary_neuron[0], 'xyz': None, 'color': "#ff0000"}, ignore_index=True)
    return neuron_df, skeleton_df


def render_scene(neurons=None,
                 img_source=None,
                 seg_source=None,
                 state_server=None,
                 annotations=None,
                 client=None,
                 return_as='url'):
    ''' render a neuroglancer scene with an arbitrary number of annotation layers
    args:
    
    neurons: list,DataFrame    Either a list of segment IDs, or a pandas DataFrame containing a segmentID column and a color column
    img_source: str   Image path url, default is None which will look for ['Image'] entry in the ~/.cloudvolume/segmentations.json
    seg_source: str   Segmentation path url, default is None which will look for ['FANC_production_segmentation'] entry in the ~/.cloudvolume/segmentations.json
    state_server: str JSON state server path, default is None which will look for ['json_server'] entry in the ~/.cloudvolume/segmentations.json
    annotations: A list of dictionaries specifying annotation dataframes. Format is  annotations = [{'name':str,'type':'points','data': DataFrame}] where
    DataFrame is a DataFrame formatted appropriately for the annotation type. Currently only points is implemented.
    return_as: 'str' Either JSON state, or neuroglancer link after saving the JSON state to the specified JSON server. 
    
    returns:
    
    state/neuroglancer link
    
    '''
    if client is None:
        client, token = authentication_utils.get_client()

    if neurons is None:
        neurons = pd.DataFrame(columns=['segment_id', 'xyz', 'color'])
    elif isinstance(neurons, list):

        cmap = cm.get_cmap('Set1', len(neurons))
        neurons_df = pd.DataFrame(columns=['segment_id', 'xyz', 'color'])
        neurons_df['segment_id'] = neurons
        neurons_df['color'] = [colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    paths = authentication_utils.get_cv_path()

    if img_source is None:
        img_source = paths['Image']['url']
    if seg_source is None:
        seg_source = paths['FANC_production_segmentation']['url']
    if state_server is None:
        state_server = paths['json_server']['url']

    # Set layer segmentation layer
    img_layer = ImageLayerConfig(img_source, name='FANCv4')

    seg_layer = SegmentationLayerConfig(name='seg_Mar2021_proofreading',
                                        source=seg_source,
                                        selected_ids_column='segment_id',
                                        color_column='color',
                                        fixed_ids=None,
                                        active=False)

    standard_state = StateBuilder(layers=[img_layer, seg_layer], resolution=[4.3, 4.3, 45])

    # Data Layer(s)

    annotation_states = []
    annotation_data = []
    if annotations is not None:
        for i in annotations:

            if i['type'] is 'points':
                anno_mapper = PointMapper(point_column='xyz')

            anno_layer = AnnotationLayerConfig(name=i['name'], mapping_rules=anno_mapper)

            annotation_states.append(StateBuilder(layers=[anno_layer], resolution=[4.3, 4.3, 45]))
            annotation_data.append(i['data'])

    chained_sb = ChainedStateBuilder([standard_state] + annotation_states)
    state = json.loads(chained_sb.render_state([neurons] + annotation_data, return_as='json'))

    # Add brain regions
    state['layers'].append({"type": "segmentation",
                            "mesh": paths['volumes']['url'],
                            "objectAlpha": 0.1,
                            "hideSegmentZero": False,
                            "ignoreSegmentInteractions": True,
                            "segmentColors": {
                                "1": "#bfbfbf",
                                "2": "#d343d6"
                            },
                            "segments": [
                                "1",
                                "2"
                            ],
                            "skeletonRendering": {
                                "mode2d": "lines_and_points",
                                "mode3d": "lines"
                            },
                            "name": "volume outlines"
                            })

    if return_as is 'url':
        return paths['neuroglancer_base']['url'] + '?json_url={path}{state_id}'.format(path=paths['json_server']['url'],
                                                                                       state_id=client.state.upload_state_json(
                                                                                           state))
    else:
        return state


def plot_neurons(segment_ids, cv=None,
                 cmap='Blues', opacity=1,
                 plot_type='mesh',
                 plot_synapses=False,
                 soma=None,
                 synapse_type='all',
                 synapse_threshold=3,
                 synapse_table_path=None,
                 camera=None,
                 save=False,
                 save_path=None):
    colormap = cm.get_cmap(cmap, len(segment_ids))

    if isinstance(segment_ids, int):
        segment_ids = [segment_ids]

    if cv is None:
        cv = authentication_utils.get_cv()

    if isinstance(camera, int):
        client, _ = authentication_utils.get_client()
        state = client.state.get_state_json(camera)
        camera = trimesh_vtk.camera_from_ngl_state(state)

    neuron_actors = []
    annotation_actors = []
    for j in enumerate(segment_ids):
        # Get mesh
        mesh = cv.mesh.get(j[1], use_byte_offsets=True)[j[1]]
        mp_mesh = trimesh_io.Mesh(mesh.vertices, mesh.faces)

        neuron = meshwork.Meshwork(mp_mesh, seg_id=j[1], voxel_resolution=[4.3, 4.3, 45])

        if soma is not None:
            if isinstance(soma, pd.DataFrame):
                neuron.add_annotations('soma_pt', soma.query('pt_root_id == @seg_id').copy(),
                                       point_column='pt_position', anchored=False)
            elif isinstance(soma, np.array) or isinstance(soma, list):
                neuron.add_annotations('soma_pt', soma, point_array=True)

        # get synapses
        if plot_synapses is True:
            if synapse_type is 'inputs':
                input_table = connectivity_utils.get_synapses(j[1],
                                                              synapse_table=synapse_table_path,
                                                              direction='inputs',
                                                              threshold=synapse_threshold)

                neuron.add_annotations('syn_in', input_table, point_column='post_pt')


            elif synapse_type is 'outputs':
                input_table = None
                output_table = connectivity_utils.get_synapses(j[1],
                                                               synapse_table=synapse_table_path,
                                                               direction='outputs',
                                                               threshold=synapse_threshold)
            elif synapse_type is 'all':
                input_table = connectivity_utils.get_synapses(j[1],
                                                              synapse_table=synapse_table_path,
                                                              direction='inputs',
                                                              threshold=synapse_threshold)

                output_table = connectivity_utils.get_synapses(j[1],
                                                               synapse_table=synapse_table_path,
                                                               direction='outputs',
                                                               threshold=synapse_threshold)

                neuron.add_annotations('syn_in', input_table, point_column='post_pt')
                neuron.add_annotations('syn_out', output_table, point_column='pre_pt')


            else:
                raise Exception('incorrect synapse type, use: "inputs", "outputs", or "all"')

        # Plot

        if 'mesh' in plot_type:
            neuron_actors.append(trimesh_vtk.mesh_actor(neuron.mesh, color=colormap(j[0])[0:3], opacity=opacity))
        elif 'skeleton' in plot_type and soma is not None:
            neuron.skeletonize_mesh(soma_pt=neuron.anno.soma_pt.points[0], invalidation_distance=5000)
            neuron_actors.append(trimesh_vtk.skeleton_actor(neuron.skeleton, line_width=3, color=colormap(j[0])[0:3]))
        elif 'skeleton' in plot_type and soma is None:
            raise Exception('need a soma point to skeletonize')
        else:
            raise Exception('incorrect plot type, use "mesh" or "skeleton"')

        for i in neuron.anno.table_names:
            if 'syn_in' in i:
                annotation_actors.append(
                    trimesh_vtk.point_cloud_actor(neuron.anno.syn_in.points, size=200, color=(0.0, 0.9, 0.9)))
            elif 'syn_out' in i:
                annotation_actors.append(
                    trimesh_vtk.point_cloud_actor(neuron.anno.syn_out.points, size=200, color=(1.0, 0.0, 0.0)))
            else:
                annotation_actors.append(
                    trimesh_vtk.point_cloud_actor(neuron.anno[i].points, size=200, color=(0.0, 0.0, 0.0)))

    trimesh_vtk.render_actors(neuron_actors + annotation_actors, camera=camera, do_save=save, filename=save_path)
