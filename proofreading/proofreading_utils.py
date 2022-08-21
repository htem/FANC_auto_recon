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
import vtk
import os
from cloudvolume import CloudVolume
from cloudvolume.frontends.precomputed import CloudVolumePrecomputed
try:
    from trimesh import exchange
except ImportError:
    from trimesh import io as exchange


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
                 gpuMemoryLimit=4000000000,
                 systemMemoryLimit=8000000000,
                 concurrentDownloads=64,
                 perspectiveViewBackgroundColor="",
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
        client, token = authentication_utils.get_caveclient()

    if neurons is None:
        neurons_df = pd.DataFrame(columns=['segment_id', 'xyz', 'color'])
    elif isinstance(neurons, list):
        cmap = cm.get_cmap('Set1', len(neurons))
        neurons_df = pd.DataFrame(columns=['segment_id', 'xyz', 'color'])
        neurons_df['segment_id'] = neurons
        neurons_df['color'] = [colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    paths = authentication_utils.get_cv_path()

    if img_source is None:
        img_source = client.info.image_source()
    if seg_source is None:
        seg_source = client.info.segmentation_source()
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

    view_options = {"layout": "xy"}
    standard_state = StateBuilder(layers=[img_layer, seg_layer], 
                                  resolution=[4.3, 4.3, 45],
                                  view_kws=view_options)

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
    state = json.loads(chained_sb.render_state([neurons_df] + annotation_data, return_as='json'))

    memory_options = {"gpuMemoryLimit": gpuMemoryLimit,
                      "systemMemoryLimit": systemMemoryLimit,
                      "concurrentDownloads": concurrentDownloads,
                      "jsonStateServer": "{}".format(paths['json_server']['url'])}
    state.update(memory_options)
    if perspectiveViewBackgroundColor != "":
        bg_color_options = {"perspectiveViewBackgroundColor": perspectiveViewBackgroundColor}
        state.update(bg_color_options)
    
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
        jsn_id = client.state.upload_state_json(state)
        return client.state.build_neuroglancer_url(jsn_id, 
                                                   authentication_utils.get_cv_path('neuroglancer_base')['url']) # client.info.viewer_site()
    else:
        return state


def plot_neurons(segment_ids, cv=None,
                 cmap='Blues', opacity=1,
                 plot_type='mesh',
                 resolution=[4.3,4.3,45],
                 camera=None,
                 zoom_factor=300,
                 client=None,
                 plot_synapses=False,
                 synapse_type='all',
                 synapse_threshold=3,
                 plot_soma=False,
                 plot_outlines=False,
                 scale_bar_orig_3D=None,
                 scale_bar_orig_2D=None,
                 view='X',
                 scale_bar_length=10000,
                 save=False,
                 save_path=None,
                 width=1080,
                 height=720):
    """
    Visualize neurons in 3d meshes, optionally saving high-resolution png images.

    Parameters
    ----------
    segment_ids :  list
        list of segment IDs of neurons
    cv : cloudvolume.frontends.precomputed.CloudVolumePrecomputed
        cloud-volume that segment IDs exist
    camera :  int
        json state id of neuroglancer scene. required to plot scale bar
    client : caveclient.frameworkclient.CAVEclientFull
        CAVEclient to retrieve tables for visualizing synapses and soma
    plot_synapses :  bool
        visualize synapses
    plot_soma : bool
        visualize soma
    plot_outlines :  bool
        visualize volume outlines
    scale_bar_orig_3D : list
        specify an origin of a 3D scale bar that users want to place in xyz
    scale_bar_orig_2D :  list
        specify an origin of a 2D scale bar that users want to place in xyz
    view : str
        'X', 'Y', or 'Z' to specify which plane you want your 2D scale bar to appear
    scale_bar_length :  int
        length of a scale bar in nm
    save : bool
        write png image to disk, if false will open interactive window (default False)
    save_path : str
        filepath to save png image

    Returns
    -------
    vtk.vtkRenderer
        renderer when code was finished
    png
        output png image 
        (generate two images with/without scale bar if you specify to plot it)
    """

    if isinstance(segment_ids, int):
        segment_ids = [segment_ids]

    colormap = cm.get_cmap(cmap, len(segment_ids))

    if cv is None:
        cv = authentication_utils.get_cloudvolume()

    if client is None:
        client = authentication_utils.get_caveclient()

    if isinstance(camera, int):
        state = client.state.get_state_json(camera)
        camera = trimesh_vtk.camera_from_ngl_state(state, zoom_factor=zoom_factor)

    neuron_actors = []
    annotation_actors = []
    # outline_actor = []
    for j in enumerate(segment_ids):
        # Get mesh
        if isinstance(cv, CloudVolumePrecomputed):
            mesh = cv.mesh.get(j[1])[j[1]]
        else:
            mesh = cv.mesh.get(j[1], use_byte_offsets=True)[j[1]]
        mp_mesh = trimesh_io.Mesh(mesh.vertices, mesh.faces)

        neuron = meshwork.Meshwork(mp_mesh, seg_id=j[1], voxel_resolution=[4.3, 4.3, 45])

        if plot_soma == True:
            soma_df = client.materialize.query_table(client.info.get_datastack_info()['soma_table'],
                                                     filter_equal_dict={'pt_root_id': j[1]})
            neuron.add_annotations('soma_pt', soma_df, point_column='pt_position', anchored=False)

        # get synapses
        if plot_synapses is True:
            if synapse_type is 'inputs':
                input_table = connectivity_utils.get_synapsesv2(j[1],
                                                                direction='inputs',
                                                                threshold=synapse_threshold)

                neuron.add_annotations('syn_in', input_table, point_column='post_pt')


            elif synapse_type is 'outputs':
                input_table = None
                output_table = connectivity_utils.get_synapsesv2(j[1],
                                                                 direction='outputs',
                                                                 threshold=synapse_threshold)
            elif synapse_type is 'all':
                input_table = connectivity_utils.get_synapsesv2(j[1],
                                                                direction='inputs',
                                                                threshold=synapse_threshold)

                output_table = connectivity_utils.get_synapsesv2(j[1],
                                                                 direction='outputs',
                                                                 threshold=synapse_threshold)

                neuron.add_annotations('syn_in', input_table, point_column='post_pt')
                neuron.add_annotations('syn_out', output_table, point_column='pre_pt')


            else:
                raise Exception('incorrect synapse type, use: "inputs", "outputs", or "all"')

        # Plot

        if 'mesh' in plot_type:
            neuron_actors.append(trimesh_vtk.mesh_actor(neuron.mesh, color=colormap(j[0])[0:3], opacity=opacity))
        elif 'skeleton' in plot_type and plot_soma is not None:
            neuron.skeletonize_mesh(soma_pt=neuron.anno.soma_pt.points[0], invalidation_distance=5000)
            neuron_actors.append(trimesh_vtk.skeleton_actor(neuron.skeleton, line_width=3, color=colormap(j[0])[0:3]))
        elif 'skeleton' in plot_type and plot_soma is None:
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

    all_actors = neuron_actors + annotation_actors

    if plot_outlines == True:
        outlines_actors = []
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'volume_meshes')
        mesh_outer = read_mesh_stl(os.path.normpath(os.path.join(base, 'tissueoutline_aug2019.stl')))
        mp_mesh = trimesh_io.Mesh(mesh_outer[0], mesh_outer[1])
        outlines_outer = meshwork.Meshwork(mp_mesh, seg_id=[1], voxel_resolution=[4.3, 4.3, 45])
        outlines_actors.append(trimesh_vtk.mesh_actor(outlines_outer.mesh, color=(191/255,191/255,191/255), opacity=0.1))

        # paths = authentication_utils.get_cv_path()
        # volume_outlines_cv = CloudVolume(paths['volumes']['url'], use_https=True)
        # mesh_outer = volume_outlines_cv.mesh.get([1], use_byte_offsets=True)[1]
        # mp_mesh = trimesh_io.Mesh(mesh_outer.vertices, mesh_outer.faces)
        # outlines_outer = meshwork.Meshwork(mp_mesh, seg_id=[1], voxel_resolution=[4.3, 4.3, 45])
        # outlines_actors.append(trimesh_vtk.mesh_actor(outlines_outer.mesh, color=(191/255,191/255,191/255), opacity=0.1))

        mesh_inner = read_mesh_stl(os.path.normpath(os.path.join(base, 'JRC2018_VNC_FEMALE_to_FANC', 'VNC_template_Aug2020.stl')))
        mp_mesh = trimesh_io.Mesh(mesh_inner[0], mesh_inner[1])
        outlines_inner = meshwork.Meshwork(mp_mesh, seg_id=[2], voxel_resolution=[4.3, 4.3, 45])
        outlines_actors.append(trimesh_vtk.mesh_actor(outlines_inner.mesh, color=(211/255,67/255,214/255), opacity=0.1))

        # mesh_inner = volume_outlines_cv.mesh.get([2], use_byte_offsets=True)[2]
        # mp_mesh = trimesh_io.Mesh(mesh_inner.vertices, mesh_inner.faces)
        # outlines_inner = meshwork.Meshwork(mp_mesh, seg_id=[2], voxel_resolution=[4.3, 4.3, 45])
        # outlines_actors.append(trimesh_vtk.mesh_actor(outlines_inner.mesh, color=(211/255,67/255,214/255), opacity=0.1))

        all_actors = all_actors + outlines_actors

    # add actor for scale bar
    if (scale_bar_orig_3D is not None) or (scale_bar_orig_2D is not None):
        if camera is not None:
            if scale_bar_orig_3D is not None:
                scale_bar_ctr = np.array(scale_bar_orig_3D)*np.array(resolution) # - np.array([0,scale_bar_length,0])
                scale_bar_actor = trimesh_vtk.scale_bar_actor(scale_bar_ctr,camera=camera,length=scale_bar_length,linewidth=1)
            else:
                scale_bar_ctr = np.array(scale_bar_orig_2D)*np.array(resolution) - np.array([0,scale_bar_length,0])
                scale_bar_actor = scale_bar_actor_2D(scale_bar_ctr,view=view,camera=camera,length=scale_bar_length,linewidth=1)
        else:
            raise Exception('Need camera to set up scale bar')

    if (scale_bar_orig_3D is None) and (scale_bar_orig_2D is None):
        trimesh_vtk.render_actors(all_actors, camera=camera, do_save=save, 
                                  filename=save_path, 
                                  scale=4, video_width=width, video_height=height)
    elif save_path is None:
        trimesh_vtk.render_actors((all_actors + [scale_bar_actor]), camera=camera, do_save=save, 
                                  filename=save_path, 
                                  scale=4, video_width=width, video_height=height)
    else:
        trimesh_vtk.render_actors(all_actors, camera=camera, do_save=save, 
                                  filename=save_path, 
                                  scale=1, video_width=width, video_height=height)
        trimesh_vtk.render_actors((all_actors + [scale_bar_actor]), camera=camera, do_save=save, 
                                    filename=(save_path.rsplit('.', 1)[0] + '_scalebar.' + save_path.rsplit('.', 1)[1]), 
                                    scale=1, video_width=width, video_height=height)


def scale_bar_actor_2D(center, camera, view='X', length=10000, color=(0, 0, 0), linewidth=5, font_size=20):
    """
    Creates a scale bar actor very similar to trimesh_vtk.scale_bar_actor(), but on a specific plane with 
    a given size.
    """
    axes_actor = vtk.vtkCubeAxesActor2D()
    axes_actor.SetBounds(center[0], center[0]+length,
                         center[1], center[1]+length,
                         center[2], center[2]+length)

    axes_actor.SetLabelFormat("")
    axes_actor.SetCamera(camera)
    axes_actor.SetNumberOfLabels(0)
    axes_actor.SetFlyModeToNone()
    axes_actor.SetFontFactor(1.0)
    axes_actor.SetCornerOffset(0.0)
    if view == 'X':
        axes_actor.XAxisVisibilityOn()
    else:
        axes_actor.XAxisVisibilityOff()
    if view == 'Y':
        axes_actor.YAxisVisibilityOn()
    else:
        axes_actor.YAxisVisibilityOff()
    if view == 'Z':
        axes_actor.ZAxisVisibilityOn()
    else:
        axes_actor.ZAxisVisibilityOff()
    axes_actor.GetProperty().SetColor(*color)
    axes_actor.GetProperty().SetLineWidth(linewidth)

    tprop = vtk.vtkTextProperty()
    tprop.SetColor(*color)
    tprop.ShadowOff()
    tprop.SetFontSize(font_size)
    if view == 'X':
        axes_actor.SetXLabel((str(length)+' nm'))
    if view == 'Y':
        axes_actor.SetYLabel((str(length)+' nm'))
    if view == 'Z':
        axes_actor.SetZLabel((str(length)+' nm'))
    axes_actor.SetAxisTitleTextProperty(tprop)
    axes_actor.SetAxisLabelTextProperty(tprop)

    return axes_actor


def read_mesh_stl(filename):
    with open(filename, 'r') as fp:
        mesh_d = exchange.stl.load_stl(fp)
    vertices = mesh_d['vertices']
    faces = mesh_d['faces']
    normals = mesh_d.get('normals', None)
    link_edges = None
    node_mask = None
    return vertices, faces, normals, link_edges, node_mask
