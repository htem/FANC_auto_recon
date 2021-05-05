#!/usr/bin/env python3

from ..segmentation import rootID_lookup
from ..transforms import realignment
from ..segmentation import authentication_utils,rootID_lookup
from nglui import statebuilder,annotation,easyviewer,parser
from nglui.statebuilder import *
import numpy as np
import pandas as pd

def render_scene(seg_ids,
                     target_volume,
                     img_source = None,
                     seg_source = None,
                     state_server = None,
                     annotation_layer = False,
                     annotation_points = None):
    ## TODO: Render brain regions.
    ## TODO: Make annotation rendering richer. 
    if img_source is None:
        img_source = authentication_utils.get_cv_path('Image')['url']
    if seg_source is None:
        seg_source = authentication_utils.get_cv_path('FANC_production_segmentation')['url']
    if state_server is None:
        state_server = 'https://api.zetta.ai/json/post'
       
    img_layer = ImageLayerConfig(img_source,name='FANCv4')


    seg_layer = SegmentationLayerConfig(name = 'seg_Mar2021_proofreading',
                                   source = seg_source,
                                   selected_ids_column=None,
                                   fixed_ids= seg_ids,
                                   active = False)
    if annotation_layer is True:
        df = pd.DataFrame([[i] for i in annotation_points],columns=['Points'])
        pt_map = statebuilder.PointMapper(point_column='Points')
        
        ann_layer = statebuilder.AnnotationLayerConfig('skeleton',color='r',mapping_rules = pt_map)
        state = StateBuilder([img_layer,seg_layer,ann_layer],resolution=[4.3,4.3,45],state_server=state_server)
        return state.render_state(df)
    else:
        state = StateBuilder([img_layer,seg_layer],resolution=[4.3,4.3,45],state_server=state_server)
        return state.render_state() 


def render_fragments(pts,
                     target_volume,
                     segment_threshold = 10,
                     node_threshold = None,
                     img_source = None,
                     seg_source = None,
                     state_server = None,
                     include_skeleton = False):
    
    if img_source is None:
        img_source = authentication_utils.get_cv_path('Image')['url']
    if seg_source is None:
        seg_source = authentication_utils.get_cv_path('FANC_production_segmentation')['url']
    if state_server is None:
        state_server = 'https://api.zetta.ai/json/post'
    
    seg_ids = rootID_lookup.segIDs_from_pts(target_volume,pts)
    
    ids,counts = np.unique(seg_ids,return_counts=True)
    value_counts = np.array(list(zip(ids,counts)),dtype=int)
    
    if segment_threshold and not node_threshold:
        ids_to_use = value_counts[np.argsort(-value_counts[:,1])[0:segment_threshold],0]
    elif node_threshold and not segment_threshold:
        ids_to_use = value_counts[value_counts[:,1]>node_threshold][:,0]
    elif node_threshold and segment_threshold:
        print('Warning: cannot use segment and node threshold concurrently,defaulting to segment threshold')
        ids_to_use = value_counts[np.argsort(-value_counts[:,1])]
    else:
        ids_to_use = seg_ids 
    
    if include_skeleton is True:
        skeleton_pts = pts
    else:
        skeleton_pts = None
    
    return render_scene(ids_to_use,
                        target_volume,
                        img_source = img_source, 
                        seg_source = seg_source, 
                        state_server = state_server,
                        annotation_layer = include_skeleton,
                        annotation_points = skeleton_pts)

def skel2seg(skeleton_id, project_id=13, copy_link=True,threshold=5, verbose=False):
    """
    Given a skeleton ID of a skeleton in the FANC community CATMAID project
    (https://radagast.hms.harvard.edu/catmaidvnc/?pid=13), create a
    neuroglancer state with segmentation objects loaded at the location of each
    skeleton node.
    """
    import pymaid_addons as pa  # github.com/htem/pymaid_addons
    pa.connect_to('hms_vnc')  # Connect to radagast.hms.harvard.edu/catmaidvnc
    # You must provide an API key in pymaid_addons/connection_configs/hms_catmaidvnc.json
    pa.set_source_project_id(13)  # FANC reconstrution community project

    nodes_v3 = pa.pymaid.get_node_table(skeleton_id)[['x', 'y', 'z']].to_numpy()
    nodes_v3 = nodes_v3 / [4.3, 4.3, 45]  # Convert nm to voxels
    if verbose:
        print('nodes_v3')
        print(nodes_v3)
    nodes_v4 = realignment.fanc3_to_4(nodes_v3, verbose=verbose)
    if verbose:
        print('nodes_v4')
        print(nodes_v4)


    from cloudvolume import CloudVolume as CV
    production_seg = CV(
        authentication_utils.get_cv_path('FANC_production_segmentation')['url'],
        agglomerate=False
    )

    neuroglancer_link = render_fragments(nodes_v4, production_seg, threshold)
    if copy_link:
        try:
            import pyperclip
            pyperclip.copy(neuroglancer_link)
        except:
            print("Install pyperclip (pip install pyperclip) for the option to"
                  " programmatically copy the output above to the clipboard")
    return neuroglancer_link
