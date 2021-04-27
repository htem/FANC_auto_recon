#!/usr/bin/env python3

from ..segmentation import authentication_utils
from ..segmentation import rootID_lookup
from ..transforms import realignment
from nglui import statebuilder,annotation,easyviewer,parser
from nglui.statebuilder import *
import numpy as np


def render_fragments(pts,
                     target_volume,
                     img_source = None,
                     seg_source = None,
                     state_server = None):
    
    if img_source is None:
        img_source = authentication_utils.get_cv_path('Image')['url']
    if seg_source is None:
        seg_source = authentication_utils.get_cv_path('FANC_production_segmentation')['url']
    if state_server is None:
        state_server = 'https://api.zetta.ai/json/post'
    
    seg_ids = rootID_lookup.segIDs_from_pts(target_volume,pts)
    
 
    img_layer = ImageLayerConfig(img_source,name='Image_Layer')


    seg_layer = SegmentationLayerConfig(name = 'Segmentation_Layer',
                                   source = seg_source,
                                   selected_ids_column=None,
                                   fixed_ids= np.unique(seg_ids),
                                   active = False)

    state = StateBuilder([img_layer,seg_layer],resolution=[4.3,4.3,45],state_server=state_server)
    
    return state.render_state()

def skel2seg(skeleton_id, project_id=13, copy_link=True, verbose=False):
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
        agglomerate=True
    )

    neuroglancer_link = render_fragments(nodes_v4, production_seg)
    if copy_link:
        try:
            import pyperclip
            pyperclip.copy(neuroglancer_link)
        except:
            print("Install pyperclip (pip install pyperclip) for the option to"
                  " programmatically copy the output above to the clipboard")
    return neuroglancer_link
