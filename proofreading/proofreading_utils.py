from ..segmentation import authentication_utils
from ..segmentation import rootID_lookup
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
        seg_source = authentication_utils.get_cv_path('Dynamic_V4')['url']
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