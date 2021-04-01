import sys
if '/Users/brandon/Documents/Repositories/Python/FANC_auto_recon/transforms/' not in sys.path:
    sys.path.append('/Users/brandon/Documents/Repositories/Python/FANC_auto_recon/transforms/')
from fanc_seg_utils.annotation_interface import neuroglancer_utilities,schema_download,schema_upload,statemanager
from fanc_seg_utils.skeletons import catmaid_utilities,skeleton_manipulations,skeletonization

import connectivity_utils

import pandas as pd
import numpy as np
from annotationframeworkclient import FrameworkClient
import nglui
from matplotlib import cm
from nglui.statebuilder import *
from matplotlib import pyplot as plt
from cloudvolume import CloudVolume
from meshparty import trimesh_vtk
import json
from matplotlib import pyplot as plt



def get_adj(pre_ids,post_ids,symmetric = False):
    if symmetric is True:
        index = set(pre_ids).intersection(post_ids)
        columns = index
    else:
        index = set(pre_ids)
        columns = set(post_ids)
        
    adj = pd.DataFrame(index=index,columns=columns).fillna(0)
    for i in adj.index:
        partners,synapses = np.unique(post_ids[pre_ids == i],return_counts=True)  
        for j in range(len(partners)):
            adj.loc[i,partners[j]] = synapses[j]
    
    return(adj)


def get_partner_synapses(root_id,df,direction='inputs',threshold=None):
    if direction == 'inputs':
        to_find = 'post_id'
        to_threshold = 'pre_id'
        
    elif direction == 'outputs':
        to_find = 'pre_id'   
        to_threshold = 'post_id'
    
    partners = df.loc[df[to_find]==root_id]
        
    if threshold is not None:
        counts = partners[to_threshold].value_counts()
        t_idx = counts >= threshold
        
        partners = partners[partners[to_threshold].isin(set(t_idx.index[t_idx==1]))]

    
    return(partners)


    
def batch_roots(cv,df,n=100000):
    groups = int(np.ceil(len(df)/n))
    full = []
    df = df.join(pd.DataFrame(np.ones([len(df),2]),columns={'pre_roots','post_roots'},dtype='int64'))
    for i in range(groups+1):
        start = n*i
        stop = n*(i+1)
        pre = cv.get_roots(df.pre_id.values[start:stop])
        post = cv.get_roots(df.post_id.values[start:stop])
        try:
            df.loc[start:stop-1,'pre_roots'] = pre
            df.loc[start:stop-1,'post_roots'] = post
        except:
            print('Failed at:',start)
            return(df)
        print(start,stop)
    return(df)
        