import sys
import numpy as np
import pymaid
import pandas as pd
from cloudvolume import CloudVolume
import json
from annotationframeworkclient import FrameworkClient
import nglui
from concurrent import futures
from pathlib import Path

def download_annotation_table(table_name,ids=range(1000)):
    entries = client.annotation.get_annotation(table_name,ids)
    annotation_table = pd.DataFrame(entries)
    return(annotation_table)


def seg_from_pt(pts,vol,image_res=np.array([4.3,4.3,45]),max_workers=4):
    ''' Get segment ID at a point. Default volume is the static segmentation layer for now. 
    Args:
        pts (list): list of 3-element np.arrays of MIP0 coordinates
        vol_url (str): cloud volume url
    Returns:
        list, segment_ID at specified point '''
    
    
    seg_mip = vol.scale['resolution']
    res = seg_mip / image_res

    pts_scaled = [pt // res for pt in pts]
    results = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        point_futures = [ex.submit(lambda pt,vol: vol[list(pt)][0][0][0][0], k,vol) for k in pts_scaled]
        
        for f in futures.as_completed(point_futures):
            results=[f.result() for f in point_futures]
       
    return results



def generate_soma_table(annotation_table,
                        segmentation_version='Dynamic_V1',
                        resolution=np.array([4.3,4.3,45]),
                        token=None):
    ''' Generate a soma table used for microns analysis. This is the workaround for a materialization engine
    Args:
        annotation_table: pd.DataFrame, output from download_cell_table. Retreived from the annotation engine.
        segmentation_version: str, Currently we have 4 for FANC. Two flat segmentations ("Flat_1" and "Flat_2") and two dynamic ("Dynamic_V1/V2"). 
                              This will only work if you have a segmentations.json in your cloudvolume folder. See examples for format.
        resolution: np.array, Resolution of the mip0 coordinates of the version (not necessarily the same as the segmentation layer resolution).
                              For all but the original FANC segmentation, this will be [4.3,4.3,45]
        token: str, currently, CloudVolume requires a workaround for passing google secret tokens. This won't work unless you edit your cloudvolume 
                              file to remove the check for hexidecimal formatting of tokens. Updates should be coming to fix this. 
        '''

    soma_table = pd.DataFrame(columns=['name','cell_type',
                                       'pt_position','pt_root_id',
                                       'soma_x_nm','soma_y_nm','soma_z_nm',
                                       'found'])
    with open(Path.home() / 'cloudvolume' / 'segmentations.json') as f:
            cloud_paths = json.load(f)
    if 'Dynamic' in segmentation_version:
        cv = CloudVolume(cloud_paths[segmentation_version]['url'],agglomerate=True,use_https=True,secrets=token)
    else:
        cv = CloudVolume(cloud_paths[segmentation_version]['url'])
        
    seg_ids = seg_from_pt(annotation_table.pt_position,cv)
    
    soma_table.name = annotation_table.tag
    soma_table.pt_position = annotation_table.pt_position
    soma_table.pt_root_id = seg_ids
    soma_table.soma_x_nm = np.array([i[0] for i in annotation_table.pt_position]) * resolution[0]
    soma_table.soma_y_nm = np.array([i[1] for i in annotation_table.pt_position]) * resolution[1]
    soma_table.soma_z_nm = np.array([i[2] for i in annotation_table.pt_position]) * resolution[2]
    
    return(soma_table)



def generate_synapse_table(annotation_table,
                        segmentation_version='Dynamic_V1',
                        resolution=np.array([4.3,4.3,45]),
                        token=None):
    ''' Generate a soma table used for microns analysis. This is the workaround for a materialization engine
    Args:
        annotation_table: pd.DataFrame, output from download_cell_table. Retreived from the annotation engine.
        segmentation_version: str, Currently we have 4 for FANC. Two flat segmentations ("Flat_1" and "Flat_2") and two dynamic ("Dynamic_V1/V2"). 
                              This will only work if you have a segmentations.json in your cloudvolume folder. See examples for format.
        resolution: np.array, Resolution of the mip0 coordinates of the version (not necessarily the same as the segmentation layer resolution).
                              For all but the original FANC segmentation, this will be [4.3,4.3,45]
        token: str, currently, CloudVolume requires a workaround for passing google secret tokens. This won't work unless you edit your cloudvolume 
                              file to remove the check for hexidecimal formatting of tokens. Updates should be coming to fix this. 
        '''
     
    
    synapse_table = pd.DataFrame(columns=['id','pre_root_id','post_root_id',
                                      'cleft_vx','ctr_pt_x_nm','ctr_pt_y_nm','ctr_pt_z_nm',
                                      'pre_pos_x_vx','pre_pos_y_vx','pre_pos_z_vx',
                                      'ctr_pos_x_vx','ctr_pos_y_vx','ctr_pos_z_vx',
                                      'post_pos_x_vx','post_pos_y_vx','post_pos_z_vx'])

    with open(Path.home() / 'cloudvolume' / 'segmentations.json') as f:
            cloud_paths = json.load(f)
    if 'Dynamic' in segmentation_version:
        cv = CloudVolume(cloud_paths[segmentation_version]['url'],agglomerate=True,use_https=True,secrets=token)
    else:
        cv = CloudVolume(cloud_paths[segmentation_version]['url'])
        
    pre_ids = seg_from_pt(annotation_table.pre_pt_position,cv)
    post_ids = seg_from_pt(annotation_table.post_pt_position,cv)
    
    synapse_table.pre_root_id = pre_ids
    synapse_table.post_root_id = post_ids
    
    # TODO: This in not a stupid way. 
    synapse_table.ctr_pt_x_nm = np.array([i[0] for i in annotation_table.ctr_pt_position]) * resolution[0]
    synapse_table.ctr_pt_y_nm = np.array([i[1] for i in annotation_table.ctr_pt_position]) * resolution[1]
    synapse_table.ctr_pt_z_nm = np.array([i[2] for i in annotation_table.ctr_pt_position]) * resolution[2]
    
    synapse_table.pre_pos_x_vx = np.array([i[0] for i in annotation_table.pre_pt_position]) 
    synapse_table.pre_pos_y_vx = np.array([i[1] for i in annotation_table.pre_pt_position]) 
    synapse_table.pre_pos_z_vx = np.array([i[2] for i in annotation_table.pre_pt_position]) 
    
    synapse_table.post_pos_x_vx = np.array([i[0] for i in annotation_table.post_pt_position]) 
    synapse_table.post_pos_x_vx = np.array([i[1] for i in annotation_table.post_pt_position]) 
    synapse_table.post_pos_x_vx = np.array([i[2] for i in annotation_table.post_pt_position]) 
    
    return(synapse_table)
    
    