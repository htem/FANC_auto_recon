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
from ..neuroglancer_utilities import seg_from_pt


def download_annotation_table(client,table_name,id_range=10000):
    
    annotation_table = pd.DataFrame(columns=['deleted','valid','schema_type','reference_table','user_id','created','table_name','id','flat_segmentation_source','description'])
    bins = np.array_split(np.arange(1,client.annotation.get_annotation_count(table_name)+1),100)
        for i in range(len(bins)): 
            annotation_table = annotation_table.append(client.annotation.get_annotation(table_name,annotation_ids=list(bins[i])))

    return(annotation_table)






def generate_soma_table(annotation_table,
                        segmentation_version='Dynamic_V1',
                        resolution=np.array([4.3,4.3,45])):
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
    with open(Path.home() / '.cloudvolume' / 'segmentations.json') as f:
            cloud_paths = json.load(f)
    if 'Dynamic' in segmentation_version:
        cv = CloudVolume(cloud_paths[segmentation_version]['url'],agglomerate=True,use_https=True,progress=True)
    else:
        cv = CloudVolume(cloud_paths[segmentation_version]['url'],use_https=True,progress=True)
        
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
                        resolution=np.array([4.3,4.3,45])):
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

    with open(Path.home() / '.cloudvolume' / 'segmentations.json') as f:
            cloud_paths = json.load(f)
    if 'Dynamic' in segmentation_version:
        cv = CloudVolume(cloud_paths[segmentation_version]['url'],agglomerate=True,use_https=True,progress=False)
    else:
        cv = CloudVolume(cloud_paths[segmentation_version]['url'],progress=False)
        
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
    synapse_table.post_pos_y_vx = np.array([i[1] for i in annotation_table.post_pt_position]) 
    synapse_table.post_pos_z_vx = np.array([i[2] for i in annotation_table.post_pt_position]) 
    
    return(synapse_table)
    
    