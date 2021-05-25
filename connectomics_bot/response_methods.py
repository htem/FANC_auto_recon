import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import pymaid
from annotationframeworkclient import FrameworkClient
sys.path.append(str(Path.cwd().parent.parent))

from FANC_auto_recon.segmentation import authentication_utils,rootID_lookup
from FANC_auto_recon.synapses import connectivity_utils
from FANC_auto_recon.annotations import schema_download
from FANC_auto_recon.proofreading import proofreading_utils
from FANC_auto_recon.skeletonization import catmaid_utilities
from FANC_auto_recon.transforms import realignment
from cloudvolume import CloudVolume


# Primary response methods
synapse_table = 't1_svIDs_v4.csv'

def get_upstream_partners(root_id,threshold=1):
    fname = Path.cwd() / synapse_table
    direction = 'inputs'
    return connectivity_utils.batch_partners(fname,root_id,direction,threshold)



def get_downstream_partners(root_id,threshold=1):
    fname = Path.cwd() / synapse_table
    direction = 'outputs'
    return connectivity_utils.batch_partners(fname,root_id,direction,threshold)



def get_top_upstream_partners(root_id,cutoff=10,threshold=1):
    fname = Path.cwd() / synapse_tabe
    direction = 'inputs'
    partners = connectivity_utils.batch_partners(fname,root_id,direction,threshold=threshold)
    top = partners.pre_id.value_counts()[0:cutoff]
    return top



def get_top_downstream_partners(root_id,cutoff=10,threshold=1):
    fname = Path.cwd() / synapse_tabe
    direction = 'outputs'
    partners = connectivity_utils.batch_partners(fname,root_id,direction,threshold=threshold)
    top = partners.post_id.value_counts()[0:cutoff]
    return top



def get_annotation_tables():
    client,token = authentication_utils.get_client()
    return client.annotation.get_tables()



def get_user_tables(user):
    client,token = authentication_utils.get_client()
    tables = client.annotation.get_tables()
    user_tables = []
    for i in tables:
        meta = client.annotation.get_table_metadata(i)
        if user in meta['user_id']:
            user_tables.append(i)
    
    return user_tables
    

    
def download_annotation_table(table_name):
    client,token = authentication_utils.get_client()
    if table_name not in client.annotation.get_tables():
        return('table does not exist')
    annotation_table = schema_download.download_annotation_table(client,table_name,get_deleted=False)
    if client.annotation.get_table_metadata(table_name)['schema_type'] == 'bound_tag':
        materialized_table = schema_download.generate_soma_table(annotation_table)
    else:
        materialized_table = schema_download.generate_synapse_table(annotation_table)    
    
    return(materialized_table)



def find_neuron(annotation):
    table = schema_download.find_neurons(annotation)
    
    return table



def getskel2seg(skid, project=13, segment_threshold = 10, node_threshold = None):
    


    link = proofreading_utils.skel2seg(skid, project, 
                                       segment_threshold=segment_threshold, 
                                       node_threshold=node_threshold,
                                       return_as = 'url') 
    return('<'+link+'>')



## Secondary methods

def empty_link():
    target_volume = CloudVolume(authentication_utils.get_cv_path('FANC_production_segmentation')['url'],use_https=True,agglomerate=False)
    return(proofreading_utils.render_scene(seg_ids = None,target_volume=target_volume))

def update_roots():
    
    cv = CloudVolume(authentication_utils.get_cv_path('FANC_production_segmentation')['url'],use_https=True,agglomerate=False)
    output,fails = rootID_lookup.update_roots(synapse_table,cv)
    if output == 'Complete':
        return('Synapses updated')
    else:
        return('Update failed')
    
    

