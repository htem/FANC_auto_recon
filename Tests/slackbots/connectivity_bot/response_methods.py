import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
from annotationframeworkclient import FrameworkClient




# Primary response methods

def get_upstream_partners(root_id,threshold=1):
    fname = Path.cwd() / 't1_synapse_roots_v4.csv'
    direction = 'inputs'
    return(batch_partners(fname,root_id,direction,threshold))

def get_downstream_partners(root_id,threshold=1):
    fname = Path.cwd() / 't1_synapse_roots_v4.csv'
    direction = 'outputs'
    return(batch_partners(fname,root_id,direction,threshold))

def get_top_upstream_partners(root_id,cutoff=10):
    fname = Path.cwd() / 't1_synapse_roots_v4.csv'
    direction = 'inputs'
    partners = batch_partners(fname,root_id,direction,threshold=1)
    top = partners.pre_id.value_counts()
    return(top[top>cutoff])

def get_top_downstream_partners(root_id,cutoff=10):
    fname = Path.cwd() / 't1_synapse_roots_v4.csv'
    direction = 'outputs'
    partners = batch_partners(fname,root_id,direction,threshold=1)
    top = partners.pre_id.value_counts()
    return(top[top>cutoff])

def get_annotation_tables():
    client = get_client()
    client.annotation.get_tables()
    
def download_annotation_table(table_name,ids=range(10000)):
    client = get_client()
    entries = client.annotation.get_annotation(table_name,ids)
    annotation_table = pd.DataFrame(entries)
    return(annotation_table)

def update_roots():
    print('Not implemented yet')
    
    
## Methods that need to go elsewhere:
def batch_partners(fname,root_id,direction,threshold):

    result = pd.DataFrame(columns=['post_id', 'pre_pt', 'post_pt', 'source', 'pre_id'])

    for chunk in pd.read_csv(fname, chunksize=10000):
        chunk_result = get_partner_synapses(root_id,chunk,direction=direction,threshold=threshold)
        if len(chunk_result) > 0:
            result = result.append(chunk_result, ignore_index=True)
    
    return(result)

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

def get_client():
    ''' Establish an ngl client for interacting with the annotation framework. 
    Returns: 
        client, FrameworkClient object
        token, str, graphene server token'''
    
    token = get_token()    
    datastack_name = 'vnc_v0' # from https://api.zetta.ai/wclee/info/

    client = FrameworkClient(
        datastack_name,
        server_address = "https://api.zetta.ai/wclee",
        auth_token = token
    )
    return(client)


def get_token(SECRET_PATH=None):
    
    if SECRET_PATH is None:
        SECRET_PATH = Path.home() / '.cloudvolume' / 'secrets'/'chunkedgraph-secret.json'
    
    if Path.exists(SECRET_PATH):
        with open(SECRET_PATH) as f:
                token = json.load(f)['token']
    else:
        raise ValueError('{} does not exist.'.format(SECRET_PATH))
    
    return(token)