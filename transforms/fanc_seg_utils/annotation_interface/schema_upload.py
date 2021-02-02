import sys
import numpy as np
import pandas as pd
from annotationframeworkclient import FrameworkClient
import nglui


def soma_table_entries(state,layer_name = 'cells'):
    ''' Generate entries for a soma table using the bound_tag schema 
    Args:
    state: json, json state from get_json_state
    layer_name: name of layer containing soma coords
    ### Change this to points later. This is just for database copying because I used a stupid annotation
    
    Returns:
    entries: list, list of dict entries for the bound_tag schema'''
    
    cell_layer = nglui.parser.get_layer(state,layer_name)
    entries = []
    for i in cell_layer['annotations']:
        entry = {'tag': i['description'],
                 'pt': {'position': i['pointA']}}
        entries.append(entry)
    return(entries)
        

def upload_cells(client,cell_entries,table_name,description=None):
    ''' Upload cell entries to a soma dable using the bound_tag schema'
    Args:
    client: annotation client object
    cell_entries: list, list of dicts from soma_table_entries
    table_name: str, table name to create. If it exists, it will populate the existing one. 
    description: str, description to supply if creating a new table. It will fail if you don't provide one. 
    
    ##TODO: Some sort of check for redundant synapses is necessary. We could make each upload need to go to a new table, but that seems absurd. 
'''
    
    try:
        client.annotation.create_table(table_name=table_name,
                                   schema_name='bound_tag',
                                   description=description)
    except:
        print('table exists')
        
    for i in cell_entries:
        try:
            client.annotation.post_annotation(table_name=table_name, data=[i])
        except:
            print('Fail',entry)



def get_synapses(state,synapse_layer='synapses'):
    ''' Get the synapse coordinates from a json state and return pre,post,center.
    Args:
        state: json,  json state from get_state_json 
        synapse_layer: str, name of layer containing synapses in the json state. Default is 'synapses'
    Returns:
        pre_pt,post_pt,ctr_pt: list, lists of coordinates for synapses.'''
    
    
    syns = nglui.parser.line_annotations(state,synapse_layer)
    if np.shape(syns)[0] != 2:
        raise Exception( print('Incorrectly formatted synapse annotation. Requires two lists: Presynapse coordinates, Postsynapse coordinates'))
           
    else:
        pre_pt = syns[0]
        post_pt = syns[1]
        ctr_pt = (np.array(pre_pt) + np.array(post_pt)) / 2
    return(pre_pt,post_pt,np.ndarray.tolist(ctr_pt))


def format_synapse(pre,post,ctr):
    ''' Format a synapse for upload to a synapse table
    Args:
           pre: list, mip0 xyz coordinates to presynapse
           post: list, mip0 xyz coordinates to postsynapse
           ctr: list, mip0 xyz coordinates to center point
    Returns:
           data: formatted dict for a synapse table annotation upload
    '''
    data = {
    "type": "synapse",
    'pre_pt': {'position': pre},
    'post_pt': {'position': post},
    'ctr_pt': {'position': ctr}

}
    return(data)

def upload_synapses(client,synapse_coordinates,table_name,description=None):
    ''' Add synapses to a synapse table.
    Args: 
        client: annotation client object
        synapse_coordinates: list, list of shape [3,n,3]. Dim 0 is [pre,post,ctr], Dim 1 is the entry, Dim 2 is [x,y,z]. Output of get_synapses
    ##TODO: Some sort of check for redundant synapses is necessary. We could make each upload need to go to a new table, but that seems absurd. 
    '''
    try:
        client.annotation.create_table(table_name=table_name,
                                   schema_name='synapse',
                                   description=description)
    except:
        print('table exists')
        
    for i in range(np.shape(synapse_coordinates)[1]):
        try:
            entry = format_synapse(synapse_coordinates[0][i],synapse_coordinates[1][i],synapse_coordinates[2][i])
            client.annotation.post_annotation(table_name=table_name, data=[entry])
        except:
            print('Fail',entry)