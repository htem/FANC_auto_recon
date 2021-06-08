import pandas as pd
import numpy as np
import json
import re
import sqlite3

def get_synapses(seg_ids,
                 synapse_table='synapses.db',
                 direction='outputs',
                 threshold=3,
                 drop_duplicates=True):
    '''Primary function for synapse table lookup. Will default to looking for a sql database, but can query either a .csv of a .db.
    args:
    seg_ids:          list, int root ids to query
    synapse_table:    str, path to synapse table
    direction:        str, inputs or outputs
    threshold:        int, synapse threshold to use. default is 3
    drop_duplicates:  bool, whether to drop links between the same supervoxel pair 
    
    returns:
    
    a pd.DataFrame of synapse points, root IDs, 
    '''
        
    if isinstance(seg_ids,int):
        seg_ids = [seg_ids]
    
    syn_table = pd.DataFrame(columns=['pre_SV','post_SV','pre_pt','post_pt','source','pre_root','post_root'])

    for i in seg_ids:
        if '.db' in synapse_table:
            syn_table = syn_table.append(get_partner_synapses_sql(i, database=synapse_table, direction=direction, threshold = threshold))
        elif '.csv' in synapse_table:
            syn_table = syn_table.append(batch_partners( i, synapse_table, direction=direction, threshold=threshold))
        
    cs = lambda x : [int(i) for i in re.findall('[0-9]+', x)]
    syn_table.pre_pt = [cs(i) for i in syn_table.pre_pt]
    syn_table.post_pt = [cs(i) for i in syn_table.post_pt] 
    
    #TODO: add a distance threshold here too. 
    if drop_duplicates is True:
        syn_table.drop_duplicates(subset=['pre_SV', 'post_SV'], inplace=True)
    
    return(syn_table)


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


def get_partner_synapses_csv(root_id, 
                             df, 
                             direction='inputs', 
                             threshold=None):
    if direction == 'inputs':
        to_find = 'post_root'
        to_threshold = 'pre_root'
        
    elif direction == 'outputs':
        to_find = 'pre_root'   
        to_threshold = 'post_root'
    
    partners = df.loc[df[to_find]==root_id]
        
    if threshold is not None:
        counts = partners[to_threshold].value_counts()
        t_idx = counts >= threshold
        
        partners = partners[partners[to_threshold].isin(set(t_idx.index[t_idx==1]))]
    
      
     
    return partners


def get_partner_synapses_sql(root_id, 
                         database='synapses.db', 
                         direction='inputs', 
                         threshold=None):
    
    con = sqlite3.connect(database)
    if direction == 'inputs':
        to_find = 'post_root'
        to_threshold = 'pre_root'
        
    elif direction == 'outputs':
        to_find = 'pre_root'   
        to_threshold = 'post_root'
    
    partners = pd.read_sql_query("SELECT * from synapses WHERE {} = {}".format(to_find,root_id),con)

    con.close()   
    if threshold is not None:
        counts = partners[to_threshold].value_counts()
        t_idx = counts >= threshold
        
        partners = partners[partners[to_threshold].isin(set(t_idx.index[t_idx==1]))]


    return partners


def batch_partners(root_id, fname, direction, threshold=None):

    result = pd.DataFrame(columns=['pre_SV','post_SV','pre_pt','post_pt','source','pre_root','post_root'])

    for chunk in pd.read_csv(fname, chunksize=1000000):
        chunk_result = get_partner_synapses_csv(root_id,chunk,direction=direction,threshold=threshold)
        if len(chunk_result) > 0:
            result = result.append(chunk_result, ignore_index=True)
    

    return result 

