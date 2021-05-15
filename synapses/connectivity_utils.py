import pandas as pd
import numpy as np
import json
import re


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


def get_partner_synapses(root_id, df, direction='inputs', threshold=None):
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

    
    return(partners)


def batch_partners(fname,root_id, direction, threshold=None):

    result = pd.DataFrame(columns=['pre_SV','post_SV','pre_pt','post_pt','source','pre_root','post_root'])

    for chunk in pd.read_csv(fname, chunksize=10000):
        chunk_result = get_partner_synapses(root_id,chunk,direction=direction,threshold=threshold)
        if len(chunk_result) > 0:
            result = result.append(chunk_result, ignore_index=True)
    
    cs = lambda x : [int(i) for i in re.findall('[0-9]+', x)]
    result.pre_pt = [cs(i) for i in result.pre_pt]
    result.post_pt = [cs(i) for i in result.post_pt]   
    return(result)

