#!/usr/bin/env python

# Filtering script by Stephan Gerhard, November 2022

import pandas as pd
from scipy import spatial
import datetime
import numpy as np

basefolder = ""
df = pd.read_csv(f"{basefolder}/20221109_fanc_synapses_filtered_zero_sv_scoresover12_noautapses_noduplicates.csv")
voxel_dims = [4.3, 4.3, 45]

# The code I sent filters out synaptic links between the same neuron pair if the 
# pre-points are within a certain distance (e.g. 150nm). 
# Code provided by Sven Dorkenwald, November 2022
def distance_filter(pre_coords, pre_pt_root_ids, post_pt_root_ids, r_nm=150):
    conns = np.ascontiguousarray(np.array([pre_pt_root_ids, post_pt_root_ids], dtype=np.uint64).T)
    
    print(conns.shape, conns.dtype)
    conns_v = conns.view(dtype='u8,u8').reshape(-1)
    pre_syn_kdtree = spatial.cKDTree(pre_coords)
    clustered_syn_ids = pre_syn_kdtree.query_ball_point(pre_coords, r=r_nm)

    removed_ids = set()
    valid_ids = []
    for i_cl in range(len(clustered_syn_ids)):
        
        if i_cl in removed_ids:
            continue

        if len(clustered_syn_ids[i_cl]) > 1:
            local_cluster_ids = np.array(clustered_syn_ids[i_cl])
            conn_m = conns_v[local_cluster_ids] == conns_v[i_cl]
            for id_ in local_cluster_ids[conn_m]:
                if id_ == i_cl:
                    continue

                removed_ids.add(id_)
            valid_ids.append(i_cl)
        else:
            valid_ids.append(i_cl)

    valid_ids = np.array(valid_ids)
    removed_ids = np.array(list(removed_ids))

    assert len(valid_ids) + len(removed_ids) == len(clustered_syn_ids)

    return removed_ids, valid_ids

df['id'] = range(len(df))
pre_coords = np.array(df[['pre_x', 'pre_y', 'pre_z']], dtype=np.float32) * voxel_dims
pre_pt_root_ids = np.array(df["pre_segment_id"], dtype=np.uint64)
post_pt_root_ids = np.array(df["post_segment_id"], dtype=np.uint64)
syn_ids = np.array(df["id"], dtype=np.uint64)
removed_ids, valid_ids = distance_filter(pre_coords, pre_pt_root_ids, post_pt_root_ids)
removed_syn_ids = syn_ids[removed_ids]
valid_syn_ids = syn_ids[valid_ids]
df.set_index('id')
df['distance_filter'] = True
df.loc[removed_syn_ids, 'distance_filter'] = False
df_valid = df.loc[valid_syn_ids, :]
df_valid['pre_pt_position'] = 'POINTZ(' + df_valid['pre_x'].astype(str) + " " +     df_valid['pre_y'].astype(str) + " " + df_valid['pre_z'].astype(str) + ')'
df_valid['post_pt_position'] = 'POINTZ(' + df_valid['post_x'].astype(str) + " " +     df_valid['post_y'].astype(str) + " " + df_valid['post_z'].astype(str) + ')'
df_valid.rename({'id': 'idold'}, inplace=True)
df_valid['score'] = df_valid['sum'].round(decimals=2)
df_valid['id'] = range(len(df_valid))
df_valid["deleted"] = False
df_valid["superceded_id"] = None
df_valid["valid"] = 't'
df_valid["created"] = datetime.datetime.utcnow()
df_valid['pre_pt_supervoxel_id'] = df_valid['pre_sv_id']
df_valid['pre_pt_root_id'] = df_valid['pre_segment_id']
df_valid['post_pt_supervoxel_id'] = df_valid['post_sv_id']
df_valid['post_pt_root_id'] = df_valid['post_segment_id']

df_final = df_valid[[
    "id","created","deleted","superceded_id","valid","pre_pt_position","post_pt_position","score"
]]

df_final_seg = df_valid[[
    "id","pre_pt_supervoxel_id","pre_pt_root_id","post_pt_supervoxel_id","post_pt_root_id"
]]

# Final export for cave team (headers may be required)
df_final.to_csv(f"{basefolder}/20221118_fanc_syn_scoreasfloat.csv", index=False, header=False)
df_final_seg.to_csv(f"{basefolder}/20221117_fanc_syn_seg.csv", index=False, header=False)

