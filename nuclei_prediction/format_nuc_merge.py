from taskqueue import LocalTaskQueue
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath("../segmentation"))
from merge_operation import create_nuc_merge_tasks
import numpy as np

parallel_cpu = 1 # 12 for htem
merge_file = "/Users/sumiya/git/FANC_auto_recon/Output/proofread_soma_temp.csv"
seg_source = 'graphene://https://cave.fanc-fly.com/segmentation/table/mar2021_prod'

# read csv file and select pairs within the same cells
df = pd.read_csv(merge_file, header=0)
df = df[(df.is_neuron=='y') & (df.is_inside=='y')]

df_to_merge = df.reindex(columns=['nuc_svID','soma_svID', 'nuc_xyz', 'soma_xyz'])
df_to_merge.columns =['nuc_sv_id', 'cell_sv_id', 'nuc_sv_id_loc', 'cell_sv_id_loc']
nuc_xyz_df = df['nuc_xyz'].str.strip('()').str.split(',',expand=True)
soma_xyz_df = df['soma_xyz'].str.strip('()').str.split(',',expand=True)
df_to_merge['nuc_sv_id_loc'] = nuc_xyz_df.astype(int).values.tolist()
df_to_merge['cell_sv_id_loc'] = soma_xyz_df.astype(int).values.tolist()
df_to_merge.reset_index(drop=True, inplace=True)

# merge
tq = LocalTaskQueue(parallel=parallel_cpu)
tq.insert(create_nuc_merge_tasks(df_to_merge,
                                         seg_source,
                                         'file:///Users/sumiya/git/FANC_auto_recon/Output/log_merge', # path in cloudfiles  format
                                         resolution=(4.3,4.3,45)))
tq.execute(progress=True)
print('Done')