import numpy as np
import pandas as pd
import pyperclip
from cloudvolume import CloudVolume, view, Bbox
from nglui import statebuilder,annotation,easyviewer,parser
from nglui.statebuilder import *
import json
from tqdm import tqdm
import argparse
import sqlite3

from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import csv

sys.path.append(os.path.abspath("/home/skuroda/FANC_auto_recon/segmentation"))
import authentication_utils as auth
import rootID_lookup as IDlook
sys.path.append(os.path.abspath("/home/skuroda/FANC_auto_recon/synapses"))
import connectivity_utils
import synaptic_links
sys.path.append(os.path.abspath("/home/skuroda/FANC_auto_recon/nuclei_prediction"))
import config

# setup paths and read csv files
outputpath = '/n/groups/htem/users/skuroda'
soma_table = '/home/skuroda/body_info_Aug2021.csv'
syn_table = '/n/groups/htem/users/skuroda/full_VNC_synapses.csv'
syn_table_db = '/n/groups/htem/users/skuroda/synapses.db'

output_somadf_name = 'full_VNC_soma_20210816.csv'
output_name = 'Premotor_20210816'
MN = pd.read_csv('/home/skuroda/MN.csv', header=0)

column_name = ['Seg ID','Synapses','Has soma?','Major merges fixed?','Major splits fixed?','Come back to me later','Other notes']

# update
synaptic_links.update_synapse_tables(csv_path=syn_table)
synaptic_links.update_synapse_tables(csv_path=syn_table, db_path=syn_table_db)
config.update_soma_table(soma_table.rsplit('/', 1)[0], input_table_name='body_info_Aug2021', output_table_name=output_somadf_name)
updated_soma_table = soma_table.rsplit('/', 1)[0] + '/' + output_somadf_name
df = pd.read_csv(updated_soma_table, header=0)
print('tables updated')

# finds premotor inputs
preMN = connectivity_utils.get_synapses(MN['pt_root_id'],syn_table_db,direction='inputs',threshold=1)
print('synapses into motor neurons found')
temp = preMN['pre_root'].value_counts(ascending=False)
preMN_table = pd.DataFrame(temp)
preMN_table.columns = column_name[0:2]
preMN_table = preMN_table.reindex(columns = preMN_table.columns.tolist() + column_name[2:])
print('premotor inputs found')
for i in range(len(preMN_table)):
    if preMN_table.loc[i,column_name[0]] in df['body_rootID']:
        preMN_table.loc[i,column_name[2]] = 'y'
    else:
        pass

num_with_nuclei = (preMN_table[column_name[2]] == 'y').sum()
output = preMN_table.fillna("")
output.to_csv(outputpath + '/' + '{}.csv'.format(output_name), index=False)
print('{num} premotor inputs ({percentage}%) have nuclei'.format(num=num_with_nuclei, percentage=num_with_nuclei/len(preMN_table)))