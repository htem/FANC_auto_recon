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

from datetime import datetime
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


# paths and files
outputpath = '/n/groups/htem/users/skuroda/da2'
updated_soma_fname = 'full_VNC_soma_20210824.csv'
MN = '/home/skuroda/MN202108240256.csv'
date = '2021-08-24 02:56' # UTC

# original tables
orig_soma = '/home/skuroda/body_info_Aug2021.csv'
orig_syn_csv = '/n/groups/htem/users/skuroda/full_VNC_synapses.csv'
orig_syn_db = '/n/groups/htem/users/skuroda/synapses.db'

# other setups
thres=3
if date != None:
    dt_date = datetime.strptime(date, '%Y-%m-%d %H:%M')
    timestamp = int(dt_date.timestamp())
else:
    timestamp = None

cv = auth.get_cv()

print('soma table updating...')
# config.update_soma_table(orig_soma.rsplit('/', 1)[0], orig_soma.rsplit('/', 1)[1], updated_soma_fname, timestamp=timestamp)
print('soma table updated')
updated_soma = orig_soma.rsplit('/', 1)[0] + '/' + updated_soma_fname
df = pd.read_csv(updated_soma, header=0)

# get segID of descending and ascending
print('retrieving segIDs of descending and ascending neurons...')
LHSda_df = pd.read_csv(outputpath + '/LHS75200Seeds.csv', header=0)
RHSda_df = pd.read_csv(outputpath + '/RHS75200Seeds.csv', header=0)

LHSxyz = LHSda_df['Coordinate 1'].str.strip('()').str.split(',',expand=True)
RHSxyz = RHSda_df['Coordinate 1'].str.strip('()').str.split(',',expand=True)

LHSda = IDlook.segIDs_from_pts_cv(LHSxyz.astype(int).to_numpy(),cv,timestamp)
RHSda = IDlook.segIDs_from_pts_cv(RHSxyz.astype(int).to_numpy(),cv,timestamp)

LHSda_indices = np.isin(LHSda.astype('int64'), df['body_rootID'].values)
left_d = LHSda[LHSda_indices]
left_a = LHSda[np.logical_not(LHSda_indices)]

RHSda_indices = np.isin(RHSda.astype('int64'), df['body_rootID'].values)
right_d = RHSda[RHSda_indices]
right_a = RHSda[np.logical_not(RHSda_indices)]

out1 = left_d.astype(np.int64)
out1.tofile(outputpath + '/' + '{}.bin'.format('left_d'))
out2 = left_a.astype(np.int64)
out2.tofile(outputpath + '/' + '{}.bin'.format('left_a'))
out3 = right_d.astype(np.int64)
out3.tofile(outputpath + '/' + '{}.bin'.format('right_d'))
out4 = right_a.astype(np.int64)
out4.tofile(outputpath + '/' + '{}.bin'.format('right_a'))

# update synapse table
print('updating synapse table...')
# synaptic_links.update_synapse_csv(orig_syn_csv,cv,max_tries=100000)
# synaptic_links.update_synapse_db(orig_syn_db,orig_syn_csv)
print('synapse table updated')

# find premotor inputs
MN_table = pd.read_csv(MN, header=0)
MNT1L = MN_table[MN_table['name'].str.endswith('T1L')]
MNT1R = MN_table[MN_table['name'].str.endswith('T1R')]
print('MN table read')

pMN_T1Ls = connectivity_utils.get_synapses(MNT1L['pt_root_id'],orig_syn_db,direction='inputs',threshold=3)
pMN_T1Rs = connectivity_utils.get_synapses(MNT1R['pt_root_id'],orig_syn_db,direction='inputs',threshold=3)
print('premotor synpases found')
temp = pMN_T1Ls['pre_root'].value_counts()
pMN_T1L = pd.DataFrame(temp).reset_index()
pMN_T1L.columns = ['root_id','synapses']
temp = pMN_T1Rs['pre_root'].value_counts()
pMN_T1R = pd.DataFrame(temp).reset_index()
pMN_T1R.columns = ['root_id','synapses']
print('premotor neurons found')

pMN_T1L.to_csv(outputpath + '/pMN_T1L.csv', index=False)
pMN_T1R.to_csv(outputpath + '/pMN_T1R.csv', index=False)