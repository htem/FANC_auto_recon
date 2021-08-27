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



# find premotor inputs
MN_table = pd.read_csv(MN, header=0)
MNT1L = MN_table[MN_table['name'].str.endswith('T1L')]
MNT1R = MN_table[MN_table['name'].str.endswith('T1R')]
print('MN table read')

pMN_T1Rs = connectivity_utils.get_synapses(MNT1R['pt_root_id'],orig_syn_db,direction='inputs',threshold=3)
print('premotor synpases found')
temp = pMN_T1Rs[pMN_T1Rs['pre_root'] == 648518346504926338]
temp.to_csv('~/grgr.csv', index=False)