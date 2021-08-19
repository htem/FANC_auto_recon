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
outputpath = '/n/groups/htem/users/skuroda'
updated_soma_fname = 'full_VNC_soma_20210615.csv'
pMN_csv = None
date = None # '2021-06-15'
MN = '/home/skuroda/MN.csv'
output_name = 'Premotor_20210615.csv'

# original tables
orig_soma = '/home/skuroda/body_info_Aug2021.csv'
orig_syn_csv = '/n/groups/htem/users/skuroda/full_VNC_synapses.csv'
orig_syn_db = '/n/groups/htem/users/skuroda/synapses.db'

# other setups
column_name = ['Seg ID','Synapses','Has soma?','Major merges fixed?','Major splits fixed?','Come back to me later','Other notes']
if date != None:
    dt_date = datetime.strptime(date, '%Y-%m-%d')
    timestamp = int(dt_date.timestamp())
else:
    timestamp = None

# update tables
cv = auth.get_cv()

config.update_soma_table(orig_soma.rsplit('/', 1)[0], orig_soma.rsplit('/', 1)[1], updated_soma_fname, timestamp=timestamp)
print('soma table updated')

if pMN_csv == None:
    synaptic_links.update_synapse_csv(orig_syn_csv,cv,max_tries=1000)
    synaptic_links.update_synapse_db(orig_syn_db,orig_syn_csv)
    print('synapse table updated')
    MN_table = pd.read_csv(MN, header=0)
    print('MN table read')

    ## find premotor inputs
    pMNs = connectivity_utils.get_synapses(MN_table['pt_root_id'],orig_syn_db,direction='inputs',threshold=1)
    print('premotor synpases found')
    temp = pMNs['pre_root'].value_counts(ascending=False)
    pMN = pd.DataFrame(temp).reset_index()
    pMN = pMN.reindex(columns = pMN.columns.tolist() + column_name[2:])
    pMN.columns = column_name
    print('premotor neurons found')

else:
    pMN = pd.read_csv(pMN_csv, header=0)
    print('premotor neurons read')

# find nuclei of premotor inputs
updated_soma = orig_soma.rsplit('/', 1)[0] + '/' + updated_soma_fname
df = pd.read_csv(updated_soma, header=0)

for i in range(len(pMN)):
    if pMN.loc[i,column_name[0]] in df['body_rootID'].values:
        pMN.loc[i,column_name[2]] = 'y'
    else:
        pass

obj_with_nuclei = (pMN[column_name[2]] == 'y').sum()
syn_with_nuclei = pMN[(pMN[column_name[2]] == 'y')][column_name[1]].sum()
output = pMN.fillna("")
output.to_csv(outputpath + '/' + output_name, index=False)

text_o = """\
    as of {date}
    {A} objects out of {B} premotor inputs have nuclei (~{C}%)
    {D} synapses out of {E} premotor inputs have nuclei (~{F}%)\
    """.format(date=date or 'now',
                A=obj_with_nuclei,
                B=len(pMN),
                C=obj_with_nuclei*100/len(pMN),
                D=syn_with_nuclei,
                E=pMN[column_name[1]].sum(),
                F=syn_with_nuclei*100/pMN[column_name[1]].sum())

print(text_o)


