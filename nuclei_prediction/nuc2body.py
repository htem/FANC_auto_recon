import numpy as np
import sys
import os
import pandas as pd
from glob import glob
import argparse

from cloudvolume import CloudVolume, Bbox
import fill_voids
from taskqueue import TaskQueue, queueable, LocalTaskQueue
from functools import partial
from config import *
sys.path.append(os.path.abspath("../segmentation"))
# to import rootID_lookup and authentication_utils like below
import rootID_lookup as IDlook
import authentication_utils as auth


parser = argparse.ArgumentParser(description='get segIDs of parent neurons from csv files') 
parser.add_argument('-c', '--choose', help='specify the numer of pixels randomly chosen to get segID of parent neuron. default is all surroundinx pixels', type=int)
parser.add_argument('-l', '--lease', help='lease_seconds for TaskQueue.poll. specify in seconds. default is 600sec', default=600, type=int)
parser.add_argument('-p', '--parallel', help='number of cpu cores for parallel processing. default is 12', default=12, type=int)
parser.add_argument('-i', '--input', help='input the list of chunk numbers need recalculated again', type=validate_file)

args = parser.parse_args()
choose=args.choose
lease=args.lease
parallel_cpu=args.parallel
file_input=args.input

# path
outputpath = '/n/groups/htem/users/skuroda/nuclei_output6'
# outputpath = '../Output'
path_to_nuc_list = '~/nuc_info_20210804.csv'
# path_to_nuc_list = '../Output/nuc_info.csv'

# variables
np.random.seed(123)
window_coef = 1.5 # window size to get nuclei in mip2
# threshold variance of surrounding id

# could-volume url setting
seg = CloudVolume(auth.get_cv_path('FANC_production_segmentation')['url'], use_https=True, agglomerate=False, cache=True, progress=False) # mip2
nuclei_seg_cv = CloudVolume(auth.get_cv_path('nuclei_seg_Jul2021')['url'], cache=False, progress=False, use_https=True,autocrop=True, bounded=False) # mip4
# read csv
df = pd.read_csv(path_to_nuc_list, header=0)


def vol_shift(input, pixel): # this is still very slow since this overuse RAM even though np.roll is fast
  # x plane
  x_p = np.roll(input, pixel, axis=0)
  x_p[:pixel,:,:] = 0
  x_n = np.roll(input, -pixel, axis=0)
  x_n[-pixel:,:,:] = 0
  # y plane
  y_p = np.roll(input, pixel, axis=1)
  y_p[:,:pixel,:] = 0
  y_n = np.roll(input, -pixel, axis=1)
  y_n[:,-pixel,:] = 0
  # z plane
  z_p = np.roll(input, pixel, axis=2)
  z_p[:,:,:pixel] = 0
  z_n = np.roll(input, -pixel, axis=2)
  z_n[:,:,-pixel] = 0

  sum = x_p + x_n + y_p + y_n + z_p + z_n
  result = sum - input*6
  result[result>0] = 1
  result[result<0] = 0

  result = result.astype('int64')

  return result


def argwhere_from_outside(volume, value, bbox_size):
  ind = np.argwhere(volume == value)
  center = bbox_size/2

  distance = np.linalg.norm(ind-center)
  sorted_indice = np.argsort(-distance)
  result = ind[sorted_indice]
  
  return result


@queueable
def task_get_surrounding(i):
  try:
    # "blockID", "x", "y", "z", "nuc_segID", "nucID", "size_x_mip4", "size_y_mip4", "size_z_mip4", "vol"
    rowi = df.iloc[i,:].values
    cord_mip0 = rowi[1:4]
    cord_mip4 = mip0_to_mip4_array(cord_mip0)
    bbox_size = np.array([rowi[6]*window_coef, rowi[7]*window_coef, rowi[8]*window_coef], dtype='int64')

    seg_nuc = nuclei_seg_cv.download_point(pt=cord_mip4, size=bbox_size, mip=[68.8,68.8,45.0]) #mip4
    
    vol_temp = seg_nuc[:,:,:]
    vol_temp2 = np.where(vol_temp == rowi[5], 1, 0)
    vol = np.squeeze(vol_temp2)

    filled = fill_voids.fill(vol, in_place=False) # fill the empty space with fill_voids. Ignore DeprecationWarning
    shifted = vol_shift(filled, pixel=1) # shift the volume one pixel
    
    location_one = argwhere_from_outside(shifted, value=1, bbox_size=bbox_size)

    if len(location_one):
      if choose is not None: 
        lchosen = location_one[0:min(choose,len(location_one)),:]
      else:
        lchosen = location_one
      
      lchosen_mip0 = np.apply_along_axis(mip4_to_mip0_array, 1, lchosen, seg_nuc)
      surrounding_IDs = IDlook.segIDs_from_pts_cv(pts=lchosen_mip0, cv=seg, progress=False) #mip0

      body_segID = find_most_frequent_ID(surrounding_IDs) # zero excluded
      
    else:
      body_segID = int(0)

    x = np.hstack((rowi, np.array(body_segID, dtype='int64')))
    x1 = x.astype(np.int64)
    x1.tofile(outputpath + '/' + 'nuc_{}.bin'.format(str(i)))

  except Exception as e:
    name=str(i)
    with open(outputpath + '/' + '{}.log'.format(name), 'w') as logfile:
        print(e, file=logfile)


@queueable
def task_save_as_csv(mergeddir, name): 
  array_withchange = []
  for file in glob(mergeddir + '/' + '*.bin'):
      xx = np.fromfile(file, dtype=np.int64)
      array_withchange.append(xx)
  arr = np.array(array_withchange, dtype='int64')

  df_o = pd.DataFrame(arr, columns =["blockID", "x", "y", "z", "nuc_segID", "nucID", "size_x_mip4", "size_y_mip4", "size_z_mip4", "vol", "body_segID"])
  df_o2 = df_o.sort_values('vol')
  df_o2.to_csv(outputpath + '/' + '{}.csv'.format(name), index=False) #header?


@queueable # edit https://caveclient.readthedocs.io/en/latest/guide/chunkedgraph.html
def task_get_current_rootIDs(mergeddir, name): 
  array_withchange = []
  for file in glob(mergeddir + '/' + '*.bin'):
      xx = np.fromfile(file, dtype=np.int64)
      array_withchange.append(xx)
  arr = np.array(array_withchange, dtype='int64')

  df_o = pd.DataFrame(arr, columns =["blockID", "x", "y", "z", "nuc_segID", "nucID", "size_x_mip4", "size_y_mip4", "size_z_mip4", "vol", "body_segID"])
  df_o2 = df_o.sort_values('vol')
  df_o2.to_csv(outputpath + '/' + '{}.csv'.format(name), index=False) #header?


def run_local(cmd): # recommended
    try:
        func = globals()[cmd]
    except Exception:
        print("Error: cmd only accepts 'task_get_surrounding', 'task_save_as_csv'")

    tq = LocalTaskQueue(parallel=parallel_cpu)
    if func == task_get_surrounding:
        if file_input is not None:
            with open(file_input) as fd:      
                txtdf = np.loadtxt(fd, dtype='int64', ndmin=1)
                tq.insert( partial(func, i) for i in txtdf )
        else:
            tq.insert(( partial(func, i) for i in range(len(df)) )) # NEW SCHOOL
    else: # 'task_save_as_csv'
      tq.insert(partial(func, outputpath,'body_info'))

    tq.execute(progress=True)
    print('Done')