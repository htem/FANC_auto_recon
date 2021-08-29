import numpy as np
import sys
import os
import pandas as pd
from glob import glob
import argparse
import  sqlite3

from cloudvolume import CloudVolume, Bbox
import fill_voids
from taskqueue import TaskQueue, queueable, LocalTaskQueue
from functools import partial
from lib import *
sys.path.append(os.path.abspath("../segmentation"))
# to import rootID_lookup and authentication_utils like below
import rootID_lookup as IDlook
import authentication_utils as auth

# -----------------------------------------------------------------------------
# argument
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
outputpath = '/n/groups/htem/users/skuroda/aug2021-2s'
# outputpath = '../Output'
path_to_nuc_list = '~/nuc_info_Aug2021ver2.csv'
# path_to_nuc_list = '../Output/nuc_info.csv'

# variables
np.random.seed(123)
window_coef = 2 # window size to get nuclei in mip2
output_name = 'soma_info_Aug2021ver2'

# could-volume url setting
seg = CloudVolume(auth.get_cv_path('FANC_production_segmentation')['url'], use_https=True, agglomerate=False, cache=True, progress=False) # mip2
nuclei_seg_cv = CloudVolume(auth.get_cv_path('nuclei_seg_Aug2021')['url'], cache=False, progress=False, use_https=True,autocrop=True, bounded=False) # mip4
# read csv
df = pd.read_csv(path_to_nuc_list, header=0)

# -----------------------------------------------------------------------------


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
    # blockID,x,y,z,xminpt,yminpt,zminpt,xmaxpt,ymaxpt,zmaxpt,
    # nuc_svID,nucID,x_svID,y_svID,z_svID,size_x_mip4,size_y_mip4,size_z_mip4,vol
    rowi = df.iloc[i,:].values
    cord_mip0 = rowi[1:4]
    cord_mip4 = mip0_to_mip4_array(cord_mip0)
    bbox_size = np.array([rowi[15]*window_coef, rowi[16]*window_coef, rowi[17]*window_coef], dtype='int64')

    seg_nuc = nuclei_seg_cv.download_point(pt=cord_mip4, size=bbox_size, mip=[68.8,68.8,45.0]) #mip4
    
    vol_temp = seg_nuc[:,:,:]
    vol_temp2 = np.where(vol_temp == rowi[10], 1, 0)
    vol = np.squeeze(vol_temp2)

    filled = fill_voids.fill(vol, in_place=False) # fill the empty space with fill_voids. Ignore DeprecationWarning
    shifted = vol_shift(filled, pixel=1) # shift the volume one pixel
    
    location_one = argwhere_from_outside(shifted, value=1, bbox_size=bbox_size)

    if len(location_one):
      if choose is None: 
        lchosen = location_one
        lchosen_mip0 = np.apply_along_axis(mip4_to_mip0_array, 1, lchosen, seg_nuc)
        surrounding_IDs = IDlook.segIDs_from_pts_cv(pts=lchosen_mip0, cv=seg, progress=False) #mip0
        body_segID = find_most_frequent_ID(surrounding_IDs) # zero excluded
      else:
        p = int(len(location_one) // choose)
        for pi in range(p+1):
          lchosen = location_one[pi*choose:min((pi+1)*choose-1,len(location_one)),:]
          lchosen_mip0 = np.apply_along_axis(mip4_to_mip0_array, 1, lchosen, seg_nuc)
          surrounding_IDs = IDlook.segIDs_from_pts_cv(pts=lchosen_mip0, cv=seg, progress=False) #mip0
          body_segID = find_most_frequent_ID(surrounding_IDs)
          if body_segID != 0:
            break

      body_svID,body_xyz = segID_to_svID(body_segID, surrounding_IDs, lchosen_mip0, reverse=True) # look up from inner voxels
      
    else:
      body_svID = int(0) # proofread
      body_xyz = int(0)

    x = np.hstack((rowi, np.array(body_svID, dtype='int64'),np.array(body_xyz, dtype='int64')))
    x1 = x.astype(np.int64)
    x1.tofile(outputpath + '/' + 'nuc_{}.bin'.format(str(i)))

  except Exception as e:
    name=str(i)
    with open(outputpath + '/' + '{}.log'.format(name), 'w') as logfile:
        print(e, file=logfile)


@queueable
def task_save(dir): 
  arr_temp = []
  for file in glob(dir + '/' + '*.bin'):
      xx = np.fromfile(file, dtype=np.int64)
      arr_temp.append(xx)
  arr = np.array(arr_temp, dtype='int64')
  
  arr2 = np.hstack((arr, np.zeros((arr.shape[0],2), dtype='int64'))) # for root ID

  df_o = pd.DataFrame(arr2, columns =["blockID", "x", "y", "z", "nuc_svID", "xminpt","yminpt","zminpt","xmaxpt","ymaxpt","zmaxpt","nucID", "x_svID","y_svID","z_svID", "size_x_mip4", "size_y_mip4", "size_z_mip4", "vol", "soma_svID", "body_x", "body_y", "body_z", "nuc_rootID", "soma_rootID"])
  df_o2 = df_o.sort_values('vol')
  df_o2 = df_o2.assign(center_xyz=[*zip(df_o2.x, df_o2.y, df_o2.z)])
  df_o2 = df_o2.assign(nuc_xyz=[*zip(df_o2.x_svID, df_o2.y_svID, df_o2.z_svID)])
  df_o2 = df_o2.assign(soma_xyz=[*zip(df_o2.body_x, df_o2.body_y, df_o2.body_z)])
  df_o2 = df_o2.assign(bbx_min=[*zip(df_o2.xminpt, df_o2.yminpt, df_o2.zminpt)])
  df_o2 = df_o2.assign(bbx_max=[*zip(df_o2.xmaxpt, df_o2.ymaxpt, df_o2.zmaxpt)])
  df_o3 = df_o2.reindex(columns=['nucID', 'center_xyz', 'nuc_xyz', 'nuc_svID', 'nuc_rootID', 'soma_xyz', 'soma_svID', 'soma_rootID', 'vol', 'bbx_min', 'bbx_max'])
  # save as csv
  df_o3.to_csv(outputpath + '/' + '{}.csv'.format(output_name), index=False)

  # save as db
  # write_in_db


def run_local(cmd): # recommended
    try:
        func = globals()[cmd]
    except Exception:
        print("Error: cmd only accepts 'task_get_surrounding', 'task_save'")

    tq = LocalTaskQueue(parallel=parallel_cpu)
    if func == task_get_surrounding:
        if file_input is not None:
            with open(file_input) as fd:      
                txtdf = np.loadtxt(fd, dtype='int64', ndmin=1)
                tq.insert( partial(func, i) for i in txtdf )
        else:
            tq.insert(( partial(func, i) for i in range(len(df)) )) # NEW SCHOOL
    else: # func == task_save
      tq.insert(partial(func, outputpath))

    tq.execute(progress=True)
    print('Done')