from re import A
import numpy as np
import sys
import os
import pandas as pd
from tqdm import tqdm
from glob import glob
import argparse

from cloudvolume import CloudVolume, view, Bbox
import fill_voids
from taskqueue import TaskQueue, queueable, LocalTaskQueue
from functools import partial
from concurrent.futures import ProcessPoolExecutor
sys.path.append(os.path.abspath("../segmentation"))
# to import rootID_lookup and authentication_utils like below
import rootID_lookup as IDlook
import authentication_utils as auth


def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

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
queuepath = '/n/groups/htem/users/skuroda/nuclei_tasks6'
# queuepath = '../Output/nuclei_tasks'
outputpath = '/n/groups/htem/users/skuroda/nuclei_output6'
# outputpath = '../Output'
path_to_nuc_list = '~/nuc_info.csv'
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


def mip0_to_mip2(x,y,z):
  xyz_mip2 = np.array([(x/(2**2)),(y/(2**2)), z])
  xyz_mip2 = xyz_mip2.astype('int64')

  return xyz_mip2[0], xyz_mip2[1], xyz_mip2[2]


def mip0_to_mip2_array(array):
  X, Y, Z = mip0_to_mip2(array[0], array[1], array[2])
  result = np.array([X, Y, Z])
  return result


def mip0_to_mip4(x,y,z):
  xyz_mip4 = np.array([(x/(2**4)),(y/(2**4)), z])
  xyz_mip4 = xyz_mip4.astype('int64')

  return xyz_mip4[0], xyz_mip4[1], xyz_mip4[2]


def mip0_to_mip4_array(array):
  X, Y, Z = mip0_to_mip4(array[0], array[1], array[2])
  result = np.array([X, Y, Z])
  return result


def look_faces(volume, value):
  pixel_total = 2*(volume.shape[0]*volume.shape[1]+volume.shape[1]*volume.shape[2]+volume.shape[0]*volume.shape[2])
  x1 = (volume[0,:,:] == value).sum()
  x2 = (volume[-1,:,:] == value).sum()
  y1 = (volume[:,0,:] == value).sum()
  y2 = (volume[0,-1,:] == value).sum()
  z1 = (volume[:,:,0] == value).sum()
  z2 = (volume[:,:,-1] == value).sum()

  result = (x1 + x2 + y1 + y2 + z1 + z2)/pixel_total
  
  return int(result*100) # percentage


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

def mip2_to_mip0(x,y,z, img):
  origin = img.bounds.minpt
  xyz_mip2 = np.add(np.array([x,y,z]), origin)
  xyz_mip0 = np.array([(xyz_mip2[0] * 2**2),(xyz_mip2[1] * 2**2), xyz_mip2[2]])
  xyz_mip0 = xyz_mip0.astype('int64')

  return xyz_mip0[0], xyz_mip0[1], xyz_mip0[2]


def mip2_to_mip0_array(array, img):
  X, Y, Z = mip2_to_mip0(array[0], array[1], array[2], img)
  result = np.array([X, Y, Z])
  return result


def mip4_to_mip0(x,y,z, img):
  origin = img.bounds.minpt
  xyz_mip4 = np.add(np.array([x,y,z]), origin)
  xyz_mip0 = np.array([(xyz_mip4[0] * 2**4),(xyz_mip4[1] * 2**4), xyz_mip4[2]])
  xyz_mip0 = xyz_mip0.astype('int64')

  return xyz_mip0[0], xyz_mip0[1], xyz_mip0[2]


def mip4_to_mip0_array(array, img):
  X, Y, Z = mip4_to_mip0(array[0], array[1], array[2], img)
  result = np.array([X, Y, Z])
  return result


def find_most_frequent_ID(array):
  uniqueID, count = np.unique(array, return_counts=True)
  unsorted_max_indices = np.argsort(-count)
  topIDs1 = uniqueID[unsorted_max_indices] 
  topIDs2 = topIDs1[~(topIDs1 == 0)] # no zero
  if topIDs2.size == 0:
      topID = np.zeros(1, dtype = 'int64') # empty then zero
  else:
      topID = topIDs2.astype('int64')[0]

  return topID


@queueable
def task_get_surrounding(i):
  try:
    # "blockID", "x", "y", "z", "nuc_segID", "nucID", "size_x_mip4", "size_y_mip4", "size_z_mip4", "vol"
    rowi = df.iloc[i,:].values
    cord_mip0 = rowi[1:4]
    cord_mip4 = mip0_to_mip4_array(cord_mip0)
    bbox_size = [rowi[5]*window_coef, rowi[6]*window_coef, rowi[7]*window_coef]

    # seg_nuc = seg.download_point(pt=cord_mip2, segids=rowi[4], size=bbox_size, coord_resolution=[17.2, 17.2, 45.0]) #mip2
    seg_nuc = nuclei_seg_cv.download_point(pt=cord_mip4, segids=rowi[5], size=bbox_size, mip=[68.8,68.8,45.0]) #mip4
    
    vol_temp = seg_nuc[:,:,:]
    vol_temp[vol_temp>0] = 1 # change nucID assigned to each cell body into 1
    vol = np.squeeze(vol_temp)

    # one_in_faces = look_faces(vol, value=1) # save in percentage

    filled = fill_voids.fill(vol, in_place=False) # fill the empty space with fill_voids. Igonore warning
    shifted = vol_shift(filled, pixel=1) # shift the volume one pixel
    
    location_one = argwhere_from_outside(shifted, value=1, bbox_size=bbox_size)

    if len(location_one):
      if choose is not None: 
        lchosen = location_one[0:choose,:]
      else:
        lchosen = location_one
      
      lchosen_mip0 = np.apply_along_axis(mip4_to_mip0_array, 1, lchosen, seg_nuc)
      surrounding_IDs = IDlook.segIDs_from_pts_cv(pts=lchosen_mip0, cv=seg, progress=False) #mip0

      body_segID = find_most_frequent_ID(surrounding_IDs) # zero excluded
      
    else:
      body_segID = int(0)

    x = np.hstack((rowi, np.array(body_segID, dtype='int64')))
    x1 = x.astype(np.int64)
    x1.tofile(outputpath + '/' + 'block_{}.bin'.format(str(i)))

  except Exception as e:
    name=str(i)
    with open(outputpath + '/' + '{}.log'.format(name), 'w') as logfile:
        print(e, file=logfile)


def save_merged(mergeddir, name): 
  array_withchange = []
  for file in glob(mergeddir + '/' + '*.bin'):
      xx = np.fromfile(file, dtype=np.int64)
      array_withchange.append(xx)
  arr = np.array(array_withchange, dtype='int64')

  df = pd.DataFrame(arr, columns =["blockID", "x", "y", "z", "nuc_segID", "nucID", "size_x_mip4", "size_y_mip4", "size_z_mip4", "vol", "body_segID"])
  df.to_csv(outputpath + '/' + '{}.csv'.format(name), index=False) #header?


def run_local(cmd): # recommended
    try:
        func = globals()[cmd]
    except Exception:
        print("Error: cmd only accepts 'task_get_surrounding', 'task_save_as_csv'")

    tq = LocalTaskQueue(parallel=parallel_cpu)
    if func == task_get_surrounding:
        if file_input is not None:
            with open(file_input) as fd:      
                txtdf = np.loadtxt(fd, dtype='int64')
                tq.insert( partial(func, i) for i in txtdf )
        else:
            tq.insert(( partial(func, i) for i in range(len(df)) )) # NEW SCHOOL
    else: # 'task_save_as_csv'
      tq.insert(partial(save_merged, outputpath,'body_info'))

    tq.execute(progress=True)
    print('Done')


def create_task_queue():
    tq = TaskQueue('fq://' + queuepath)
    if file_input is None:
      tq.insert(( partial(task_nuc2body, i) for i in range(len(df)) ), parallel=parallel_cpu) # NEW SCHOOL?
      print('Done adding {} tasks to queue at {}'.format(len(df), queuepath))
    else:
      with open(file_input) as fd:      
        txtdf = np.loadtxt(fd, dtype='int64')
        tq.insert(( partial(task_nuc2body, i) for i in txtdf ), parallel=parallel_cpu)
        print('Done adding {} tasks to queue at {}'.format(len(txtdf), queuepath))
        
    tq.rezero()


def run_tasks_from_queue():
    tq = TaskQueue('fq://' + queuepath)
    print('Working on tasks from filequeue "{}"'.format(queuepath))
    tq.poll(
        verbose=True, # prints progress
        lease_seconds=int(lease),
        tally=True # makes tq.completed work, logs 1 byte per completed task
    )
    print('All Done')


# run with taskset?