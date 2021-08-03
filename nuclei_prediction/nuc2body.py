from re import A
import numpy as np
import sys
import os
import pandas as pd
from tqdm import tqdm
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


def summarize_topIDs(array, place, fill):
  uniqueID, count = np.unique(array, return_counts=True)
  count2 = -count
  unsorted_max_indices = np.argsort(count2)
  topIDs = uniqueID[unsorted_max_indices] 
  count2 = count2 /sum(count2)
  end_indice = min(place,len(topIDs))
  id_and_count = np.array([topIDs[0:end_indice-1], count2[0:end_indice-1]],dtype='int64')

  if fill == True:
    arr_temp = sum(id_and_count.tolist(), [])
    result = np.array(map(lambda x: x + [0]*(place-len(x)), arr_temp))
  else:
    result = id_and_count

  return result


@queueable
def task_get_surrounding(i):
  try:
    # "blockID", "x", "y", "z", "nuc_segID", "nucID", "size_x_mip4", "size_y_mip4", "size_z_mip4", "vol"
    rowi = df.iloc[i,:].values
    cord_mip0 = rowi[1:4]
    cord_mip2 = mip0_to_mip2_array(cord_mip0)
    bbox_size = [rowi[5]*2**2*window_coef, rowi[6]*2**2*window_coef, rowi[7]*2**2*window_coef]

    seg_nuc = seg.download_point(pt=cord_mip2, segids=rowi[4], size=bbox_size, coord_resolution=[17.2, 17.2, 45.0]) #mip2
    
    vol_temp = seg_nuc[:,:,:]
    vol_temp[vol_temp>0] = 1 # change segID assigned to each cell body into 1
    vol = np.squeeze(vol_temp)

    one_in_faces = look_faces(vol, value=1) # save in percentage

    filled = fill_voids.fill(vol, in_place=False) # fill the empty space with fill_voids. Igonore warning
    shifted = vol_shift(filled, pixel=1) # shift the volume one pixel
    
    location_one = argwhere_from_outside(shifted, value=1, bbox_size=bbox_size)

    if len(location_one):
      if choose is not None: 
        lchosen = location_one[0:choose,:]
      else:
        lchosen = location_one
      
      lchosen_mip0 = np.apply_along_axis(mip2_to_mip0_array, 1, lchosen, seg_nuc)
      parent_IDs = IDlook.segIDs_from_pts_cv(pts=lchosen_mip0, cv=seg, progress=False) #mip0

      summarized = summarize_topIDs(parent_IDs,10,fill=True)

      # save
      uniqueID, count = np.unique(parent_IDs, return_counts=True)
      unsorted_max_indices = np.argsort(-count)
      topIDs = uniqueID[unsorted_max_indices] # gives me top5 IDs
      topIDs2 = topIDs[~(topIDs == id)] # I hope this keeps order, remove if same as nuclei id
      topIDs3 = topIDs2[~(topIDs2 == 0)] # no zero
      topIDs3 = np.append(topIDs3, np.zeros(3, dtype = 'uint64'))
      A = np.append(cord_mip0.values, id).astype('int64')
      B = topIDs3.astype('int64')[0:3]
      output = np.append(A, B) #top3

      rowi one_in_faces summarized
      
    else:
      A = np.append(cord_mip0.values, id).astype('int64')
      B = np.zeros(3, dtype = 'int64')
      output = np.append(A, B) #xyz, id, 0,0,0

    output_df = pd.DataFrame(columns=["x", "y", "z", "segIDs", "Parent1", "Parent2", "Parent3"])
    output_df.loc[0] = output
    name = str(i)
    output_df.to_csv(outputpath + '/' + 'cellbody_{}.csv'.format(name), index=False)

  except Exception as e:
    name=str(i)
    with open(outputpath + '/' + '{}.log'.format(name), 'w') as logfile:
        print(e, file=logfile)



def task_nuc2body(i):
  try:
    cord_mip0 = df.iloc[i,0:3] #xyz coordinates
    cord_mip2 = cord_mip0.values.copy() # change coordination from mip0 to mip2
    cord_mip2[0]  = (cord_mip0.values[0] /(2**2))
    cord_mip2[1]  = (cord_mip0.values[1] /(2**2))
    cord_mip2 = cord_mip2.astype('int64')
    id = df.iloc[i,3] #segid

    if id == 0:
      A = np.append(cord_mip0.values, id).astype('int64')
      B = np.zeros(3, dtype = 'int64')
      output = np.append(A, B) #xyz, id, 0,0,0
    else:
      seg_nuc = seg.download_point(pt=cord_mip2, segids=id, size=[size_xy, size_xy, 160], coord_resolution=[17.2, 17.2, 45.0])
      # lowest resolution of seg is [17.2, 17.2, 45.0]
      # pt should be mip0??
      vol_temp = seg_nuc[:,:,:]
      vol_temp[vol_temp>0] = 1 # change segID assigned to each cell body into 1
      vol = np.squeeze(vol_temp)

      filled = fill_voids.fill(vol, in_place=False) # fill the empty space with one 
      # ignore warning

      shifted = vol_shift(filled) # shift the volume
      shifted = shifted.astype('float32')
      shifted[shifted>0] = 1
      shifted[shifted<0] = 0

      location_one = np.argwhere(shifted == 1)
      len(location_one)

      if len(location_one):
        origin = seg_nuc.bounds.minpt # 3072,5248,1792
        parent_coordinates_mip2 = np.add(np.array(location_one), origin)
        parent_coordinates = parent_coordinates_mip2
        parent_coordinates[:,0]  = (parent_coordinates_mip2[:,0] * 2**2)
        parent_coordinates[:,1]  = (parent_coordinates_mip2[:,1] * 2**2)
        parent_coordinates = parent_coordinates.astype('int64')

        #random selection?
        if choose == 0:
          location_random = parent_coordinates
        else:
          index = np.random.choice(parent_coordinates.shape[0], size=choose, replace=False)
          location_random = parent_coordinates[index]

        # Lets get IDs using cell_body_coordinates
        parent_IDs = IDlook.segIDs_from_pts_cv(pts=location_random, cv=seg, progress=False) #mip0

        # save
        uniqueID, count = np.unique(parent_IDs, return_counts=True)
        unsorted_max_indices = np.argsort(-count)
        topIDs = uniqueID[unsorted_max_indices] # gives me top5 IDs
        topIDs2 = topIDs[~(topIDs == id)] # I hope this keeps order, remove if same as nuclei id
        topIDs3 = topIDs2[~(topIDs2 == 0)] # no zero
        topIDs3 = np.append(topIDs3, np.zeros(3, dtype = 'uint64'))
        A = np.append(cord_mip0.values, id).astype('int64')
        B = topIDs3.astype('int64')[0:3]
        output = np.append(A, B) #top3
        
      else:
        A = np.append(cord_mip0.values, id).astype('int64')
        B = np.zeros(3, dtype = 'int64')
        output = np.append(A, B) #xyz, id, 0,0,0

      output_df = pd.DataFrame(columns=["x", "y", "z", "segIDs", "Parent1", "Parent2", "Parent3"])
      output_df.loc[0] = output
      name = str(i)
      output_df.to_csv(outputpath + '/' + 'cellbody_{}.csv'.format(name), index=False)

    seg.cache.flush()

  except Exception as e:
    name=str(i)
    with open(outputpath + '/' + '{}.log'.format(name), 'w') as logfile:
        print(e, file=logfile)



# task queue


def run_local(cmd, count_data=False): # recommended
    try:
        func = globals()[cmd]
    except Exception:
        print("Error: cmd only accepts 'task_get_surrounding', 'task_get_cellbody'")

    tq = LocalTaskQueue(parallel=parallel_cpu)
    if func == task_get_surrounding:
        if file_input is not None:
            with open(file_input) as fd:      
                txtdf = np.loadtxt(fd, dtype='int64')
                tq.insert( partial(func, i) for i in txtdf )
        else:
            tq.insert(( partial(func, i) for i in range(len(df)) )) # NEW SCHOOL
    elif func == task_merge_within_block:
        if count_data == True:
            countdir = outputpath + '/' + 'count_{}'.format(cmd.split('_', 1)[1])
            os.makedirs(countdir, exist_ok=True)
            tq.insert(( partial(func, i, count_data, countdir) for i in range(start, len(block_centers)) ))
            tq.insert(partial(save_count_data, countdir, func, cmd.split('_', 1)[1]))
        else:
            tq.insert(( partial(func, i, count_data) for i in range(start, len(block_centers)) ))
    elif func == task_merge_across_block:
        nuc_data = [] # store input
        for ii in range(len(block_centers)):
            z = np.fromfile(outputpath + '/' + 'block2_{}.bin'.format(str(ii)), dtype=np.int64) # z has [block id, center location in mip0, bbox min, bbox max, nuc_segid, nucid] in int64
            nuc_data.append(z.reshape(int(len(z)/12),12))
        r = np.concatenate(nuc_data)
        r2 = r[~np.all(r == 0, axis=1)] # reomve all zero rows
        u_across, c_across = np.unique(r2[:,11], return_counts=True)
        nucID_duplicated_across = u_across[c_across > 1]
        row_nochange = r[np.isin(r[:,11], u_across[c_across == 1])]
        keep = add_bbox_size_column(row_nochange).astype('int64')

        mergeddir = outputpath + '/' + 'merged_across_block'
        os.makedirs(mergeddir, exist_ok=True)
        
        tq.insert( partial(func, n, r , mergeddir) for n in nucID_duplicated_across)
        tq.insert(partial(save_merged, mergeddir, keep, 'merged')) # [block id, center location in mip0, nuc_segid, nucid, new bbox size] in int64
        if count_data == True:
            tq.insert(partial(save_count, c_across, cmd.split('_', 1)[1])) # save count_data
    else: # task_apply_size_threshold
        previous_df = pd.read_csv(outputpath + '/' + 'merged.csv', header=0)
        # no file means you haven't merged
        tq.insert(partial(func, previous_df))

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