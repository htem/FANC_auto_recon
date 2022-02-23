import numpy as np
import sys
import os
import pandas as pd
from glob import glob
import argparse

from cloudvolume import CloudVolume, Bbox
import cc3d
from taskqueue import TaskQueue, queueable, LocalTaskQueue
from functools import partial
from lib import *

sys.path.append(os.path.abspath("../segmentation")) # to import rootID_lookup and authentication_utils like below
import rootID_lookup as IDlook
import authentication_utils as auth

# -----------------------------------------------------------------------------
# argument
parser = argparse.ArgumentParser(description='get segIDs (= root IDs) of nuclei and save into csv files') 
parser.add_argument('-s', '--start', help='specify starting chunk. default is 0', default=0, type=int)
parser.add_argument('-l', '--lease', help='lease_seconds for TaskQueue.poll. specify in seconds. default is 1800sec', default=1800, type=int)
parser.add_argument('-p', '--parallel', help='number of cpu cores for parallel processing. default is 12', default=12, type=int)
parser.add_argument('-i', '--input', help='input the list of chunk numbers need recalculated again', type=validate_file)
parser.add_argument('-c', '--choose', help='specify the numer of pixels randomly chosen to get segID of nuclei. default is all pixels inside each nucleus', type=int)
args = parser.parse_args()

start=args.start
lease=args.lease
parallel_cpu=args.parallel
file_input=args.input
choose=args.choose

# path
outputpath = '/n/groups/htem/users/skuroda/aug2021-2'
# outputpath = '../Output'

# variables
np.random.seed(123)
block_x = 256 # block size in mip4 (72x72x45)
block_y = 256
block_z = 256
skip_y = block_y*288 # 73728, this will ignore y < skip_y, choose around 75000/block_y
thres_s = 0.7 # signal threshold, > threshold to generate nuclei_seg_cv=0.2
thres_x_min = 20 # size threshold for xyz in mip4 (68.8x68.8x45)
thres_y_min = 20 
thres_z_min = 40 
thres_x_max = None
thres_y_max = None
thres_z_max = None
connectivity = 26 # number of nearby voxels to look at for calculating connective components

# name
final_product = 'nuc_info_Aug2021ver2'

# could-volume url setting
cv = CloudVolume(auth.get_cv_path('Image')['url'], use_https=True, agglomerate=False) # mip0
seg = CloudVolume(auth.get_cv_path('FANC_production_segmentation')['url'], use_https=True, agglomerate=False, cache=False) # mip2
nuclei_cv = CloudVolume( # mip4
    auth.get_cv_path('nuclei_map_Aug2021')['url'],
    progress=False,
    cache=False, # to aviod conflicts with LocalTaskQueue
    use_https=True,
    autocrop=True, # crop exceeded volumes of request
    bounded=False
)
nuclei_seg_cv = CloudVolume( # mip4
    auth.get_cv_path('nuclei_seg_Aug2021')['url'],
    cache=False,
    progress=False,
    use_https=True, # this is precomputed so no need to specify agglomerate=False
    autocrop=True, # crop exceeded volumes of request
    bounded=False
)

# -----------------------------------------------------------------------------
# calculate blocks in mip0
## cv.mip_volume_size(0)
## Vec(83968,223232,4390, dtype=int64) in mip0 (4.3x4.3x45)

# first centers
start_x = block_x*2**(4-1) + cv.bounds.minpt[0] # (block width)/2 + offset
start_y = block_y*2**(4-1) + cv.bounds.minpt[1]
start_z = block_z*2**(-1) + cv.bounds.minpt[2] 

# array of block centers
centerX = np.arange(start_x, cv.bounds.maxpt[0], block_x*2**4)
centerY = np.arange(start_y + skip_y, cv.bounds.maxpt[1], block_y*2**4)
centerZ = np.arange(start_z, cv.bounds.maxpt[2], block_z)

# cover the final block but keep the size of each block same
centerX = np.append(centerX, cv.bounds.maxpt[0]-start_x)
centerY = np.append(centerY, cv.bounds.maxpt[1]-start_y)
centerZ = np.append(centerZ, cv.bounds.maxpt[2]-start_z)

block_centers = np.array(np.meshgrid(centerX, centerY, centerZ), dtype='int64').T.reshape(-1,3)
len(block_centers)

# -----------------------------------------------------------------------------


def merge_bbox(array, xminpt=4, xmaxpt=7, row_saved=0):
    out = array[row_saved,:].copy()
    out[xminpt] = min(array[:,xminpt]) # xmin
    out[xminpt+1] = min(array[:,xminpt + 1]) # ymin
    out[xminpt+2] = min(array[:,xminpt + 2]) # zmin
    out[xmaxpt] = max(array[:,xmaxpt]) # xman
    out[xmaxpt+1] = max(array[:,xmaxpt + 1]) # ymax
    out[xmaxpt+2] = max(array[:,xmaxpt + 2]) # xmax

    return out

def update_bbox_to_mip0(array, xminpt=4, xmaxpt=7):
    if array.ndim == 2:
        bbox_size = array[:,xmaxpt:xmaxpt+3] - array[:,xminpt:xminpt+3] # still mip4
        array[:,xminpt] = block_centers[array[:,0]][:,0] + (array[:,xminpt] - block_x/2) * 2**4
        array[:,xminpt+1] = block_centers[array[:,0]][:,1] + (array[:,xminpt+1] - block_y/2) * 2**4
        array[:,xminpt+2] = block_centers[array[:,0]][:,2] + (array[:,xminpt+2] - block_z/2)
        array[:,xmaxpt] = block_centers[array[:,0]][:,0] + (array[:,xmaxpt] - block_x/2) * 2**4
        array[:,xmaxpt+1] = block_centers[array[:,0]][:,1] + (array[:,xmaxpt+1] - block_y/2) * 2**4
        array[:,xmaxpt+2] = block_centers[array[:,0]][:,2] + (array[:,xmaxpt+2] - block_z/2)
        array[:,1:4] = np.array([array[:,xminpt]+array[:,xmaxpt],array[:,xminpt+1]+array[:,xmaxpt+1],array[:,xminpt+2]+array[:,xmaxpt+2]]).T / 2
 
    else: # array.ndim == 1
        bbox_size = array[xmaxpt:xmaxpt+3] - array[xminpt:xminpt+3] # still mip4
        array[xminpt] = block_centers[array[0]][0] + (array[xminpt] - block_x/2) * 2**4
        array[xminpt+1] = block_centers[array[0]][1] + (array[xminpt+1] - block_y/2) * 2**4
        array[xminpt+2] = block_centers[array[0]][2] + (array[xminpt+2] - block_z/2)
        array[xmaxpt] = block_centers[array[0]][0] + (array[xmaxpt] - block_x/2) * 2**4
        array[xmaxpt+1] = block_centers[array[0]][1] + (array[xmaxpt+1] - block_y/2) * 2**4
        array[xmaxpt+2] = block_centers[array[0]][2] + (array[xmaxpt+2] - block_z/2)
        array[1:4] = np.array([array[xminpt]+array[xmaxpt],array[xminpt+1]+array[xmaxpt+1],array[xminpt+2]+array[xmaxpt+2]]) / 2
   
    out2 = np.hstack((array, bbox_size))
    return out2


@queueable
def task_get_nuc_info(i): # use i = 7817 for test, close to [7953 118101 2584]
    try:
        nuclei = nuclei_cv.download_point(block_centers[i], mip=[68.8,68.8,45.0], size=(block_x, block_y, block_z) ) # download_point uses mip0 for pt

        mask_temp = nuclei[:,:,:]
        mask = np.where(mask_temp > thres_s, 1, 0)
        masked = np.squeeze(mask)

        cc_out, N = cc3d.connected_components(masked, return_N=True, connectivity=connectivity)

        stats = cc3d.statistics(cc_out)
        ccid = np.arange(1, N+1)
        centroids = stats["centroids"][1:N+1]
        bbx = stats["bounding_boxes"][1:N+1]

        bbx_pt = np.apply_along_axis(lambda x: Bbox.from_slices(x), 1, bbx)
        bbx_cloud = list(map(lambda x: Bbox2cloud(x), bbx_pt))

        arr = np.column_stack([ccid,centroids, bbx_cloud]) # [cc id, center location, bbox min, bbox max] all in mip4
        arr = np.vstack([arr, np.zeros(10)]) # convert 1D to 2D

        if arr.ndim == 2:
            arr2 = np.hstack((arr.copy().astype('int64'), np.zeros((arr.shape[0],5), dtype='int64'))) # array to store output
            for obj in range(N):
                center_mip0 = mip4_to_mip0_array(arr[obj,1:4], nuclei)
                vinside = np.argwhere(cc_out == int(arr[obj,0]))

                if choose is not None: # random selection
                    index = np.random.choice(vinside.shape[0], size=min(len(vinside), choose), replace=False)
                    lrandom = vinside[index]
                else:
                    lrandom = vinside

                lrandom_mip0 = np.apply_along_axis(mip4_to_mip0_array, 1, lrandom, nuclei)
                lrandom_mip4 = lrandom + nuclei.bounds.minpt

                segIDs = IDlook.segIDs_from_pts_cv(pts=lrandom_mip0, cv=seg, progress=False) # segIDs_from_pts_cv uses mip0 for pt
                nuc_segID = find_most_frequent_ID(segIDs)
                nuc_svID,nuc_xyz = segID_to_svID(nuc_segID, segIDs, lrandom_mip0, reverse=False)

                nucIDs_list = []
                for k in range(len(lrandom_mip4)):
                    nucID_temp = nuclei_seg_cv.download_point(lrandom_mip4[k], mip=[68.8,68.8,45.0], size=(1, 1, 1) ) # download_point uses mip4 for pt
                    nucIDs_list.append(np.squeeze(nucID_temp))
                nucID = find_most_frequent_ID(np.array(nucIDs_list, dtype='int64'))

                arr2[obj,1:4] = center_mip0 # change xyz from mip4 to mip0
                arr2[obj,10] = nuc_svID # insert
                arr2[obj,11] = nucID # insert
                arr2[obj,12:15] = nuc_xyz # insert
                arr2[obj,0] = i # no longer need ccid

        else:
            arr2 = np.zeros(15, dtype = 'int64')
            arr2[0] = i

        # arr2 has [block id, center location in mip0, bbox min, bbox max, nuc_segid, nucid, nuc_xyz in mip0] in int64
        x = arr2.astype(np.int64)
        x.tofile(outputpath + '/' + 'block_{}.bin'.format(str(i)))
    except Exception as e:
        with open(outputpath + '/' + '{}.log'.format(str(i)), 'w') as logfile:
            print(e, file=logfile)


@queueable
def task_merge_within_block(i, count_data, countdir):
    try:
        y = np.fromfile(outputpath + '/' + 'block_{}.bin'.format(str(i)), dtype=np.int64) # y has [block id, center location in mip0, bbox min, bbox max, nuc_segid, nucid, nuc_xyz in mip0] in int64
        y1 = y.reshape(int(len(y)/15),15)
        if y1.ndim == 1: # only one row
            c = 1
            y2 = y1
        else: # more than two rows
            u, c = np.unique(y1[:,11], return_counts=True)
            nucID_duplicated = u[c > 1]
            if len(nucID_duplicated):
                merged=[]
                for dup in range(len(nucID_duplicated)):
                    foo = y1[(y1[:,11] == nucID_duplicated[dup])]
                    hoge = np.where(foo[:,10] != 0)[0]
                    if len(hoge) == 0:
                        bar = merge_bbox(foo)
                    else:
                        bar = merge_bbox(foo, row_saved=hoge[0])
                    merged.append(bar)

                y2 = np.vstack((np.array(merged), y1[np.isin(y1[:,11], u[c == 1])]))
            else:
                y2 = y1

        # y2 still has [block id, center location in mip0, bbox min, bbox max, nuc_segid, nucid, nuc_xyz in mip0] in int64
        y_out = y2.astype(np.int64)
        y_out.tofile(outputpath + '/' + 'block2_{}.bin'.format(str(i)))
        if count_data == True:
            z = np.array([u, c], dtype='int64')
            z.tofile(countdir + '/' + '{}.bin'.format(str(i)))
        else:
            pass
    except Exception as e:
        with open(outputpath + '/' + 'within_{}.log'.format(str(i)), 'w') as logfile:
            print(e, file=logfile)


@queueable
def save_count(count, name):
    sorted = np.sort(count.astype('int64'))[::-1]
    np.savetxt(outputpath + '/' + 'count_{}.txt'.format(name), sorted.astype('int64'))


@queueable
def task_merge_across_block(i, data, mergeddir):
    try:
        extracted = data[data[:,11] == i]
        # [block id, center location in mip0, bbox min, bbox max, nuc_segid, nucid, nuc_xyz in mip0] in int64
        dup_info = extracted[np.argsort(extracted[:, 0])] #sort
        hoge = np.where(dup_info[:,10] != 0)[0]
        
        if len(hoge) == 0:
            dup_info_0 = dup_info[0,:]
        else:
            dup_info_0 = dup_info[hoge[0],:]

        block_ids = dup_info[:,0]
        block_loc = (block_centers[block_ids] - block_centers[dup_info_0[0]]) / (block_x*2**4, block_y*2**4, block_z)
        new_minpts = dup_info[:,4:7] + block_loc*(block_x,block_y,block_z)
        new_maxpts = dup_info[:,7:10]+ block_loc*(block_x,block_y,block_z)

        dup_info_0_new = dup_info_0.copy().astype('int64') # int to store possible negative values
        dup_info_0_new[4:7] = np.amin(new_minpts, axis=0)
        dup_info_0_new[7:10] = np.amax(new_maxpts, axis=0) 
        # [block id, center location in mip0, new bbox min, new bbox max, nuc_segid, nucid, nuc_xyz in mip0] in int64

        # [nX, nY, nZ] = np.array([dup_info_0_new[4]+dup_info_0_new[4+3],dup_info_0_new[5]+dup_info_0_new[5+3],dup_info_0_new[6]+dup_info_0_new[6+3]]) / 2
        # [nX, nY, nZ] = np.array([dup_info_0_new[4]+dup_info_0_new[4+3],dup_info_0_new[5]+dup_info_0_new[5+3],dup_info_0_new[6]+dup_info_0_new[6+3]]) / 2
        hoge = update_bbox_to_mip0(dup_info_0_new).astype('int64')
        
        hoge.tofile(mergeddir + '/' + 'block3_{}.bin'.format(str(i)))
      
    except Exception as e:
        with open(outputpath + '/' + 'across_{}.log'.format(str(i)), 'w') as logfile:
            print(e, file=logfile)


@queueable
def save_merged(mergeddir, array_nochange, name): 
    array_withchange = []
    for file in glob(mergeddir + '/' + 'block3_*.bin'):
        xx = np.fromfile(file, dtype=np.int64)
        array_withchange.append(xx)
    arr = np.array(array_withchange, dtype='int64')

    stacked  = np.vstack([arr, array_nochange])
    df = pd.DataFrame(stacked, columns =["blockID", "x", "y", "z", "xminpt", "yminpt", "zminpt", "xmaxpt", "ymaxpt", "zmaxpt", "nuc_svID", "nucID", "x_svID", "y_svID", "z_svID", "size_x_mip4", "size_y_mip4", "size_z_mip4"])
    df.to_csv(outputpath + '/' + '{}.csv'.format(name), index=False)


@queueable
def task_apply_size_threshold(df):
    df['vol'] = df['size_x_mip4'] * df['size_y_mip4'] * df['size_z_mip4'] # add vol column
    df_1 = df.loc[(df['size_x_mip4'] > thres_x_min) & (df['size_y_mip4'] > thres_y_min) & (df['size_z_mip4'] > thres_z_min)] 
    # apply max threshold if u want
    df_o = df_1.sort_values('vol') # sort based on vol column
    df_o = df_o[df_o['nucID'] != 0]
    df_o.to_csv(outputpath + '/' + '{}.csv'.format(final_product), index=False)
    # "blockID", "x", "y", "z", "xminpt", "yminpt", "zminpt", "xmaxpt", "ymaxpt", "zmaxpt", "nuc_svID", "nucID", "x_svID", "y_svID", "z_svID", "size_x_mip4", "size_y_mip4", "size_z_mip4", "vol"


def run_local(cmd, count_data=False): # recommended
    try:
        func = globals()[cmd]
    except Exception:
        print("Error: cmd only accepts 'task_get_nuc_info', 'task_merge_within_block', 'task_merge_across_block', 'task_apply_size_threshold'")

    tq = LocalTaskQueue(parallel=parallel_cpu)
    if func == task_get_nuc_info:
        if file_input is not None:
            with open(file_input) as fd:      
                txtdf = np.loadtxt(fd, dtype='int64', ndmin=1)
                tq.insert( partial(func, i) for i in txtdf )
        else:
            tq.insert(( partial(func, i) for i in range(start, len(block_centers)) )) # NEW SCHOOL
    elif func == task_merge_within_block:
        if count_data == True:
            countdir = outputpath + '/' + 'count_{}'.format(cmd.split('_', 1)[1])
            os.makedirs(countdir, exist_ok=True)
            tq.insert(( partial(func, i, count_data, countdir) for i in range(start, len(block_centers)) ))
        else:
            tq.insert(( partial(func, i, count_data) for i in range(start, len(block_centers)) ))
    elif func == task_merge_across_block:
        nuc_data = [] # store input
        for ii in range(len(block_centers)):
            z = np.fromfile(outputpath + '/' + 'block2_{}.bin'.format(str(ii)), dtype=np.int64) 
            # z has [block id, center location in mip0, bbox min, bbox max, nuc_segid, nucid, nuc_xyz in mip0] in int64
            nuc_data.append(z.reshape(int(len(z)/15),15))
        r = np.concatenate(nuc_data)
        r2 = r[~np.all(r == 0, axis=1)] # reomve all zero rows
        u_across, c_across = np.unique(r2[:,11], return_counts=True)
        nucID_duplicated_across = u_across[c_across > 1]
        row_nochange = r[np.isin(r[:,11], u_across[c_across == 1])]
        keep = update_bbox_to_mip0(row_nochange).astype('int64')

        mergeddir = outputpath + '/' + 'merged_across_block'
        os.makedirs(mergeddir, exist_ok=True)
        
        tq.insert( partial(func, n, r , mergeddir) for n in nucID_duplicated_across)
        tq.insert(partial(save_merged, mergeddir, keep, 'merged'))
        if count_data == True:
            tq.insert(partial(save_count, c_across, cmd.split('_', 1)[1])) # save count_data
    else: # task_apply_size_threshold
        previous_df = pd.read_csv(outputpath + '/' + 'merged.csv', header=0)
        # no file means you haven't merged
        tq.insert(partial(func, previous_df))

    tq.execute(progress=True)
    print('Done')
