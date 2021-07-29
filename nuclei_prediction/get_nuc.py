from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
import csv
from numpy.random.mtrand import f
from tqdm import tqdm
import argparse

from cloudvolume import CloudVolume, view, Bbox
import cc3d
from tifffile.tifffile import imagej_metadata, imwrite
from taskqueue import TaskQueue, queueable, LocalTaskQueue
from functools import partial
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

parser = argparse.ArgumentParser(description='get segIDs of cell bodies and save into csv files') 
parser.add_argument('-s', '--start', help='specify starting chunk. default is 0', default=0, type=int)
parser.add_argument('-l', '--lease', help='lease_seconds for TaskQueue.poll. specify in seconds. default is 1800sec', default=1800, type=int)
parser.add_argument('-p', '--parallel', help='number of cpu cores for parallel processing. default is 12', default=12, type=int)
parser.add_argument('-i', '--input', help='input the list of chunk numbers need recalculated again', type=validate_file)
parser.add_argument('-x', '--xyz', help='You have the list of nuclei info you want to try again? Use this one', type=validate_file)
parser.add_argument('-c', '--choose', help='specify the numer of pixels randomly chosen to get segID of nuclei. default is all pixels inside each nucleus', type=int)
args = parser.parse_args()

start=args.start
lease=args.lease
parallel_cpu=args.parallel
file_input=args.input
xyz_input=args.xyz
choose=args.choose

if (file_input is not None) & (xyz_input is not None):
    print("Error: Choose either --input or --xyz")
else:
    pass

# path
queuepath = '/n/groups/htem/users/skuroda/nuclei_tasks4'
# queuepath = '../Output/nuclei_tasks'
outputpath = '/n/groups/htem/users/skuroda/nuclei_output4'
# outputpath = '../Output'

# variables
np.random.seed(123)
block_x = 256 # block size in mip4 (72x72x45)
block_y = 256
block_z = 256
skip_y = block_y*288 # 73728, this will ignore y < skip_y
thres_s = 0.5 # signal threshold, > thrreshold to generate nuclei_seg_cv=0.2
thres_x = 33-10 # size threshold for xyz in mip4 (68.8x68.8x45)
thres_y = 33-10 # 50/(4.3*2^4/45) = 50/1.53??
thres_z = 50-10 
connectivity = 26 # number of nearby voxels to look at for calculating connective components

# name
merged_product = 'merged'
final_product = 'nuc_info'

# could-volume url setting
cv = CloudVolume(auth.get_cv_path('Image')['url'], use_https=True, agglomerate=False) # mip0
seg = CloudVolume(auth.get_cv_path('FANC_production_segmentation')['url'], use_https=True, agglomerate=False, cache=False) # mip2
nuclei_cv = CloudVolume( # mip4
    auth.get_cv_path('nuclei_map_Jul2021')['url'],
    progress=False,
    cache=False, # to aviod conflicts with LocalTaskQueue
    use_https=True,
    autocrop=True, # crop exceeded volumes of request
    bounded=False
)
nuclei_seg_cv = CloudVolume( # mip4
    auth.get_cv_path('nuclei_seg_Jul2021')['url'],
    cache=False,
    progress=False,
    use_https=True # this is precomputed so no need to specify agglomerate=False
)

# calculate blocks in mip0
# cv.mip_volume_size(0)
# Vec(83968,223232,4390, dtype=int64) in mip0 (4.3x4.3x45)

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


def merge_bbox(array, xminpt=4, xmaxpt=7, row_saved=0):
    out = array[row_saved,:].copy()
    out[xminpt] = min(array[:,xminpt]) # xmin
    out[xminpt+1] = min(array[:,xminpt + 1]) # ymin
    out[xminpt+2] = min(array[:,xminpt + 2]) # zmin
    out[xmaxpt] = max(array[:,xmaxpt]) # xman
    out[xmaxpt+1] = max(array[:,xmaxpt + 1]) # ymax
    out[xmaxpt+2] = max(array[:,xmaxpt + 2]) # xmax

    return out


@queueable
def task_get_nuc_info(i): # use i = 7817 for test, close to [7953 118101 2584]
    try:
        if xyz_input is not None:
            xyzdf = pd.read_csv(xyz_input, header=0)
            xyz_mip0 = xyzdf.iloc[i,1:4] #xyz coordinates
            nuclei = nuclei_cv.download_point(xyz_mip0, mip=[68.8,68.8,45.0], size=(block_x, block_y, block_z) )
        else:
            nuclei = nuclei_cv.download_point(block_centers[i], mip=[68.8,68.8,45.0], size=(block_x, block_y, block_z) ) # download_point uses mip0 for pt

        mask_temp = nuclei[:,:,:]
        mask = np.where(mask_temp > thres_s, 1, 0)
        masked = np.squeeze(mask)

        cc_out, N = cc3d.connected_components(masked, return_N=True, connectivity=connectivity)

        mylist=[]
        for j in range(1, N+1):
            point_cloud = np.argwhere(cc_out == j)
            bbx = Bbox.from_points(point_cloud)
            mylist.append(np.append([j], [bbx.center(), bbx.minpt, bbx.maxpt]))

        mylist.append(np.zeros(10)) # convert 1D to 2D
        arr = np.array(mylist) # [cc id, center location, bbox min, bbox max] all in mip4

        if arr.ndim == 2:
            arr2 = np.hstack((arr.copy().astype('int64'), np.zeros((arr.shape[0],2), dtype='int64'))) # array to store output
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

                nucIDs_list = []
                for k in range(len(lrandom_mip4)):
                    nucID_temp = nuclei_seg_cv.download_point(lrandom_mip4[k], mip=[68.8,68.8,45.0], size=(1, 1, 1) ) # download_point uses mip4 for pt
                    nucIDs_list.append(np.squeeze(nucID_temp))
                nucID = find_most_frequent_ID(np.array(nucIDs_list, dtype='int64'))

                arr2[obj,1:4] = center_mip0 # change xyz from mip4 to mip0
                arr2[obj,10] = nuc_segID # insert
                arr2[obj,11] = nucID # insert
                arr2[obj,0] = i # no longer need ccid

        else:
            arr2 = np.zeros(12, dtype = 'int64')
            arr2[0] = i

        # arr2 has [block id, center location in mip0, bbox min, bbox max, nuc_segid, nucid] in int64
        x = arr2.astype(np.int64)
        if xyz_input is not None:
            x.tofile(outputpath + '/' + 'new_block_{}.bin'.format(str(i)))
        else:
            x.tofile(outputpath + '/' + 'block_{}.bin'.format(str(i)))
    except Exception as e:
        with open(outputpath + '/' + '{}.log'.format(str(i)), 'w') as logfile:
            print(e, file=logfile)


@queueable
def task_merge_within_bbox(i, clist):
    try:
        y = np.fromfile(outputpath + '/' + 'block_{}.bin'.format(str(i)), dtype=np.int64) # y has [block id, center location in mip0, bbox min, bbox max, nuc_segid, nucid] in int64
        if y.ndim == 1: # only one row
            c = 1
            y2 = y
        else: # more than two rows
            u, c = np.unique(y[:,11], return_counts=True)
            nucID_duplicated = u[c > 1]
            if len(nucID_duplicated):
                merged=[]
                for dup in range(len(nucID_duplicated)):
                    foo = y[(y[:,11] == nucID_duplicated[dup])]
                    bar = merge_bbox(foo)
                    merged.append(bar)

                y2 = np.vstack((np.array(merged), y[(y[:,11] == u[c == 1])]))
            else:
                y2 = y

        # y2 still has [block id, center location in mip0, bbox min, bbox max, nuc_segid, nucid] in int64
        y_out = y2.astype(np.int64)
        clist.append(c)
        if xyz_input is not None:
            y_out.tofile(outputpath + '/' + 'new_block2_{}.bin'.format(str(i)))
        else:
            y_out.tofile(outputpath + '/' + 'block2_{}.bin'.format(str(i)))
    except Exception as e:
        with open(outputpath + '/' + 'within_{}.log'.format(str(i)), 'w') as logfile:
            print(e, file=logfile)


@queueable
def save_count_data(list_input, func, name):
    array_input = np.array(list_input, dtype='int64').flatten()
    sorted = np.sort(array_input)[::-1]
    if func == task_merge_within_bbox:
        sorted = sorted[0:len(array_input) - 1 - len(block_centers)] # every block stil has np.zeros(12) in task_merge_within_bbox
    np.savetxt(outputpath + '/' + 'count_{}.txt'.format(name), sorted)


@queueable
def task_merge_across_bbox(i, output, skipped, data):
    try:
        if xyz_input is not None:
            dup_info_0 = data[i,:]
        else:
            dup_info = data[data[:,11] == i]
            dup_info_0 = dup_info[0,:]

        loc_mip4 = np.array([(dup_info_0[1]/2**4),(dup_info_0[2]/2**4), dup_info_0[3]], dtype='int64')
        nuclei = nuclei_seg_cv.download_point(loc_mip4, mip=[68.8,68.8,45.0], size=(block_x*2, block_y*2, block_z*2) ) # download_point uses mip4 for pt, and use first row

        nuclei_temp = nuclei[:,:,:]
        nuclei_temp2 = np.where(nuclei_temp == dup_info_0[11], 1, 0)
        nuclei_temp3 = np.squeeze(nuclei_temp2)
        new_bbx = Bbox.from_points(nuclei_temp3)

        dup_info_0_new = dup_info_0.copy().astype('int64')
        dup_info_0_new[4:7] = new_bbx.minpt
        dup_info_0_new[7:10] = new_bbx.maxpt # [block id, center location in mip0, new bbox min, new bbox max, nuc_segid, nucid] in int64

        output.append(dup_info_0_new)
      
    except Exception as e:
        skipped.merge(dup_info_0)
        with open(outputpath + '/' + 'across_{}.log'.format(str(i)), 'w') as logfile:
            print(e, file=logfile)


@queueable
def array_to_csv(array_withchange, array_nochange, name, xyz_input): # name
    if xyz_input is not None:
        previous_df = pd.read_csv(outputpath + '/' + '{}.csv'.format(str(name)), header=0)
        new_df = pd.DatFrame(array_withchange)
        df = pd.concat([previous_df, new_df])
    else:
        stacked  = np.vstack([array_withchange, array_nochange])
        df = pd.DatFrame(stacked)
    df.to_csv(outputpath + '/' + '{}.csv'.format(str(name)), index=False)


@queueable
def save_skipped(list_input, name):
    array_input = np.array(list_input, dtype='int64')
    np.savetxt(outputpath + '/' + '{}.txt'.format(name), array_input)


# @queueable
# def task_apply_size_threshold(array, func):
#     array_input = np.array(list_input, dtype='int64').flatten()
#     sorted = np.sort(array_input)[::-1]
#     if cmd == task_merge_within_bbox:
#         sorted = sorted[0:len(array_input) - 1 - len(block_centers)] # every block stil has np.zeros(12)
#     np.savetxt(outputpath + '/' + 'count_{}.txt'.format(cmd.split('_', 1)), sorted)
#     a  = np.vstack((np.array(nuc_data_out), r[(r[:,11] == u_across[c_across == 1])])) # save csv as one single file


def run_local(cmd, count_data=False): # recommended
    try:
        func = globals()[cmd]
    except Exception:
        print("cmd only accepts 'task_get_nuc_info', 'task_merge_within_bbox', 'task_merge_across_bbox', 'task_apply_size_threshold'")

    tq = LocalTaskQueue(parallel=parallel_cpu)
    if func == task_get_nuc_info:
        if file_input is not None:
            with open(file_input) as fd:      
                txtdf = np.loadtxt(fd, dtype='int64')
                tq.insert( partial(func, i) for i in txtdf )
        elif xyz_input is not None:
            xyzdf = pd.read_csv(xyz_input, header=0)
            tq.insert(( partial(func, i) for i in range(len(xyzdf)) ))
        else:
            tq.insert(( partial(func, i) for i in range(start, len(block_centers)) )) # NEW SCHOOL
    elif func == task_merge_within_bbox:
        clist=[] # save count_data
        if file_input is not None:
            with open(file_input) as fd:      
                txtdf = np.loadtxt(fd, dtype='int64')
                tq.insert( partial(func, i, clist=clist) for i in txtdf )
        else:
            tq.insert(( partial(func, i, clist=clist) for i in range(start, len(block_centers)) ))
        if count_data == True:
            tq.insert(partial(save_count_data, clist, func, cmd.split('_', 1)[1]))
    elif func == task_merge_across_bbox:
        nuc_data_out = [] # store output
        skipped = [] # store skipped IDs
        if xyz_input is not None:
            xyzdf = pd.read_csv(xyz_input, header=0)
            row_nochange = np.zeros(12,dtype='int64')
            tq.insert(( partial(func, n, nuc_data_out, skipped, xyzdf) for n in range(len(xyzdf)) ))
            tq.insert(partial(save_skipped, skipped, 'skipped'))
        else:
            nuc_data = [] # store input
            for ii in range(len(block_centers)):
                z = np.fromfile(outputpath + '/' + 'block2_{}.bin'.format(str(ii)), dtype=np.int64) # z has [block id, center location in mip0, bbox min, bbox max, nuc_segid, nucid] in int64
                nuc_data.append(z)
            r = np.array(z)
            r2 = r[~np.all(r == 0, axis=1)] # reomve all zero rows
            u_across, c_across = np.unique(r2[:,11], return_counts=True)
            nucID_duplicated_across = u_across[c_across > 1]
            row_nochange = r[(r[:,11] == u_across[c_across == 1])]
            tq.insert( partial(func, n, nuc_data_out, skipped, r) for n in nucID_duplicated_across)
            tq.insert(partial(save_skipped, skipped, 'skipped'))
            if count_data == True:
                tq.insert(partial(save_count_data, c_across, func, cmd.split('_', 1)[1])) # save count_data
        tq.insert(partial(array_to_csv, (np.array(nuc_data_out)), row_nochange, merged_product, xyz_input))
    else: # task_apply_size_threshold
        previous_df = pd.read_csv(outputpath + '/' + '{}.csv'.format(merged_product), header=0)
        # tq inset
        # save
        # duplicate de error

    tq.execute(progress=True)
    print('Done')


def create_task_queue():
    tq = TaskQueue('fq://' + queuepath)
    if file_input is not None:
        with open(file_input) as fd:      
            txtdf = np.loadtxt(fd, dtype='int64')
            tq.insert(( partial(task_get_nuc_id, i) for i in txtdf ), parallel=parallel_cpu)
            print('Done adding {} tasks to queue at {}'.format(len(txtdf), queuepath))
    elif xyz_input is not None:
        xyzdf = pd.read_csv(xyz_input, header=0)
        tq.insert(( partial(task_get_nuc_id, i) for i in range(len(xyzdf)) ), parallel=parallel_cpu)
        print('Done adding {} tasks to queue at {}'.format(len(xyzdf), queuepath))
    else:
        tq.insert(( partial(task_get_nuc_id, i) for i in range(len(block_centers)) ), parallel=parallel_cpu) # NEW SCHOOL?
        print('Done adding {} tasks to queue at {}'.format(len(block_centers), queuepath))
        
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