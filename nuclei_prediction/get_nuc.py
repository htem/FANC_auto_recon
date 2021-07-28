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
parser.add_argument('-x', '--xyz', help='You have the list of nuclei ID you want to try again? Use this one', type=validate_file)
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

# def merge_within_block(i):
# def merge_between_block(i):



@queueable
def task_get_nuc_info(i): # use i = 7817 for test, close to [7953 118101 2584]
    try:
        if xyz_input is not None:
            xyzdf = pd.read_csv(xyz_input, header=0)
            xyz_mip0 = xyzdf.iloc[i,0:3] #xyz coordinates
            nuclei = nuclei_cv.download_point(xyz_mip0, mip=[68.8,68.8,45.0], size=(block_x, block_y, block_z) )
        else:
            nuclei = nuclei_cv.download_point(block_centers[i], mip=[68.8,68.8,45.0], size=(block_x, block_y, block_z) ) # use mip0 for pt

        mask_temp = nuclei[:,:,:]
        mask = np.where(mask_temp > thres_s, 1, 0)
        masked = np.squeeze(mask)

        cc_out, N = cc3d.connected_components(masked, return_N=True, connectivity=connectivity)

        mylist=[]
        for j in range(1, N+1):
            point_cloud = np.argwhere(cc_out == j)
            bbx = Bbox.from_points(point_cloud)
            mylist.append(np.append([j], [bbx.center(), bbx.minpt, bbx.maxpt]))

        arr = np.array(mylist) # [cc id, center location, bbox min, bbox max] all in mip4

        if len(arr):
            arr2 = np.hstack((arr.copy().astype('int64'), np.zeros((arr.shape[0],2), dtype='int64'))) # array to store output
            for obj in range(len(arr)):
                center_mip0 = mip4_to_mip0_array(arr[obj,1:4], nuclei)
                vinside = np.argwhere(cc_out == int(arr[obj,0]))

                if choose is not None: # random selection
                    index = np.random.choice(vinside.shape[0], size=min(len(vinside), choose), replace=False)
                    lrandom = vinside[index]
                else:
                    lrandom = vinside

                lrandom_mip0 = np.apply_along_axis(mip4_to_mip0_array, 1, lrandom, nuclei)
                lrandom_mip4 = lrandom + nuclei.bounds.minpt

                segIDs = IDlook.segIDs_from_pts_cv(pts=lrandom_mip0, cv=seg, progress=False) # use mip0 for pt
                nuc_segID = find_most_frequent_ID(segIDs)

                nucIDs_list = []
                for k in range(len(lrandom_mip4)):
                    nucID_temp = nuclei_seg_cv.download_point(lrandom_mip4[k], mip=[68.8,68.8,45.0], size=(1, 1, 1) ) # use mip4 for pt
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
def task_get_segid(i):
    try:
        output=[]
        if xyz_input is not None:
            xyzdf = pd.read_csv(xyz_input, header=0)
            xyz_mip0 = xyzdf.iloc[i,0:3] #xyz coordinates
            nuclei = nuclei_cv.download_point(xyz_mip0, mip=[68.8,68.8,45.0], size=(128, 128, 256) )
        else:
            nuclei = nuclei_cv.download_point(block_centers[i], mip=[68.8,68.8,45.0], size=(128, 128, 256) ) # mip0 and 4 only

        mask_temp = nuclei[:,:,:]
        mask = np.where(mask_temp > 0.5, 1, 0)
        mask_s = np.squeeze(mask)

        cc_out, N = cc3d.connected_components(mask_s, return_N=True, connectivity=connectivity) # free

        mylist=[]
        for j in range(1, N+1):
            point_cloud = np.argwhere(cc_out == j)
            bbx = Bbox.from_points(point_cloud)
            if (bbx.size3()[0] >= x_thres) & (bbx.size3()[1] >= y_thres) & (bbx.size3()[2] >= z_thres):
                mylist.append(np.append(bbx.center(), j))
            else:
                pass
        
        arr = np.array(mylist)

        if len(mylist):
            for segid in range(len(arr)):
                center = mip4_to_mip0_array(arr[segid,:], nuclei)
                vinside_mip4 = np.argwhere(cc_out == int(arr[segid,3]))
                vinside = np.apply_along_axis(mip4_to_mip0_array, 1, vinside_mip4, nuclei)

                #random selection?
                if choose == 0:
                    location_random = vinside
                else:
                    index = np.random.choice(vinside.shape[0], size=min(len(vinside.shape[0]), choose), replace=False)
                    location_random = vinside[index]

                # Lets get IDs using cell_body_coordinates
                cell_body_IDs = IDlook.segIDs_from_pts_cv(pts=location_random, cv=seg, progress=False) #mip0

                uniqueID, count = np.unique(cell_body_IDs, return_counts=True)
                unsorted_max_indices = np.argsort(-count)
                topIDs = uniqueID[unsorted_max_indices] 
                topIDs2 = topIDs[~(topIDs == 0)] # no zero
                topIDs2 = np.append(topIDs2, np.zeros(1, dtype = 'uint64'))
                nucID = topIDs2.astype('int64')[0]

                # save
                # type(cell_body_coordinates.shape)
                cord_pd = pd.DataFrame(columns=["x", "y", "z"])
                cord_pd.loc[0] = center
                temp = cord_pd
                temp['segIDs'] = nucID
                output.append(temp)

        else:
            foo = np.zeros(3, dtype = 'int64')
            bar = np.zeros(1, dtype = 'int64')

            cord_pd = pd.DataFrame(columns=["x", "y", "z"])
            cord_pd.loc[0] = foo
            temp = cord_pd
            temp['segIDs'] = bar
            output.append(temp)

        output_appended = pd.concat(output)
        output_appended
        output_s = output_appended.drop_duplicates(keep='first', subset='segIDs')
        output_s
        name = str(i)
        if xyz_input is not None:
            output_s.to_csv(outputpath + '/' + 'new_nuc_{}.csv'.format(name), index=False)
        else:
            output_s.to_csv(outputpath + '/' + 'cellbody_cord_id_{}.csv'.format(name), index=False)
    except Exception as e:
        name=str(i)
        with open(outputpath + '/' + '{}.log'.format(name), 'w') as logfile:
            print(e, file=logfile)

# merge duplicted nucleus within block
# u, c = np.unique(arr2[:,11], return_counts=True)
# nucID_duplicated = u[c > 1]
# if len(nucID_duplicated):
# merged=[]
# for dup in range(len(nucID_duplicated)):
# foo = arr2[(arr2[:,11] == nucID_duplicated[dup])]
 # bar = merge_bbox(foo)
# merged.append(bar)

# arr3 = np.vstack((np.array(merged), arr2[(arr2[:,11] == u[c == 1])]))
# else:
# arr3 = arr2


def run_local(): # recommended
    tq = LocalTaskQueue(parallel=parallel_cpu)
    if file_input is not None:
        with open(file_input) as fd:      
            txtdf = np.loadtxt(fd, dtype='int64')
            tq.insert( partial(task_get_nuc_info, i) for i in txtdf )
    elif xyz_input is not None:
        xyzdf = pd.read_csv(xyz_input, header=0)
        tq.insert(( partial(task_get_nuc_info, i) for i in range(len(xyzdf)) ), parallel=parallel_cpu)
    else:
        tq.insert(( partial(task_get_nuc_info, i) for i in range(start, len(block_centers)) )) # NEW SCHOOL

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