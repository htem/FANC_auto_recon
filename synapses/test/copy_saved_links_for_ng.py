import sys
import json
from secrets import token_hex

import numpy as np
#sys.path.append('../ground_truth/')
#import synapse_cutout_rois


def line_anno(pre, post):
    anno = {
        'pointA': list(int(x) for x in pre),
        'pointB': list(int(x) for x in post),
        'type': 'line',
        'id': token_hex(40)
    }
    return anno


def flip_xyz_zyx_convention(array):
    assert array.shape[1] == 6
    array[:, 0:3] = array[:, 2::-1]
    array[:, 3:6] = array[:, 5:2:-1]


def flip_pre_post_order(array):
    assert array.shape[1] == 6
    tmp = array[:, 0:3].copy()
    array[:, 0:3] = array[:, 3:6]
    array[:, 3:6] = tmp


if sys.argv[1].endswith('.npy'):
    #print('Mode 1: npy')
    # For opening .npy files saved from np.save
    links = np.load(sys.argv[1])

    # The .npy files Jasper generated on Feb 8 were saved in zyx, so flip them to xyz
    if True:
        flip_xyz_zyx_convention(links)
    # The .npy files Jasper generated on Feb 8 were saved in post-pre order
    if True:
        flip_pre_post_order(links)

    # The .npy files Jasper generated on Feb 8 are saved in nm, so convert to
    # units of voxels at (4, 4, 40) nm voxel size for easier entering into ng
    links = links / (4, 4, 40, 4, 4, 40)

    #links is now pre-post, xyz, in units of voxels at (4, 4, 40)nm

elif sys.argv[1].endswith('.csv'):
    #print('Mode 2: csv')
    # For opening ground truth annotation files
    links = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=np.uint16)

    # Ground truth annotations were saved in zyx, so flip them to xyz
    if True:
        flip_xyz_zyx_convention(links)

    # Ground truth annotations were saved in nm, so convert to units of voxels
    # at (4, 4, 40) nm voxel size for easier entering into ng
    links = links / (4, 4, 40, 4, 4, 40)

    #links is now pre-post, xyz, in units of voxels at (4, 4, 40)nm

else:
    #print('Mode 3: binary')
    # For opening binary files saved by ../detection/worker.py
    links = np.fromfile(sys.argv[1], dtype=np.int32).reshape(-1, 6)

    # The Feb 7 predictions were saved in post-pre order
    if True:
        flip_pre_post_order(links)

    # The Feb 7 predictions are in units of mip1 voxels ((8.6, 8.6, 45) nm)
    # so convert to mip0 voxels for easier entering into ng
    links = links * (2, 2, 1, 2, 2, 1)

    #links is now pre-post, xyz, in units of voxels at (4.3, 4.3, 45)nm

annotations = [line_anno(links[i, 0:3], links[i, 3:6]) for i in range(links.shape[0])]
print(json.dumps(annotations, indent=2))
try:
    import pyperclip
    answer = input("Want to copy the output to the clipboard? (Only works if "
                   "you're running this script on a local machine, not on a "
                   "server.) [y/n] ")
    if answer.lower() == 'y':
        print('Copying')
        pyperclip.copy(json.dumps(annotations))
except:
    print("Install pyperclip (pip install pyperclip) for the option to"
          "programmatically copy the output above to the clipboard")
