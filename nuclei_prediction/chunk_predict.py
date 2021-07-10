import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import csv
from tqdm import tqdm

from cloudvolume import CloudVolume, view
import cc3d
from tifffile.tifffile import imwrite

sys.path.append(os.path.abspath("../segmentation"))

import rootID_lookup as IDlook
import authentication_utils as auth


output=[]
cv = CloudVolume(auth.get_cv_path('Image')['url'], use_https=True, agglomerate=False)
seg = CloudVolume(auth.get_cv_path('FANC_production_segmentation')['url'], use_https=True, agglomerate=False, cache=True)

[X,Y,Z]=cv.mip_volume_size(0)


step_xy = 128*2**4 # width of each chunk = x or y space between each chunk center in mip0
step_z = 256 # depth of each chunk = z space between each chunk center in mip0

start_xy = 128*2**(4-1) # first chunk center
start_z = 256*2**(-1) # first chunk center

centerX = np.arange(start_xy, X, step_xy)
centerY = np.arange(start_xy, Y, step_xy)
centerZ = np.arange(start_z, Z, step_z)

# looks okay but there can be only a few space < step/2 at the end of these sequences, causing error when making chunks
if (X - centerX[-1]) < start_xy:
    np.put(centerX, -1, X-start_xy)
else:
    centerX = np.append(centerX, X-start_xy)

if (Y - centerY[-1]) < start_xy:
    np.put(centerY, -1, Y-start_xy)
else:
    centerY = np.append(centerY, Y-start_xy)

if (Z - centerZ[-1]) < start_z:
    np.put(centerZ, -1, Z-start_z)
else:
    centerZ = np.append(centerZ, Z-start_z)

# make nx3 arrays of the chunk center coordinates
chunk_center = np.array(np.meshgrid(centerX, centerY, centerZ), dtype='uint32').T.reshape(-1,3)

x_thres = 33-10 # 50/(4.3*2^4/45) = 50/1.53
y_thres = 33-10
z_thres = 50-10

def mybbox(img):

    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax

nuclei_cv = CloudVolume(
    auth.get_cv_path('nuclei_map')['url'],
    progress=False,
    cache=True, # cache to disk to avoid repeated downloads
    use_https=True,
    autocrop=True,
    bounded=False
)

for i in tqdm(range(len(chunk_center))):
    nuclei = nuclei_cv.download_point(chunk_center[i], mip=[68.8,68.8,45.0], size=(128, 128, 256))
    mask_temp = nuclei[:,:,:]
    mask = np.where(mask_temp > 0.5, 1, 0)  

    # print(mask.shape) 
    # (128, 128, 256, 1)
    mask_s = np.squeeze(mask)
    cc_out, N = cc3d.connected_components(mask_s, return_N=True, connectivity=26) # free

    list=[]
    for segid in range(1, N+1):
        extracted_image = cc_out * (cc_out == segid)
        bbox = mybbox(extracted_image)
        list.append(bbox)

    list2=[]
    for segid in range(0, N):
    xwidth = list[segid][1] - list[segid][0]
    ywidth = list[segid][3] - list[segid][2]
    zwidth = list[segid][5] - list[segid][4]
    if xwidth >= x_thres and ywidth >= y_thres and zwidth >= z_thres:
        center = ((list[segid][1] + list[segid][0])/2,
        (list[segid][3] + list[segid][2])/2,
        (list[segid][5] + list[segid][4])/2)
        list2.append(center)
    else:
        pass

    if len(list2):
        origin = nuclei.bounds.minpt # 3072,5248,1792
        cell_body_coordinates_mip4 = np.add(np.array(list2), origin)
        cell_body_coordinates = cell_body_coordinates_mip4
        cell_body_coordinates[:,0]  = (cell_body_coordinates_mip4[:,0] * 2**4)
        cell_body_coordinates[:,1]  = (cell_body_coordinates_mip4[:,1] * 2**4)
        cell_body_coordinates = cell_body_coordinates.astype('uint32')

        cell_body_IDs = IDlook.segIDs_from_pts_cv(pts=cell_body_coordinates, cv=seg) #mip0
        nuclei_cv.cache.flush()
        cell_body_IDs_list = cell_body_IDs.tolist()
        output.append(cell_body_IDs_list)
    else:
        pass

sum = sum(output,[])
output_s = set(sum)
output_str = [str(n) for n in output_s]
output_2D = np.array(output_str ).reshape(len(output_str ),1).tolist()

with open('./output.csv', 'w') as result:
    writer = csv.writer(result)
    writer.writerows(output_2D)
