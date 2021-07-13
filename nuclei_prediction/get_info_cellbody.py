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
# libraries 2
# %pwd
sys.path.append(os.path.abspath("../segmentation"))
# to import rootID_lookup and authentication_utils like below

import rootID_lookup as IDlook
import authentication_utils as auth
# cv setting
cv = CloudVolume(auth.get_cv_path('Image')['url'], use_https=True, agglomerate=False)

nuclei_cv = CloudVolume(
    auth.get_cv_path('nuclei_map')['url'],
    progress=False,
    cache=True, # cache to disk to avoid repeated downloads
    use_https=True,
    autocrop=True,
    bounded=False
)

seg = CloudVolume(auth.get_cv_path('FANC_production_segmentation')['url'], use_https=True, agglomerate=False, cache=True)
[X,Y,Z]=cv.mip_volume_size(0)

step_xy = 128*2**4 # width of each chunk = x or y space between each chunk center in mip0
step_z = 256 # depth of each chunk = z space between each chunk center in mip0

start_x = 128*2**(4-1) # first chunk center
start_y = 128*2**(4-1) + 73728 # step_xy*36=73728
start_z = 256*2**(-1) +10  # 10 is offset

centerX = np.arange(start_x, X, step_xy)
centerY = np.arange(start_y, Y, step_xy)
centerZ = np.arange(start_z, Z, step_z)

if (X - centerX[-1]) < start_x:
    np.put(centerX, -1, X-start_x)
else:
    centerX = np.append(centerX, X-start_x)

if (Y - centerY[-1]) < start_y:
    np.put(centerY, -1, Y-start_y)
else:
    centerY = np.append(centerY, Y-start_y)

if (Z - centerZ[-1]) < start_z:
    np.put(centerZ, -1, Z-start_z)
else:
    centerZ = np.append(centerZ, Z-start_z)

chunk_center = np.array(np.meshgrid(centerX, centerY, centerZ), dtype='int64').T.reshape(-1,3)
len(chunk_center)
output=[]

x_thres = 33-10 # 50/(4.3*2^4/45) = 50/1.53
y_thres = 33-10
z_thres = 50-10

connectivity = 26

def mybbox(img):

    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax

# for loop
for i in tqdm(range(len(chunk_center))):
    nuclei = nuclei_cv.download_point(chunk_center[i], mip=[68.8,68.8,45.0], size=(128, 128, 256) ) # mip0 and 4 only
    mask_temp = nuclei[:,:,:]
    mask = np.where(mask_temp > 0.5, 1, 0)  
    mask_s = np.squeeze(mask)

    cc_out, N = cc3d.connected_components(mask_s, return_N=True, connectivity=connectivity) # free

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

    if len(list2): # segIDs_from_pts_cv makes error is there is none in list2
        origin = nuclei.bounds.minpt # 3072,5248,1792
        cell_body_coordinates_mip4 = np.add(np.array(list2), origin)
        cell_body_coordinates = cell_body_coordinates_mip4
        cell_body_coordinates[:,0]  = (cell_body_coordinates_mip4[:,0] * 2**4)
        cell_body_coordinates[:,1]  = (cell_body_coordinates_mip4[:,1] * 2**4)
        cell_body_coordinates = cell_body_coordinates.astype('int64')

        # Lets get IDs using cell_body_coordinates
        cell_body_IDs = IDlook.segIDs_from_pts_cv(pts=cell_body_coordinates, cv=seg) #mip0

        # save
        # type(cell_body_coordinates.shape)
        cord_pd = pd.DataFrame(cell_body_coordinates, columns=["x", "y", "z"])
        temp = cord_pd
        temp['segIDs'] = cell_body_IDs
        output.append(temp)
    else:
        pass

    nuclei_cv.cache.flush()

output_appended = pd.concat(output)
output_appended
output_s = output_appended.drop_duplicates(keep='first', subset='segIDs')
output_s
output_s.to_csv('../Output/cellbody_cord_id.csv', index=False)
