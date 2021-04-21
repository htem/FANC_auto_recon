import numpy as np
import os
import requests
import tqdm

def fanc4_to_3(points,scale=2):
    ''' Convert from realigned dataset to original coordinate space.
    Args:
             points: an nx3 array of mip0 voxel coordinates
             scale:  selects the granularity of the field being used, but does not change the units.
    
    Returns: a dictionary of transformed x/y/z values and the dx/dy/dz values'''
             
    base = "https://spine.janelia.org/app/transform-service/dataset/fanc_v4_to_v3/s/{}".format(scale)
                      
    if len(np.shape(points)) > 1:
        full_url = base + '/values_array'
        points_dict = {'x': list(points[:,0]),'y':list(points[:,1]),'z':list(points[:,2])}
        r = requests.post(full_url, json = points_dict)
    else:
        full_url = base + '/' + 'z/{}/'.format(str(int(points[2]))) + 'x/{}/'.format(str(int(points[0]))) + 'y/{}/'.format(str(int(points[1])))
        r = requests.get(full_url)
    
    try:
        return(r.json())
    except:
        return(r)