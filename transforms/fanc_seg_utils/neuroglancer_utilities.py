''' Quick methods for getting voxel/absolute coordinates from neuroglancer/CATMAID urls.  This is by no means the best way to do this. It is just for rapid searching between datasets.'''
import requests
from cloudvolume import CloudVolume
import numpy as np


def get_point(vol, pt):
    """Download MIP0 point from a CloudVolume
    
    Args:
        vol (CloudVolume): CloudVolume at any MIP
        pt (np.array): 3-element point defined at MIP0
        
    Returns:
        CloudVolume element at location
    """
    mip = vol.mip
    res = np.array([2**mip, 2**mip, 1])
    print(res)
    return vol[list(pt // res)] 


def to_vec(v):
    """Format CloudVolume element as vector
    
    Args:
        v (np.array): 4D element as np.int16
        
    Returns:
        np.array, 3D vector as np.float
    """
    return np.array([np.float32(v[0,0,0,1]) / 4, np.float32(v[0,0,0,0]) / 4, 0])


def get_vec(vol, pt):
    """Download vector at location defined at MIP0 from CloudVolume at any MIP
    
    Args:
        vol (CloudVolume): CloudVolume of field as int16
        pt (np.array): 3-element point
        
    Returns:
        np.array, 3D vector at MIP0
    """
    return to_vec(get_point(vol, pt))   



def seg_from_pt(pt,vol_url=None,seg_mip=None,image_res=None):
    ''' Get segment ID at a point. Default volume is the static segmentation layer for now. 
    Args:
        pt (np.array): 3-element point at MIP0
        vol_url (str): cloud volume url
    Returns:
        int, segment_ID at specified point '''
    
    if vol_url is None:
        vol = CloudVolume('https://storage.googleapis.com/zetta_lee_fly_vnc_001_segmentation/vnc1_full_v3align_2/realigned_v1/seg/full_run_v1',
                          parallel=True,
                          progress=True,
                          cache=True)
    else:
        vol = CloudVolume(vol_url,
                          parallel=True,
                          progress=True,
                          cache=True)
        
    seg_mip = vol.scale['resolution']
    
    if image_res is None:
        image_res = np.array([4.3,4.3,45])
        
    res = seg_mip / image_res
    segpt = pt // res
    seg_id = vol[segpt[0],segpt[1],segpt[2]]
    return(int(seg_id))




def fanc4_to_3(points,scale=2):
    ''' Convert from realigned dataset to original coordinate space.
    Inputs:
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
    
    return(r.json())
    
  


def seg_ids_from_link(ng_link):
    segments = ng_link[ng_link.find('segments%22:%5B%22')+len('segments%22:%5B%22'):ng_link.find('%22%5D%2C%22')]    
    seg_ids = segments.split(sep = '%22%2C%22')
    return(np.array(seg_ids,dtype='int'))






def coords_from_catmaid_url(catmaid_url,voxel_dims = [4.3,4.3,45], return_voxels = False):
    ''' Input a catmaid url and extract the xyz coordinates. If return_voxels is true, returns voxel coords in mip0, else returns nm coords.
        The coordinates only work for fanc3'''
    
    if if not return_voxels:
        x_coord = float(catmaid_url[catmaid_url.find('xp=')+3:catmaid_url.find('&',catmaid_url.find('xp=')+3)]) / voxel_dims[0]
        y_coord = float(catmaid_url[catmaid_url.find('yp=')+3:catmaid_url.find('&',catmaid_url.find('yp=')+3)]) / voxel_dims[1]
        z_coord = float(catmaid_url[catmaid_url.find('zp=')+3:catmaid_url.find('&',catmaid_url.find('zp=')+3)]) / voxel_dims[2]
    else: 
        x_coord = float(catmaid_url[catmaid_url.find('xp=')+3:catmaid_url.find('&',catmaid_url.find('xp=')+3)])
        y_coord = float(catmaid_url[catmaid_url.find('yp=')+3:catmaid_url.find('&',catmaid_url.find('yp=')+3)]) 
        z_coord = float(catmaid_url[catmaid_url.find('zp=')+3:catmaid_url.find('&',catmaid_url.find('zp=')+3)])
    
    
    return x_coord,y_coord,z_coord
    

    
    
def coords_from_ng_url(ng_url,return_voxels= False):
    ''' Takes a long format neuroglancer link and extracts coordinates. Default is pixel coords in mip0. '''
    
    voxel_dims = ng_url[ng_url.find('voxelSize'):ng_url.find('voxelCoordinates')]
    x_dim = float(voxel_dims[voxel_dims.find('%5B')+3:voxel_dims.find('%2C')])
    y_dim = float(voxel_dims[voxel_dims.find('%2C',voxel_dims.find('%5B'))+3:voxel_dims.find('%2C',voxel_dims.find('%2C')+1)])
    z_dim = float(voxel_dims[voxel_dims.find('%2C',voxel_dims.find('%2C')+1)+3:voxel_dims.find('%5D')])

    voxel_coords = ng_url[ng_url.find('voxelCoordinates'):ng_url.find('zoomFactor')]
    
    if return_voxels is False:
        x_coord = float(voxel_coords[voxel_coords.find('%5B')+3:voxel_coords.find('%2C')])
        y_coord = float(voxel_coords[voxel_coords.find('%2C',voxel_coords.find('%5B'))+3:voxel_coords.find('%2C',voxel_coords.find('%2C')+1)])
        z_coord = float(voxel_coords[voxel_coords.find('%2C',voxel_coords.find('%2C')+1)+3:voxel_coords.find('%5D')])
    else:
        x_coord = float(voxel_coords[voxel_coords.find('%5B')+3:voxel_coords.find('%2C')]) * x_dim
        y_coord = float(voxel_coords[voxel_coords.find('%2C',voxel_coords.find('%5B'))+3:voxel_coords.find('%2C',voxel_coords.find('%2C')+1)]) * y_dim
        z_coord = float(voxel_coords[voxel_coords.find('%2C',voxel_coords.find('%2C')+1)+3:voxel_coords.find('%5D')]) * z_dim

    return x_coord, y_coord, z_coord, [x_dim,y_dim,z_dim]




def CM_from_NG(ng_url,catmaid_pid = 60, zoom_factor = 1,transform = True, voxel_dims = [4.3,4.3,45]):
    ''' Navigate to a region in a CATMAID instance from a Neuroglancer link. transform = True will transform the points from fanc4 to fanc3'''
    
    base_url = 'https://catmaid3.hms.harvard.edu/catmaidvnc/'
    pid = '?pid=' + str(catmaid_pid)

    [x,y,z,vd] = coords_from_ng_url(ng_url)
    
    if transform is True:
        transformed_points = fanc4_to_3(np.array([x,y,z]),scale=2)     
        x_comp = '&xp=' + str(transformed_points['x']* voxel_dims[0]) 
        y_comp = '&yp=' + str(transformed_points['y']* voxel_dims[1]) 
        z_comp = '&zp=' + str(transformed_points['z']* voxel_dims[2]) 
    else:
        x_comp = '&xp=' + str(x * voxel_dims[0])
        y_comp = '&yp=' + str(y * voxel_dims[1])
        z_comp = '&zp=' + str(z * voxel_dims[2])
    
    active_comps = '&tool=tracingtool&sid0=10'
    
    zoom_comp = '&s0=' + str(zoom_factor)
    
    catmaid_url = base_url + pid + z_comp + y_comp + x_comp + active_comps + zoom_comp
    
    return catmaid_url

