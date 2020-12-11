''' Quick methods for getting voxel/absolute coordinates from neuroglancer/CATMAID urls.  This is by no means the best way to do this. It is just for rapid searching between datasets.'''
import requests
from cloudvolume import CloudVolume
import numpy as np
from concurrent import futures

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



def seg_from_pt(pts,vol,image_res=np.array([4.3,4.3,45]),max_workers=4):
    ''' Get segment ID at a point. Default volume is the static segmentation layer for now. 
    Args:
        pt (np.array): 3-element point at MIP0
        vol_url (str): cloud volume url
    Returns:
        list, segment_ID at specified point '''
    
    
    seg_mip = vol.scale['resolution']
        
    
    res = seg_mip / image_res
    if np.size(np.shape(pts)) < 2:
        pts = [pts]
        
    pts_scaled = [pt // res for pt in pts]
    results = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        point_futures = [ex.submit(lambda pt,vol: vol[list(pt)][0][0][0][0], k,vol) for k in pts_scaled]
        
        for f in futures.as_completed(point_futures):
            results=[f.result() for f in point_futures]
       
        

    return results






def seg_from_pt_graphene(pts,vol_url,image_res=np.array([4.3,4.3,45]), max_workers=4):
    """Get SegIDs from a list of points from a graphene volume object
    
    Args:
      pts: np.array, nx3 mip0 coords
      vol: cloudvolume, graphene version
      image_res: np.array, resolution of the image volume. default is [4.3,4.3,45]
      max_workers: int,the max number of workers for parallel chunk requests.
    
    Returns:
      (points, data): parallel Numpy arrays of the requested points from all
          cumulative calls to add_points, and the corresponding data loaded from
          volume.
    """
    vol= CloudVolume(vol_url,use_https=True,agglomerate=True)
    seg_mip = vol.scale['resolution']
        
    res = seg_mip / image_res
    pts_scaled = [pt // res for pt in pts]
    print(pts_scaled)
    
    results = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        point_futures = [ex.submit(lambda pt,vol: vol[list(pt)][0][0][0][0], k,vol) for k in pts_scaled]
        
        for f in futures.as_completed(point_futures):
            results=[f.result() for f in point_futures]
       
        

    return results



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
    
    return(r.json())
    
  


def seg_ids_from_link(ng_link):
    segments = ng_link[ng_link.find('segments%22:%5B%22')+len('segments%22:%5B%22'):ng_link.find('%22%5D%2C%22')]    
    seg_ids = segments.split(sep = '%22%2C%22')
    return(np.array(seg_ids,dtype='int'))




