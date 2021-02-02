''' Quick methods for getting voxel/absolute coordinates from neuroglancer/CATMAID urls.'''
import requests
from cloudvolume import CloudVolume
import numpy as np
from concurrent import futures
from pathlib import Path
import json
from annotationframeworkclient import FrameworkClient


def setup_credentials(tokens,segmentations,overwrite=False):
    ''' Setup the api keys and segmentation links in ~/cloudvolume. 
    Args:
        tokens: dict, hex string api tokens. The other modules currently use two. 'api' and 'dev' and both are currently necessary. api
                is used for generating an annotation framework client, and dev is used for creating a cloudvolume object.
        segmentations: dict, segmentation paths and respective resolutions. Format is {'segmentation_name':{'url':'path_to_segmentation','resolution':'[x,y,z]'}}' '''


    BASE = Path.home() / 'cloudvolume'


    if Path.exists(BASE / 'secrets'):
        if Path.exists(BASE / 'secrets' / 'chunkedgraph-secret.json') and overwrite is False:
            print('credentials exist')
        else:
            with open(Path.home() / 'cloudvolume' / 'secrets'/'chunkedgraph-secret.json',mode='w') as f:
                json.dump(tokens,f)
        print('credentials created')

    else: 
        Path.mkdir(BASE / 'secrets', parents=True)
        with open(Path.home() / 'cloudvolume' / 'secrets'/'chunkedgraph-secret.json',mode='w') as f:
                json.dump(tokens,f)
        print('credentials created')

    if not Path.exists(BASE / 'segmentations.json'):
        with open(BASE / 'segmentations.json',mode='w') as f:
            json.dump(segmentations,f)
        print('setup complete')



def get_client():
    ''' Establish an ngl client for interacting with the annotation framework. This requires you to have ~/cloudvolume/secrets/chunkedgraph-secret.json set up with a token
    for both the dev and api servers. Use .generate_token(api_token,dev_token) to set this up if you do not have it already.
    Returns: 
        client, FrameworkClient object
        tokens, dict, hex string api tokens for dev and api servers.'''
    if Path.exists(Path.home() / 'cloudvolume' / 'secrets'/'chunkedgraph-secret.json'):
        with open(Path.home() / 'cloudvolume' / 'secrets'/'chunkedgraph-secret.json') as f:
                tokens = json.load(f)
    else:
        raise ValueError('chunkedgraph-secret.json does not exist. Does ~/cloudvolume/secrets?')
    
    if isinstance(tokens,dict):
        # This token is for accessing the cloud volume for the V1 chunkedgraph. 
        dev_token = tokens['dev']
        # This token is for interacting with the annotation framework.
        auth_token = tokens['api']
    else:
        raise ValueError('For now we need an api token and a dev token. Add both to chunkedgraph-secret.json')

        
    datastack_name = 'vnc_v0' # from https://api.zetta.ai/wclee/info/

    client = FrameworkClient(
        datastack_name,
        server_address = "https://api.zetta.ai/wclee",
        auth_token = auth_token
    )
    return(client,{'dev':dev_token,'api':auth_token})


def get_token(version='dev'):
    fname = Path.home() / 'cloudvolume' / 'secrets' / 'chunkedgraph-secret.json'
    with open(fname) as f:
        keys = json.load(f)
    return(keys[version])

def get_cv_path(version=None):
    fname = Path.home() / 'cloudvolume' / 'segmentations.json'
    with open(fname) as f:
        paths = json.load(f)
    
    if version is None:
        return(paths)
    else:
        return(paths[version])


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
        pts (list): list of 3-element np.arrays of MIP0 coordinates
        vol_url (str): cloud volume url
    Returns:
        list, segment_ID at specified point '''
    
    
    seg_mip = vol.scale['resolution']
    res = seg_mip / image_res

    pts_scaled = [pt // res for pt in pts]
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
    
    try:
        return(r.json())
    except:
        return(r)
    
  




