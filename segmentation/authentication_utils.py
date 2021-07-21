import requests
from cloudvolume import CloudVolume
import numpy as np
from concurrent import futures
from pathlib import Path
import json
from annotationframeworkclient import FrameworkClient


def setup_credentials(tokens=None,segmentations=None,overwrite=False):
    ''' Setup the api keys and segmentation links in ~/cloudvolume. 
    Args:
        token: str, auth token for chunk graph. 
        segmentations: dict, segmentation paths and respective resolutions. Format is {'segmentation_name':{'url':'path_to_segmentation','resolution':'[x,y,z]'}}' '''

    tokens = {'token':tokens}
    
    BASE = Path.home() / '.cloudvolume'


    if Path.exists(BASE / 'secrets') and tokens is not None:
        if Path.exists(BASE / 'secrets' / 'chunkedgraph-secret.json') and overwrite is False:
            print('credentials exist')
        else:
            with open(BASE / 'secrets'/'chunkedgraph-secret.json',mode='w') as f:
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
    elif segmentations is not None and overwrite is True:
        
        add_path(BASE / 'segmentations.json',segmentations)
        
        print('setup complete')


def add_path(path_name,path):
    ''' Add a path to ./cloudvolume/segmentations.json
    
    args:
        path:   dict, dict of path info in form {'path_name':{'url': 'graphene://https://segmentation_path','resolution':[4.3,4.3,45]}}
        '''
    segmentation_file = Path.home() / '.cloudvolume/segmentations.json'
    if Path.exists(segmentation_file):
        with open(segmentation_file, 'r+') as f:
            segmentations = json.load(f)
        
        segmentations[path_name] = path
        json.dump(segmentations,segmentation_file)
    else:
        return '.cloudvolume/segmentations.json does not exist. Set up credentials first.'
    
    return 'Segmentation list updated'


def get_client(server_address = "https://api.zetta.ai/wclee", datastack_name = 'vnc_v0'):
    ''' Establish an ngl client for interacting with the annotation framework. 
    Returns: 
        client, FrameworkClient object
        token, str, graphene server token'''
    
    token = get_token()    
    datastack_name = datastack_name

    client = FrameworkClient(
        datastack_name,
        server_address = server_address,
        auth_token = token
    )
    return(client,token)


def get_token(SECRET_PATH=None):
    
    if SECRET_PATH is None:
        SECRET_PATH = Path.home() / '.cloudvolume' / 'secrets'/'chunkedgraph-secret.json'
    
    if Path.exists(SECRET_PATH):
        with open(SECRET_PATH) as f:
                token = json.load(f)['token']
    else:
        raise ValueError('{} does not exist.'.format(SECRET_PATH))
    
    return(token)

def update_token(token,SECRET_PATH=None):
    
    if isinstance(token,str):
        token = {'token':token}
        
    if SECRET_PATH is None:
        SECRET_PATH = Path.home() / '.cloudvolume' / 'secrets'/'chunkedgraph-secret.json'
    
    if Path.exists(SECRET_PATH):
        with open(SECRET_PATH,mode='w') as f:
                json.dump(token,f)
    else:
        raise ValueError('{} does not exist.'.format(SECRET_PATH))
    
    return('token updated')

def get_cv_path(version=None):
    fname = Path.home() / '.cloudvolume' / 'segmentations.json'
    with open(fname) as f:
        paths = json.load(f)
    
    if version is None:
        return(paths)
    else:
        return(paths[version])
    
def get_cv(segmentation = 'FANC_production_segmentation',
           use_https=True,
           agglomerate=False):
    
    return CloudVolume(get_cv_path(segmentation)['url'],use_https=True,agglomerate=agglomerate)

    