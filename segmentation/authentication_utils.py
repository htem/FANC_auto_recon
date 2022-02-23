import os
from caveclient import CAVEclient
from cloudvolume import CloudVolume
import json
from pathlib import Path

CAVE_DATASETS = {'production': 'fanc_production_mar2021',
                 'sandbox': 'fanc_sandbox'}


#def set_fafbseg_datasets(production='fanc_production_mar2021', sandbox='fanc_sandbox'):
#    utils.CAVE_DATASETS['production'] = production
#   utils.CAVE_DATASETS['sandbox'] = sandbox


def set_cave_credentials(token, datastack_name, overwrite=False):
    client = CAVEclient()
    client.auth.save_token(token, token_key=datastack_name, overwrite=overwrite)

    return f'Token succesfully stored at: ~/.cloudvolume/secrets/cave-secret.json under key {datastack_name}'


def get_caveclient(dataset='production'):
    datastack_name = CAVE_DATASETS[dataset]
    return CAVEclient(datastack_name, auth_token_key=datastack_name)


def get_chunkedgraph_secret(domain='fanc_production_mar2021'):
    return utils.get_chunkedgraph_secret(domain)


def get_cloudvolume(client=None, dataset='production'):
    datastack=CAVE_DATASETS[dataset]
    if client is None:
        client = CAVEclient(datastack)

    return CloudVolume(client.info.get_datastack_info()['segmentation_source'], use_https=True,
                       secrets=client.auth.token)


# PRE CAVE CLIENT CODE.
# Setting up your environment this way is no longer necessary with the release of CAVEclient.
def get_cv_path(version=None):
    fname = Path.home() / '.cloudvolume' / 'segmentations.json'
    with open(fname) as f:
        paths = json.load(f)

    if version is None:
        return (paths)
    else:
        return (paths[version])


def add_path(path_name, path):
    ''' Add a path to ./cloudvolume/segmentations.json

    args:
        path:   dict, dict of path info in form {'path_name':{'url': 'graphene://https://segmentation_path','resolution':[4.3,4.3,45]}}
        '''
    segmentation_file = Path.home() / '.cloudvolume/segmentations.json'
    if Path.exists(segmentation_file):
        with open(segmentation_file, 'r+') as f:
            segmentations = json.load(f)

        segmentations[path_name] = path
        json.dump(segmentations, segmentation_file)
    else:
        return '.cloudvolume/segmentations.json does not exist. Set up credentials first.'

    return 'Segmentation list updated'


def setup_credentials(tokens=None, segmentations=None, overwrite=False):
    ''' Setup the api keys and segmentation links in ~/cloudvolume.
    Args:
        token: str, auth token for chunk graph.
        segmentations: dict, segmentation paths and respective resolutions. Format is {'segmentation_name':{'url':'path_to_segmentation','resolution':'[x,y,z]'}}' '''

    tokens = {'token': tokens}

    BASE = Path.home() / '.cloudvolume'

    if Path.exists(BASE / 'secrets') and tokens is not None:
        if Path.exists(BASE / 'secrets' / 'chunkedgraph-secret.json') and overwrite is False:
            print('credentials exist')
        else:
            with open(BASE / 'secrets' / 'chunkedgraph-secret.json', mode='w') as f:
                json.dump(tokens, f)
            print('credentials created')

    else:
        Path.mkdir(BASE / 'secrets', parents=True)
        with open(Path.home() / 'cloudvolume' / 'secrets' / 'chunkedgraph-secret.json', mode='w') as f:
            json.dump(tokens, f)
        print('credentials created')

    if not Path.exists(BASE / 'segmentations.json'):
        with open(BASE / 'segmentations.json', mode='w') as f:
            json.dump(segmentations, f)
    elif segmentations is not None and overwrite is True:

        add_path(BASE / 'segmentations.json', segmentations)

        print('setup complete')
