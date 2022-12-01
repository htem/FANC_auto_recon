import os
from caveclient import CAVEclient
from cloudvolume import CloudVolume
import json
from pathlib import Path

CAVE_DATASETS = {'production': 'fanc_production_mar2021',
                 'sandbox': 'fanc_sandbox'}


def set_cave_credentials(token, datastack_name, overwrite=False):
    client = CAVEclient()
    client.auth.save_token(token, token_key=datastack_name, overwrite=overwrite)

    return f'Token succesfully stored at: ~/.cloudvolume/secrets/cave-secret.json under key {datastack_name}'


def get_caveclient(dataset='production', auth_token_key=True):
    datastack_name = CAVE_DATASETS[dataset]
    if auth_token_key==True:
        return CAVEclient(datastack_name, auth_token_key=datastack_name)
    else:
        return CAVEclient(datastack_name)


def get_cloudvolume(client=None, dataset='production'):
    datastack=CAVE_DATASETS[dataset]
    if client is None:
        client = CAVEclient(datastack)

    return CloudVolume(client.info.get_datastack_info()['segmentation_source'], use_https=True,
                       secrets=client.auth.token)
