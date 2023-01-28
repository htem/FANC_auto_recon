#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pymaid

default_server = 'https://radagast.hms.harvard.edu/catmaidvnc'
default_project_id = 13
default_credentials_file = Path.home() / '.cloudvolume' / 'secrets' / 'catmaid_keys.json'
default_key_name = 'FANC'


def save_catmaid_credentials(api_key,
                             catmaid_url=default_server,
                             project_id=default_project_id,
                             credentials_file=default_credentials_file,
                             key_name=default_key_name,
                             overwrite=False):
    """
    Save CATMAID API key to file in a format that can be read by
    `fanc.catmaid.catmaid_login`.

    Most users in the FANC community can leave all the default arguments,
    and just call `save_catmaid_credentials(your_api_key)`. You can get
    your API key by logging into https://radagast.hms.harvard.edu/catmaidvnc
    then hovering over "You are [Your Name]" in the top-right corner, then
    clicking "Get API token"
    """
    credentials = {
        'catmaid_server_url': catmaid_url,
        'catmaid_api_keys': {
            key_name: api_key
        },
        'default_api_key_name': key_name,
        'default_project_id': project_id,

    }

    credentials_file = Path(credentials_file).expanduser()
    if not credentials_file.exists() or overwrite is True:
        credentials_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        credentials_file.touch(mode=0o600)
        with open(credentials_file, 'w') as f:
            json.dump(credentials, f, indent=4)
        print(f'CATMAD credentials saved to {credentials_file}')
    else:
        raise FileExistsError(f'{credentials_file} already exists and overwrite is False')


def connect(credentials_file=default_credentials_file,
            key_name=None,
            project_id=None):
    """
    Connect to catmaid using settings specified in a credentials file.

    Parameters
    ----------
    credentials_file: str or pathlib.Path (default '~/.cloudvolume/secrets/catmaid_keys.json')
        Path of where your CATMAID credentials are stored. (You must first
        run `save_catmaid_credentials` to generate this file.)
    key_name: str (default None)
        Identifier of the API key to use. If not specified, key_name is read
        from 'default_api_key_name' in the credentials_file.
    project_id: int (default None)
        ID of the project to connect to. If not specified, project_id is read
        from 'default_project_id' in the credentials_file.

    Returns
    -------
    A `pymaid.CatmaidInstance` connected to the specified CATMAID server and project
    """
    if not Path(credentials_file).exists():
        raise FileNotFoundError(
            f'No file {credentials_file}. Use `fanc.catmaid.save_catmaid_credentials()`'
            ' or specify the `credentials_file` argument.')
    with open(credentials_file) as f:
        credentials = json.load(f)

    if key_name is None:
        key_name = credentials['default_api_key_name']
    if project_id is None:
        project_id = credentials['default_project_id']

    return pymaid.CatmaidInstance(credentials['catmaid_server_url'],
                                  credentials['catmaid_api_keys'][key_name],
                                  project_id=project_id);


def upload_to_CATMAID(neuron,
                      target_project=None,
                      annotations=None,
                      NG_voxel_resolution=np.array([4.3, 4.3, 45]),
                      CM_voxel_resolution=np.array([4.3, 4.3, 45])):
    target_project = pymaid.utils._eval_remote_instance(target_project)

    neuron.nodes[['x', 'y', 'z']] = neuron.nodes[['x', 'y', 'z']] / NG_voxel_resolution
    neuron.nodes[['x', 'y', 'z']] = neuron.nodes[['x', 'y', 'z']] * CM_voxel_resolution

    upload_info = pymaid.upload.upload_neuron(neuron, source_type='skeleton', remote_instance=target_project,
                                              import_annotations=True)
    if annotations is not None:
        pymaid.add_annotations(upload_info['skeleton_id'], annotations)

    return upload_info
