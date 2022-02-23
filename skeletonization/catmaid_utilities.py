import numpy as np
from pathlib import Path
import pymaid
import json

def set_catmaid_credentials(catmaid_url,
                            api_key=None,
                            project_id=13,
                            username='default_user',
                            key_path='~/.cloudvolume/catmaid_keys.json',
                            overwrite=False):
    filepath = Path(key_path).expanduser()
    key_json = {
        "catmaid_account_api_keys": {
            username: api_key
        },
        "source_catmaid_url": catmaid_url,
        "source_catmaid_account_to_use": username,
        "source_project_id": project_id,

    }
    if not filepath.exists() or overwrite is True:
        Path(key_path).parent.mkdir(parents=True)
        Path(key_path).touch()
        with open(key_path, 'w') as f:
            json.dump(key_json, f)
    else:
        print(f'{key_path} already exists')


def catmaid_login(username='default_user',
                  project_id=13,
                  key_file_path=None):
    '''' Establish a CATMAID login instance for pulling data from a project. Usually this will run as part of a pull request, not directly.

    Parameters
    ----------
    username :    str, Username of catmaid account. Not necessary if you have only one api key. Default is 'default_user'.
    project_id:   int, Project ID in the case that there are multiple stacks in the catmaid instance. Default is 13 (FANC community project).
    key_file:     str, Full file name of the api keys for your catmaid project, check example for format. Default will be ~/.cloudvolume/secrets/catmaid_keys.json
    Returns
    -------
    myInstance:   A CATMAID login instance.
    '''
    if key_file_path is None:
        try:
            fname = Path.home() / '.cloudvolume' / 'catmaid_keys.json'
            with open(fname) as f:
                apikeys = json.load(f)
        except:
            print('Default location {} does not exist. Provide a path.'.format(fname))
    else:

        fname = key_file_path
        with open(fname, 'r') as f:
            apikeys = json.load(f)

    myInstance = pymaid.CatmaidInstance(apikeys["source_catmaid_url"],
                                        apikeys["catmaid_account_api_keys"][username],
                                        project_id=project_id);
    return (myInstance)


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

    return (upload_info)