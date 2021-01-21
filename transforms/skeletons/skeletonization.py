import numpy as np
from pathlib import Path

from meshparty import skeletonize
from meshparty import skeleton as mps
from meshparty import trimesh_io
import skeletor as sk
import pymaid
from cloudvolume import CloudVolume
from . import skeleton_manipulations
from . import skeletonization
from . import neuroglancer_utilities
import pandas as pd

## DEFAULTS FOR SKELETONIZATION

skeletonization.contraction_defaults = {'epsilon': 1e-05,'iter_lim': 8, 'precision': 1e-5, 'SL': 2, 'WH0': .05,'WL0':'auto'}



skeletonization.skeletonization_defaults = {'NG_voxel_resolution': np.array([4.3,4.3,45]),
                                              'CATMAID_voxel_resolution': np.array([4.3,4.3,45]),
                                              'merge_size_threshold': 100,
                                              'merge_max_dist':1000,
                                              'merge_distance_step':500,
                                              'soma_radius':12000,
                                              'invalidation_distance':8500,
                                              'merge_collapse_soma':True}

skeletonization.diameter_smoothing_defaults = {'smooth_method':'smooth',
                                               'smooth_bandwidth':1000}



## PRIMARY FUNCTION:

def neuron_from_skeleton(seg_id,
                         vol,
                         transform = True,
                         method='kimimaro',
                         soma_coord = None,
                         node_labels = None,
                         xyz_scaling=1,
                         contraction_params = skeletonization.contraction_defaults,
                         skeltonization_params = skeletonization.skeletonization_defaults):
    
    if isinstance(vol,str):
        cv = set_cv(vol)
    else:
        cv = vol
    
    if 'kimimaro' in method:
        km_skel = get_kimimaro_skeleton(cv,seg_id)
        mp_skel = km_to_mp(km_skel)
        recalculate_radius = False
        mesh = None
    else:
        if soma_coord is None:
            raise ValueError('Skeleton generation requires a soma point for Mesh Party')  
        mesh = get_mesh(cv,seg_id) 
        cmesh = skeletor_contraction(mesh,**contraction_params)
        mp_mesh = trimesh_io.Mesh(cmesh.vertices,cmesh.faces,cmesh.face_normals)
        mp_skel = meshparty_skeletonize(mp_mesh,soma_coord,**skeltonization_params)
        recalculate_radius = True
    
    # Convert to pymaid, add radius, and transform
    ng_voxel_res = skeltonization_params['NG_voxel_resolution']
    cm_voxel_res = skeltonization_params['CATMAID_voxel_resolution']
    
    neuron = mp_to_pymaid(mp_skel,
                          node_labels=node_labels,
                          xyz_scaling=xyz_scaling,
                          recalculate_radius = recalculate_radius,
                          mesh = mesh)
    neuron.nodes[['x','y','z']] = neuron.nodes[['x','y','z']] / ng_voxel_res  * cm_voxel_res
    
        
    neuron = skeleton_manipulations.diameter_smoothing(neuron,smooth_method='smooth', smooth_bandwidth=1000)
    
    if soma_coord is not None:
        neuron = skeleton_manipulations.set_soma(neuron,soma_coord)
    
    #neuron.downsample(10)
   
    if transform is True:
        transformed_neuron = transform_neuron(neuron,voxel_resolution = skeltonization_params['NG_voxel_resolution'])
        return(transformed_neuron)
    else:
        return(neuron)
        
    
    

                                          
## METHODS
                                   
# 1. set cv path:
def set_cv(cv_path):
    if 'graphene' in cv_path:  
        cv = CloudVolume(cv_path,use_https=True,agglomerate=True)
    else:
        cv = CloudVolume(cv_path)
    return(cv)


# 2. Download skeletons and meshes

def get_kimimaro_skeleton(cv,seg_id):
    return(cv.skeleton.get(seg_id))


def get_mesh(cv,seg_id):
    if 'graphene' in cv.meta.info['mesh']:
        mesh = cv.mesh.get(seg_id,use_byte_offsets=True)[seg_id]
    else:
        mesh = cv.mesh.get(seg_id,use_byte_offsets=False)
    return(mesh)


# 3. Skeletonize using skeletor and meshparty 

def skeletor_contraction(mesh,**contraction_params):   
    cmesh = sk.contract(mesh,**contraction_params,progress=False)
    return(cmesh)


def meshparty_skeletonize(mesh,
                   soma_coords,
                   NG_voxel_resolution = np.array([4.3,4.3,45]),
                   CATMAID_voxel_resolution = np.array([4.3,4.3,45]),
                   merge_size_threshold=100,
                   merge_max_dist=1000,
                   merge_distance_step=500,
                   soma_radius=12000,
                   invalidation_distance=8500,
                   merge_collapse_soma=True):
    
    
    # Convert to voxel space
    adjusted_soma = soma_coords * NG_voxel_resolution
    
    # Repair mesh
    try:
        mesh.merge_large_components(size_threshold=merge_size_threshold,
                                    max_dist=merge_max_dist,
                                    dist_step=merge_distance_step)
    except:
        print('mesh heal failed')
    
    # Skeletonize
    skeleton =skeletonize.skeletonize_mesh(mesh,
                                     adjusted_soma,
                                     soma_radius=soma_radius,
                                     invalidation_d=invalidation_distance,
                                     collapse_soma=merge_collapse_soma)
    
                  
    return(skeleton)



# 4. Convert to pymaid

def km_to_mp(skel):
    
    meshparty_skel = mps.Skeleton(skel.vertices,skel.edges,vertex_properties = {'rs':skel.radius})
    if hasattr(skel,'metadata'):
        meshparty_skel.metadata = skel.metadata
    else:
        meshparty_skel.metadata = None
    
    return(meshparty_skel)


def mp_to_pymaid(meshparty_skel,node_labels=None, xyz_scaling=1, recalculate_radius = True, mesh = None):
    '''
    Convert a meshparty skeleton into a dataframe for navis/catmaid import.
    Args
    ----
    skel:        meshparty skeleton
    node_labels: list , list of node labels, default is None and will generate new ones.
    xyz_scaling: int, scale factor for coordinates
    recalculate_radius: bool, If true, will use the skeletor method of radius calculate, and will require a mesh input
    mesh: mesh, required only if recalculating radius
 
    
    '''
    ds = meshparty_skel.distance_to_root
    order_old = np.argsort(ds)
    new_ids = np.arange(len(ds))
    order_map = dict(zip(order_old, new_ids))
    
    
    if node_labels is None:
        node_labels = new_ids    
    else:
        node_labels = np.array(node_labels)[order_old]
        
    xyz = meshparty_skel.vertices[order_old]
    par_ids = np.array([order_map.get(nid, -1)
                        for nid in meshparty_skel._parent_node_array[order_old]])
    
    data = {'node_id': node_labels,
         'parent_id': par_ids,
         'x': xyz[:,0]/xyz_scaling,
         'y': xyz[:,1]/xyz_scaling,
         'z': xyz[:,2]/xyz_scaling}
    df = pd.DataFrame(data=data)
    
    df['label'] = np.ones(len(df))
    if recalculate_radius:
        df['radius']= sk.radii(df,mesh,method = 'ray')
    else:
        df['radius'] = meshparty_skel.vertex_properties['rs']


    neuron = pymaid.from_swc(df)

    return neuron


# 5. Transform neuron
def transform_neuron(neuron,voxel_resolution = np.array([4.3,4.3,45])):
    nodes = neuron.nodes[['x','y','z']] / voxel_resolution
    output = neuroglancer_utilities.fanc4_to_3(nodes.values)
    neuron.nodes[['x','y','z']] = np.array(list(zip(output['x'],output['y'],output['z']))) * voxel_resolution
    return(neuron)
    

    









