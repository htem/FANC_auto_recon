import skeletor as sk
from cloudvolume import CloudVolume
import navis
import pymaid
from meshparty import trimesh_vtk as tv
from matplotlib import pyplot as plt
from meshparty import skeletonize
import numpy as np
from . import skeletonization
from . import single_skeletonization
import pandas as pd
import meshparty 

single_skeletonization.contraction_defaults = {'epsilon': 1e-05,'iter_lim': 8, 'precision': 1e-5, 'SL': 2, 'WH0': .05}



single_skeletonization.skeletonization_defaults = {'cache_path': None,
                                              'NG_voxel_resolution': np.array([4,4,40]),
                                              'CATMAID_voxel_resolution': np.array([4.3,4.3,45]),
                                              'merge_size_threshold': 100,
                                              'merge_max_dist':1000,
                                              'merge_distance_step':500,
                                              'soma_radius':1200,
                                              'invalidation_distance':8500,
                                              'merge_collapse_soma':True}



def single_skeleton(seg_id,
                    cv_url,
                    soma_coords,
                    contraction_params = single_skeletonization.contraction_defaults,
                    skeltonization_params = single_skeletonization.skeletonization_defaults):
    # Get mesh
    mesh = get_mesh(seg_id,cv_url)
    
    # Contract mesh
    cmesh = sk.contract(mesh,
                    **contraction_params,
                    WL0='auto',
                    progress=False)
    # Skeletonize
    
    mp_mesh = meshparty.trimesh_io.Mesh(cmesh.vertices,cmesh.faces,cmesh.face_normals)
    skeleton = meshparty_skeletonize(mp_mesh,soma_coords,**skeltonization_params)
    
    # Convert to pymaid and add radius 
    pymaid_neuron = mp_to_pymaid(skeleton,mesh,node_labels=None,xyz_scaling=1)
    pymaid_neuron.nodes[['x','y','z']] = (pymaid_neuron.nodes[['x','y','z']] / single_skeletonization.skeletonization_defaults['NG_voxel_resolution']) * single_skeletonization.skeletonization_defaults['CATMAID_voxel_resolution']
    return(pymaid_neuron)
    
    
    
    
    
    


def get_mesh(seg_id,cv_url):
    vol = CloudVolume(cv_url)
    return(vol.mesh.get(seg_id))





def meshparty_skeletonize(mesh,
                   soma_coords,
                   cache_path = None,
                   NG_voxel_resolution = np.array([4.3,4.3,45]),
                   CATMAID_voxel_resolution = np.array([4.3,4.3,45]),
                   merge_size_threshold=100,
                   merge_max_dist=1000,
                   merge_distance_step=500,
                   soma_radius=24000,
                   invalidation_distance=8500,
                   merge_collapse_soma=True):
    
    
    # Convert to voxel space
    adjusted_soma = soma_coords * NG_voxel_resolution
    
    # Repair mesh
    mesh.merge_large_components(size_threshold=merge_size_threshold,
                                max_dist=merge_max_dist,
                                dist_step=merge_distance_step)
    
    # Skeletonize
    skeleton =skeletonize.skeletonize_mesh(mesh,
                                     adjusted_soma,
                                     soma_radius=soma_radius,
                                     invalidation_d=invalidation_distance,
                                     collapse_soma=merge_collapse_soma)
    
                  
    return(skeleton)







def mp_to_pymaid(skeleton,mesh,node_labels=None, xyz_scaling=1):
    '''
    Convert a meshparty skeleton into a dataframe for navis/catmaid import.
    Parameters
    ----------
    skel:     meshparty skeleton
    radius:   
    
    '''
    ds = skeleton.distance_to_root
    order_old = np.argsort(ds)
    new_ids = np.arange(len(ds))
    order_map = dict(zip(order_old, new_ids))
    
    
    if node_labels is None:
        node_labels = new_ids    
    else:
        node_labels = np.array(node_labels)[order_old]
        
    xyz = skeleton.vertices[order_old]
    par_ids = np.array([order_map.get(nid, -1)
                        for nid in skeleton._parent_node_array[order_old]])
    
    data = {'node_id': node_labels,
         'parent_id': par_ids,
         'x': xyz[:,0]/xyz_scaling,
         'y': xyz[:,1]/xyz_scaling,
         'z': xyz[:,2]/xyz_scaling}
    df = pd.DataFrame(data=data)
    
    df['radius']= sk.radii(df,mesh,method = 'ray')
    df['label'] = np.ones(len(df))

    neuron = pymaid.from_swc(df)

    return neuron



