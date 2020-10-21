import numpy as np
from pathlib import Path

from meshparty import skeletonize
from meshparty import skeleton as mps
from meshparty import trimesh_io
import skeletor as sk
import kimimaro
import navis
import pymaid
from cloudvolume import CloudVolume
import requests
from . import skeleton_manipulations
from . import skeletonization
from . import neuroglancer_utilities
from itertools import repeat
import pandas as pd

##TODO: fix caches

def get_skeleton(seg_id,
                 cv_path,method='kimimaro',
                 transform=True, 
                 cache_path = None, 
                 annotations = None, 
                 name = None, 
                 output = 'pymaid',
                 merge = False,
                 **kwargs):
    
    ## TODO: Add merge function to other skeletonization formats using pymaid stitch. Right now, it only works with kimimaro and uses    kimimaro.join_close_components
    ''' Downloads a skeleton generated with kimimaro,meshparty, or skeletor based on its segment ID. 
    Parameters
    ----------
    segment_ids :      list or int. Neuroglancer segment ID
    
    cv_path:           str. Path to cloudvolume.
    
    method:            str. 'kimimaro' , 'meshparty' , or 'skeletor'  This will not rerun kimimaro skeletonization, it will simply retrieve them from the segmentation layer specified by the cv_path. 
    
    transform:         Bool. Whether to transform the output skeleton back to FANC3. Default = True
    
    cache_path:        str. Where to save the mesh locally. 
    
    annotations:       list or str. Annotations to add to the mesh itself. This can be useful for keeping track of things. 
    
    name:              list or str. A name for the mesh. Default is the segment id. 
    
    ** kwargs:         These are specific parameters to pass to each skeletonization.  At a minimum, meshparty skeletonization requires a soma    coordinate. Other parameters are all set to usable defaults.  To see the default parameters, skeletonization.meshparty_params or skeletonization.skeletor_params will show them. 
    
    Returns
    -------
    mesh:             kimimaro skeleton object'''
     
    format_kwargs = {}
    if 'xyz_scaling' in kwargs.keys():
        format_kwargs['xyz_scaling']  = kwargs['xyz_scaling']
    if 'radius' in kwargs.keys():
        format_kwargs['radius'] = kwargs['radius']
        
        
    if 'kimimaro' in method: 
        skeletons = get_kimimaro_skeleton(seg_id,cv_path=cv_path,annotations = annotations, name = name, merge=merge, **format_kwargs )
        
       
          
    elif 'meshparty' in method:
        
        if 'soma_coords' not in kwargs.keys():
            raise ValueError('meshparty skeletonization requires a soma coordinate.')
        
        sk_params = {'cache_path': None,
                      'NG_voxel_resolution': np.array([4.3,4.3,45]),
                      'merge_size_threshold': 100,
                      'merge_max_dist':1000,
                      'merge_distance_step':500,
                      'soma_radius':1200,
                      'invalidation_distance':8500,
                      'merge_collapse_soma':True,
                      'format_kwargs':format_kwargs}

        for i in sk_params:
            if i in kwargs.keys():
                sk_params[i] = kwargs[i]
        
        mesh = get_meshparty_mesh(seg_id,cv_path = cv_path, annotations = annotations)
        skeletons = []
        for i in mesh:
            skeletons.append(skeletonization.get_meshparty_skeleton(i,kwargs['soma_coords'],**sk_params))
            
    
    elif 'skeletor' in method:
        sk_params = {'iter_lim':10, 
                     'sk_method': 'vertex_clusters', 
                     'sampling_dist': 1500, 
                     'output': 'swc',
                     'diam_method':'knn', 
                     'knn_n':5, 
                     'diam_aggregate':'mean'}
        for i in sk_params:
            if i in kwargs.keys():
                sk_params[i] = kwargs[i]
                
        mesh = get_cv_mesh(seg_id,cv_path=cv_path,annotations=annotations)
        skeletons = []
        for i in mesh:
            skeletons.append(skeletonization.get_skeletor_skeleton(i,**sk_params))
            
       
        
    nlist = []
    for i in skeletons:
        if transform is True:
            new_xyz = neuroglancer_utilities.fanc4_to_3(np.array(i[['x','y','z']]/np.array([4.3,4.3,45])),2)
            i['x'] = np.array(new_xyz['x']) * 4.3
            i['y'] = np.array(new_xyz['y']) * 4.3
            i['z'] = np.array(new_xyz['z']) * 45
        
        if 'pymaid' in output:
            i['label'] = np.ones(len(i))
            n = pymaid.from_swc(i)
        elif 'navis' in output:
            n = navis.TreeNeuron(i)
        elif 'swc' in output:
            n = i
        else:
            raise ValueError('inappropriate output format.')
            
            
            
        if 'swc' not in output:
            n.meta_data = i.attrs
            
            if 'pymaid' in output:
                n.neuron_name = i.attrs['name']
            else:
                n.name = i.attrs['name']
                
            if annotations is not None:
                if 'https://storage.googleapis.com/zetta_lee_fly_vnc_001_segmentation/vnc1_full_v3align_2/realigned_v1/seg/full_run_v1' in cv_path:
                    annotations.append('FANC4')
                elif 'https://storage.googleapis.com/zetta_lee_fly_vnc_001_segmentation_temp/vnc1_full_v3align_2/37674-69768_41600-134885_430- 4334/seg/v3' in cv_path:
                    annotations.append('FANC3')
            else:
                if 'https://storage.googleapis.com/zetta_lee_fly_vnc_001_segmentation/vnc1_full_v3align_2/realigned_v1/seg/full_run_v1' in cv_path:
                    annotations = ('FANC4')
                elif 'https://storage.googleapis.com/zetta_lee_fly_vnc_001_segmentation_temp/vnc1_full_v3align_2/37674-69768_41600-134885_430-4334/seg/v3' in cv_path:
                    annotations = ('FANC3')
            
            n.annotations = annotations
 
        nlist.append(n)
        
        
    if 'pymaid' in output:
        nl = pymaid.CatmaidNeuronList(nlist)
    elif 'navis' in output:
        nl = navis.NeuronList(nlist)
    elif 'swc' in output:
        nl = nlist
    else:
        raise ValueError('inappropriate output format.')
    
    if len(nl) == 1:
        nl = nl[0]
    
    return(nl)

    






## Kimimaro Skeletonization
def kimimaro_skeletons(segment_ids,
                       cv_path,
                       cache_path=None,
                       annotations=None,
                       name=None,
                       merge = False):
    
    ''' Downloads a skeleton generated with kimimaro based on its segment ID.
    Parameters
    ----------
    segment_ids :      Neuroglancer segment ID
    
    cv_path:           Path to cloudvolume.  
    
    cache_path:        Where to save the mesh locally. 
    
    annotations:       Annotations to add to the mesh itself. This can be useful for keeping track of things. 
    
    name:              A name for the mesh. Default is the segment id. 
    
    Returns
    -------
    mesh:             kimimaro skeleton object'''
    
    if isinstance(segment_ids,list):
        if name is None:
            names = [str(seg) for seg in segment_ids]
        else:
            n = [name + '_' + str(seg) for seg in segment_ids]
            names = n
    else:
        
        segment_ids = [segment_ids]
        
        if name is None:
            names = segment_ids
        else:
            names = name
     
    
        
    segmentation= CloudVolume(cv_path,
                              parallel=True,
                              progress=True, 
                              )

    skels = segmentation.skeleton.get(segment_ids)
    
    if merge is True:
            skels =  kimimaro.join_close_components(skels)
            
  
    
    if isinstance(skels,list) is not True:
        skels = [skels]
        
    for i in range(len(skels)):
        
        skels[i] = kimimaro.join_close_components(skels[i])
        
        skels[i].metadata = {}
        skels[i].metadata['mesh_type'] = 'cv_mesh'
        skels[i].metadata['skeleton_type'] = 'kimimaro'
        if merge is True:
            skels[i].metadata['segment_ids'] = segment_ids
        else:
            skels[i].metadata['segment_ids'] = segment_ids[i]
        skels[i].metadata['annotations'] = annotations
        if isinstance(names,list):
            skels[i].metadata['name']= names[i]
        else:
            skels[i].metadata['name'] = names

    
    return(skels)


def get_kimimaro_skeleton(segment_ids,
                          cv_path,
                          cache_path=None,
                          annotations=None,
                          name=None,
                          merge = False,
                          **format_kwargs):

    skels = kimimaro_skeletons(segment_ids,cv_path,cache_path=cache_path,annotations=annotations,name=name,merge=merge)
    

    
    if 'xyz_scaling' in format_kwargs.keys():
        xyz_scaling = format_kwargs['xyz_scaling']
    else:
        xyz_scaling = 1
  
    if 'radius' in format_kwargs.keys():
        radius = format_kwargs['radius']
    else:
        radius = True
    
    
    swc_list = []
    for i in skels:
        swc_list.append(km_to_swc(i,radius=radius,xyz_scaling=xyz_scaling))
    
    
    return(swc_list)

 



def km_to_swc(skel,radius=True,xyz_scaling=1):
    
    meshparty_skel = mps.Skeleton(skel.vertices,skel.edges,vertex_properties = {'rs':skel.radius})
    if hasattr(skel,'metadata'):
        meshparty_skel.metadata = skel.metadata
    else:
        meshparty_skel.metadata = None

    swc = mp_to_swc(meshparty_skel,radius=True,xyz_scaling=1,node_labels=None)
    
    return(swc)




## Meshparty Skeletonization


def get_meshparty_mesh(segment_ids,
                       cv_path,
                       cache_path=None,
                       annotations=None,
                       name = None):
    ''' Downloads a mesh based on its segment ID.
    Parameters
    ----------
    segment_ids :      Neuroglancer segment ID
    
    cv_path:           Path to cloudvolume.  
    
    cache_path:        Where to save the mesh locally. 
    
    annotations:       Annotations to add to the mesh itself. This can be useful for keeping track of things. 
    
    name:              A name for the mesh. Default is the segment id. 
    
    Returns
    -------
    mesh:            a meshparty mesh object'''
    
    
    if isinstance(segment_ids,list):
        if name is None:
            names = [str(seg) for seg in segment_ids]
        else:
            n = [name + '_' + str(seg) for seg in segment_ids]
            names = n
    else:
        names = [str(segment_ids)]
        segment_ids = [segment_ids]
        
            
    
    meshes = []
    for i in range(len(segment_ids)):
        mesh_meta = trimesh_io.MeshMeta(cv_path=cv_path, cache_size = 0, disk_cache_path=cache_path, map_gs_to_https=True)
        mesh = mesh_meta.mesh(seg_id=segment_ids[i])
        mesh.metadata['mesh_type'] = 'meshparty'
        mesh.metadata['segment_ids'] = segment_ids[i]
        mesh.metadata['annotations'] = annotations
        mesh.metadata['name']= names[i]
        meshes.append(mesh)

    return(meshes)



def meshparty_skeletonize(mesh,
                   soma_coords,
                   cache_path = None,
                   NG_voxel_resolution = np.array([4.3,4.3,45]),
                   merge_size_threshold=100,
                   merge_max_dist=1000,
                   merge_distance_step=500,
                   soma_radius=24000,
                   invalidation_distance=8500,
                   merge_collapse_soma=True,
                   **format_kwargs):
    
    ''' Runs skeletonization using meshparty algorithm. Only works when large components are merged first. 
    
    Parameters
    ----------
    mesh :                       Meshparty mesh
    
    soma_coords:                 np.array. Soma coords in mip0 pixels
    
    cache_path:                  str. Where to save the mesh locally.
    
    NG_voxel_resolution:         np.array. Segmentation resolution. 
    
    merge_size_threshold:        int. Passed to merge_large_components; see meshparty documentation. 
    
    merge_max_dist:              int. Passed to merge_large_components; see meshparty documentation.
    
    soma_radius:                 int. Approximate size of soma, 
    
    invalidation_distance:       int. Passed to meshparty.skeletonize_mesh; see meshparty documentation.  
    
    merge_collapse_soma:         Default = True.  Just leave it at True. 
    
    Returns
    -------
    sk:                         A skeleton object.  Corresponding .swc file is exported.'''
    
    
    
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
        
    if mesh.metadata['name'] is not None:
        fname = mesh.metadata['name']
        skeleton.metadata = mesh.metadata
        
    else:
        fname = str(mesh.metadata['segment_ids'])
        
    skeleton.metadata['skeleton_type'] = 'meshparty'
    skeleton.metadata['skeletonization_params'] = locals()
    

             
    return(skeleton)


def get_meshparty_skeleton(mesh,
                   soma_coords,
                   cache_path = None,
                   NG_voxel_resolution = np.array([4.3,4.3,45]),
                   merge_size_threshold=100,
                   merge_max_dist=1000,
                   merge_distance_step=500,
                   soma_radius=24000,
                   invalidation_distance=8500,
                   merge_collapse_soma=True,
                   **format_kwargs):

    
    skeleton = meshparty_skeletonize(mesh,
                   soma_coords,
                   cache_path = cache_path,
                   NG_voxel_resolution = NG_voxel_resolution,
                   merge_size_threshold=merge_size_threshold,
                   merge_max_dist=merge_max_dist,
                   merge_distance_step=merge_distance_step,
                   soma_radius=soma_radius,
                   invalidation_distance=invalidation_distance,
                   merge_collapse_soma=merge_collapse_soma)
    
    
    if 'xyz_scaling' in format_kwargs.keys():
        xyz_scaling = format_kwargs['xyz_scaling']
    else:
        xyz_scaling = 1
  
    if 'radius' in format_kwargs.keys():
        radius = format_kwargs['radius']
    else:
        radius = True
    
    
    swc = mp_to_swc(skeleton,
                    radius=radius,
                    xyz_scaling=xyz_scaling,
                    node_labels=None)

    return(swc)



def mp_to_swc(skeleton,
              radius=True, 
              xyz_scaling=1,
              node_labels=None):
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
    
    if radius is True:
        radius = skeleton.vertex_properties['rs']
    else:
        radius = np.ones(len(skeleton.vertex_properties['rs']),1)
    
    if node_labels is None:
        node_labels = new_ids    
    else:
        node_labels = np.array(node_labels)[order_old]
        
    xyz = skeleton.vertices[order_old]
    radius = radius[order_old]
    par_ids = np.array([order_map.get(nid, -1)
                        for nid in skeleton._parent_node_array[order_old]])
    
    data = {'node_id': node_labels,
         'parent_id': par_ids,
         'x': xyz[:,0]/xyz_scaling,
         'y': xyz[:,1]/xyz_scaling,
         'z': xyz[:,2]/xyz_scaling,
         'radius': radius/xyz_scaling}
    df = pd.DataFrame(data=data)
    
    if skeleton.metadata is not None:
        df.attrs = skeleton.metadata


    return df






## Skeletor Skeletonization

def get_cv_mesh(segment_ids,
                cv_path,
                cache_path=None,
                annotations=None,
                name=None):
    
    ''' Downloads a mesh based on its segment ID.
    Parameters
    ----------
    segment_ids :      Neuroglancer segment ID
    
    cv_path:           Path to cloudvolume.  
    
    cache_path:        Where to save the mesh locally. 
    
    annotations:       Annotations to add to the mesh itself. This can be useful for keeping track of things. 
    
    name:              A name for the mesh. Default is the segment id. 
    
    Returns
    -------
    mesh:            a cloudvolume mesh object'''
    
    if isinstance(segment_ids,list):
        if name is None:
            names = [str(seg) for seg in segment_ids]
        else:
            n = [name + '_' + str(seg) for seg in segment_ids]
            names = n
    else:
        segment_ids = [segment_ids]
        names = [str(segment_ids)]
    
        
    segmentation= CloudVolume(cv_path,
                              parallel=True,
                              progress=True, 
                              )

    meshes = []
    for i in range(len(segment_ids)):
        mesh = segmentation.mesh.get(segment_ids[i])
        mesh.metadata = {}
        mesh.metadata['mesh_type'] = 'cv_mesh'
        mesh.metadata['segment_ids'] = segment_ids[i]
        mesh.metadata['annotations'] = annotations
        mesh.metadata['name']= names[i]
        meshes.append(mesh)
    
    return(meshes)



def get_skeletor_skeleton(mesh, 
                         iter_lim=10, 
                         sk_method='vertex_clusters', 
                         sampling_dist = 1500, 
                         output='swc',
                         diam_method='knn', 
                         knn_n=5, 
                         diam_aggregate='mean',
                         ):

    skel_metadata = mesh.metadata
    skel_metadata['skeletonization_params'] = locals()
    skel_metadata['skeleton_type'] = 'skeletor'
    
    contraction = sk.contract(mesh, iter_lim=iter_lim,progress=False)
    
    swc = sk.skeletonize(contraction, method=sk_method, sampling_dist = sampling_dist, output=output,progress=False)
    
    swc['radius'] = sk.radii(swc, mesh, method=diam_method, n=knn_n, aggregate=diam_aggregate)
    

    swc.attrs = skel_metadata

    return(swc)


## Default parameters for reference:

skeletonization.meshparty_defaults = {'cache_path': None,
              'NG_voxel_resolution': np.array([4.3,4.3,45]),
              'merge_size_threshold': 100,
              'merge_max_dist':1000,
              'merge_distance_step':500,
              'soma_radius':1200,
              'invalidation_distance':8500,
              'merge_collapse_soma':True,
              'xyz_scaling':1,
              'radius': True}

skeletonization.skeletor_defaults = {'iter_lim':10, 
                     'sk_method': 'vertex_clusters', 
                     'sampling_dist': 1500, 
                     'output': 'swc',
                     'diam_method':'knn', 
                     'knn_n':5, 
                     'diam_aggregate':'mean'}