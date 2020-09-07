import pymaid
import json
import numpy as np
from pathlib import Path
import requests

def fanc4_to_3(points,scale=2):
    ''' Convert from realigned dataset to original coordinate space.
    Inputs:
             points: an nx3 array of mip0 pixel coordinates
             scale:  selects the granularity of the field being used, but does not change the units.
    
    Returns: a dictionary of transformed x/y/z values and the dx/dy/dz values'''
             
    base = "https://spine.janelia.org/app/transform-service/dataset/fanc_v4_to_v3/s/{}/".format(scale)
    
    points = np.round(points)
    if len(np.shape(points)) > 1:
        full_url = base + '/values_array'
        points_dict = {'x': list(points[:,0]),'y':list(points[:,1]),'z':list(points[:,2])}
        r = requests.post(url, json = points_dict)
    else:
        full_url = base + 'z/{}/'.format(str(int(points[2]))) + 'x/{}/'.format(str(int(points[0]))) + 'y/{}/'.format(str(int(points[1])))
        r = requests.get(full_url)
    
    
    return(r.json())

def diameter_smoothing(neuron,smooth_method='strahler',smooth_bandwidth=1000):
    
    
    ''' This will smooth out the node diameters by either setting every node of a similar strahler order to the mean radius of every node with that     strahler order, or apply a smoothing function by setting the radius of a node to the mean of every node within a given bandwidth.  For the latter case, it will also make sure that the nodes radii being averaged are from the same strahler order. 

        Parameters
        ----------
        neuron :           A pymaid neuron.

        method:            Either 'strahler' or 'smooth'. Default is 'strahler' as it is much faster, and gave good results for motor neurons.  This determines the method of smoothing.  See above for the difference.  

        bandwidth:         If 'smooth' is chosen, this is the distance threshold (in nm) whose radii will be averaged to determine a given nodes radius.
                           Default is 1000nm.

        Returns
        -------
        neuron:            a pymaid neuron'''
    
    gm = pymaid.geodesic_matrix(neuron)
    if 'strahler' in smooth_method:
        if 'strahler_index' not in neuron.nodes:
            pymaid.strahler_index(neuron)
            
        for i in range(max(neuron.nodes.strahler_index)+1):            
            radius = np.mean(neuron.nodes.radius[(neuron.nodes.strahler_index==i) & (neuron.nodes.treenode_id != neuron.soma)])
            neuron.nodes.radius.loc[(neuron.nodes.strahler_index==i) & (neuron.nodes.treenode_id != neuron.soma)] = radius
            print(i,radius)
    elif 'smooth' in smooth_method:
        smooth_r =[]
        for i in range(len(neuron.nodes)):
            smooth_r.append(np.mean(neuron.nodes.radius.loc[(gm.loc[i,:]<smooth_bandwidth) & 
                                                            (neuron.nodes.strahler_index.loc[gm.loc[i,:]<smooth_bandwidth] ==
                                                             neuron.nodes.strahler_index[i])]))
        
        neuron.nodes.radius = smooth_r

    return(neuron)


def downsample_neuron(neuron,downsample_factor=4):
    
    downsampled = pymaid.resample.downsample_neuron(neuron,downsample_factor,inplace=False)
    
    return(downsampled)
    
