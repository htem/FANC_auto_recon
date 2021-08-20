#!/usr/bin/env python3

import json
from secrets import token_hex
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
import sys
import sqlite3
import csv
sys.path.append(os.path.abspath("/home/skuroda/FANC_auto_recon/segmentation"))
import rootID_lookup
import authentication_utils


def flip_xyz_zyx_convention(array, inplace=True):
    """
    Given an Nx6 array, swap values in columns 1 and 3 and swap values in
    columns 4 and 6. This converts xyz-ordered point pairs to be zyx-ordered,
    or zyx-order point pairs to be xyz-ordered.
    """
    if not inplace:
        array = np.copy(array)
    assert array.shape[1] == 6
    array[:, 0:3] = array[:, 2::-1]
    array[:, 3:6] = array[:, 5:2:-1]
    if not inplace:
        return array


def flip_pre_post_order(array, inplace=True):
    """
    Given an Nx6 array, swap columns 1-3 with columns 4-6.
    """
    if not inplace:
        array = np.copy(array)
    assert array.shape[1] == 6
    tmp = array[:, 0:3].copy()
    array[:, 0:3] = array[:, 3:6]
    array[:, 3:6] = tmp
    if not inplace:
        return array


def upscale(array, scale_factor, inplace=True):
    """
    Given an Nx6 array and a scaling factor (constant or 3-length), multiply the
    first 3 columns and the last 3 columns of the array by the scale factor.
    """
    if not inplace:
        array = np.copy(array)
    array[:, 0:3] = array[:, 0:3] * scale_factor
    array[:, 3:6] = array[:, 3:6] * scale_factor
    if not inplace:
        return array


def downscale(array, scale_factor, inplace=True):
    """
    Given an Nx6 array and a scaling factor (constant or 3-length), divide the
    first 3 columns and the last 3 columns of the array by the scale factor.
    """
    if not inplace:
        array = np.copy(array)
    array[:, 0:3] = array[:, 0:3] / scale_factor
    array[:, 3:6] = array[:, 3:6] / scale_factor
    if not inplace:
        return array


def load(fn, convention='xyz', units='voxels', voxel_size=None, verbose=False, threshold = 12):
    """
    Given a filename of a file containing synaptic links, load the links and
    return them as an Nx6 numpy array representing the N links.  The first 3
    columns represent presynaptic coordinates, last 3 columns represent
    postsynaptic coordinates.

    Supports .npy, .csv, and binary files, though the code makes some
    assumptions about the units and column orderings of each format that you
    should verify are correct for your files.

    fn: filename
    convention: 'xyz' (default) or 'zyx'
        Determines ordering of the 3 columns representing each point.
    units: 'voxels' (default) or 'nm'/'nanometers'
        Determines units of the returned points.
    voxel_size: None (default) or 3-tuple (e.g. (4, 4, 40))
        Determines voxel size to use for conversions. If left as None, the code
        knows what default voxel size to use for different file formats.
    
    threshold: int, threshold to apply based on "sum"
    """
    assert convention in ['xyz', 'zyx']
    assert units in ['voxels', 'nm', 'nanometers']

    if fn.endswith('.npy'):
        if verbose: print('Mode 1: npy')
        # For opening .npy files saved from np.save
        links = np.load(fn)

        # The .npy files Jasper generated on Feb 8 were saved in zyx, so flip them to xyz
        if True:  # Update this if convention changes
            flip_xyz_zyx_convention(links)
        # The .npy files Jasper generated on Feb 8 were saved in post-pre order
        if True:  # Update this if convention changes
            flip_pre_post_order(links)

        if voxel_size is None:
            # The .npy files Jasper generated on Feb 8 are saved in nm, so
            # convert to units of voxels at (4, 4, 40) nm voxel size for easier
            # entering into ng.
            voxel_size = (4, 4, 40)
        if units == 'voxels':
            downscale(links, voxel_size)

        # If the default kwargs were used, links is now pre-post, xyz, in units
        # of voxels at (4, 4, 40)nm

    elif fn.endswith('.csv'):
        if verbose: print('Mode 2: csv')
        # For opening ground truth annotation files
        links = np.genfromtxt(fn, delimiter=',', skip_header=1, dtype=np.uint16)

        # Ground truth annotations were saved in zyx, so flip them to xyz
        if True:  # Update this if convention changes
            flip_xyz_zyx_convention(links)
        # Ground truth annotations were saved in pre-post order, so OK as is
        if False:  # Update this if convention changes
            flip_pre_post_order(links)

        if voxel_size is None:
            # Ground truth annotations were saved in nm, so convert to units of
            # voxels at (4, 4, 40) nm voxel size for easier entering into ng.
            voxel_size = (4, 4, 40)
        if units == 'voxels':
            downscale(links, voxel_size)

        # If the default kwargs were used, links is now pre-post, xyz, in units
        # of voxels at (4, 4, 40)nm

    else:
        if verbose: print('Mode 3: binary')
        # For opening binary files saved by ../detection/worker.py
        # post coord(x,y,z), pre coord(x,y,z), mean, max, area, 4x4x4 moments
        data = np.fromfile(fn, dtype=np.dtype("6f8,3f8,(4,4,4)f8"))
        
        # Apply threshold based on "sum" and return links that pass.
        try:
            links = np.stack([x[0].astype("int32") for x in data if x[2][0][0][0] > threshold])
        except:
            return np.array([])

        if True:  # The Feb 7 predictions were saved in post-pre order
            flip_pre_post_order(links)

        if units == 'voxels':
            # The Feb 7 predictions are in units of mip1 voxels ((8.6, 8.6, 45)
            # nm) so convert to mip0 voxels for easier entering into ng.
            upscale(links, (2, 2, 1))
            # To indicate the location in the middle of the mip1 voxel, add 1
            # after the upscaling (since integers indicate top-left corners).
            links = links + np.array([1, 1, 0, 1, 1, 0])
        else:
            if voxel_size is None:
                voxel_size = (8.6, 8.6, 45)
            upscale(links, voxel_size)

        # If the default kwargs were used, links is now pre-post, xyz, in units
        # of voxels at (4.3, 4.3, 45)nm

    if convention == 'zyx':
        flip_xyz_zyx_convention(links)

    return links


def to_ng_annotations(links, input_order='xyz', input_units=(1, 1, 1),
                      voxel_mip_center=None):
    """
    Create a json representation of a set of synaptic links, appropriate for
    pasting into a neuroglancer annotation layer.
    links: Nx6 numpy array representing N pre-post point pairs.
    input_order: 'xyz' (default) or 'zyx'
        Indicate which column order the input array has.
    input_units: (1, 1, 1) (default) or some other 3-tuple
        If your links are in nm, indicate the voxel size in nm. e.g. (4, 4,
        40) or (40, 4, 4) depending on the input order. If your links are
        already in units of voxels, leave this at the default value.
    voxel_mip_center: None or int
        In neuroglancer, an annotation with an integer coordinate value appears
        at the top-left corner of a voxel, not at the center of that voxel.
        Point annotations often make more sense being placed in the middle of
        the voxel. If False, nothing is added and the neuroglancer default of
        integer values pointing to voxel corners is kept. If voxel_mip_center
        is set to 0, 0.5 will be added to each coordinate so that
        integer-valued inputs end up pointing to the middle of the mip0 voxel.
        If set to 1, 1 will be added to point to the middle of the mip1 voxel.
        If set to x, 0.5 * 2^x will be added to point to the middle of the mipx
        voxel.
        The z coordinate is not changed no matter what, since mips only
        downsample x and y.
    """
    assert input_order in ['xyz', 'zyx']

    def line_anno(pre, post):
        return {
            'pointA': [x for x in pre],
            #'pointA': [int(x) for x in pre],
            'pointB': [x for x in post],
            #'pointB': [int(x) for x in post],
            'type': 'line',
            'id': token_hex(40)
        }

    if isinstance(links, str):
        links = load(links)

    if input_units is not (1, 1, 1):
        links = downscale(links.astype(float), input_units, inplace=False)
        # Now links are in units of voxels

    if input_order == 'zyx':
        links = flip_xyz_zyx_convention(links, inplace=False)

    if voxel_mip_center is not None:
        delta = 0.5 * 2**voxel_mip_center
        adjustment = (delta, delta, 0, delta, delta, 0)
        links = links.astype(float) + adjustment

    annotations = [line_anno(links[i, 0:3],
                             links[i, 3:6])
                   for i in range(links.shape[0])]
    print(json.dumps(annotations, indent=2))

    try:
        import pyperclip
        answer = input("Want to copy the output to the clipboard? (Only works if "
                       "you're running this script on a local machine, not on a "
                       "server.) [y/n] ")
        if answer.lower() == 'y':
            print('Copying')
            pyperclip.copy(json.dumps(annotations))
    except:
        print("Install pyperclip (pip install pyperclip) for the option to"
              " programmatically copy the output above to the clipboard")

        
        
## Methods for updating synapses

def update_synapse_tables(csv_path=None, db_path=None, cv=None):
    
    if cv is None:
        cv = authentication_utils.get_cv()
    
    if csv_path is not None:
        update_synapse_csv(csv_path,cv)
    
    if db_path is not None and csv_path is not None:
        update_synapse_db(db_path,csv_path)

def update_synapse_db(synapse_db,synapse_csv_fname):
    
    if isinstance(synapse_db,str):
        synapse_db = Path(synapse_db)
    elif isinstance(synapse_db, Path):
        synapse_db = synapse_db  
    else:
        raise Exception('Wrong path format. Use string or pathlib.Path')
        
    temp_file = synapse_db.parent / '{}.db'.format(random.randint(111111,999999)) 
    
    con = sqlite3.connect(str(temp_file))
    cur = con.cursor()

    # Create table
    cur.execute('''CREATE TABLE synapses
                   (source text, post_pt text, pre_SV INTEGER, post_SV INTEGER, pre_pt text, pre_root INTEGER, post_root INTEGER)''')

    # Insert a row of data

    with open(synapse_csv_fname, 'r') as fin: # `with` statement available in 2.5+
        # csv.DictReader uses first line in file for column headings by default
        dr = csv.DictReader(fin) # comma is default delimiter
        to_db = [(i['source'], i['post_pt'], i['pre_SV'], i['post_SV'], i['pre_pt'], i['pre_root'], i['post_root']) for i in dr]

    cur.executemany("INSERT INTO synapses (source, post_pt, pre_SV, post_SV, pre_pt, pre_root, post_root) VALUES (?, ?, ?, ?, ?, ?, ?);", to_db)
    con.commit()
    con.close()
    
    os.replace(temp_file,synapse_db)


    
def update_synapse_csv(synapse_csv_fname, cv, retry=True, max_tries=10, chunksize = 100000):
    ''' Update roots of a synapse table.
    
    args:
    synapse_csv_fname: str, path to csv file containing synapses.
    cv:         CloudVolume object
    retry:      bool, If errors occur duriong lookup, retry failed chunk. Default = True
    max_tries:  int, number of tries for a given chunk before failing
    chunksize:  int, number of rows to read from csv file at once. Default is 10000
    
    returns:
    first a temp csv file is generated, and if no failures occur, a replaced csv file with updated root IDs will be generated.'''
    
    if isinstance(synapse_csv_fname,str):
        synapse_csv_fname = Path(synapse_csv_fname)
    elif isinstance(synapse_csv_fname, Path):
        synapse_csv_fname = synapse_csv_fname  
    else:
        raise Exception('Wrong path format. Use string or pathlib.Path')
        
    temp = synapse_csv_fname.parent / '{}.db'.format(random.randint(111111,999999)) 

    header = True
    idx = 0
    for chunk in pd.read_csv(synapse_csv_fname, chunksize=chunksize):    
        try:
            chunk.loc[:,'pre_root'] = rootID_lookup.get_roots(chunk.pre_SV.values,cv)
            chunk.loc[:,'post_root'] = rootID_lookup.get_roots(chunk.post_SV.values,cv)
            chunk.to_csv(temp, mode='a',index=False,header=header)
            
        except Exception as e:
            print(e)
            if retry is True:
                tries = 0
                while tries < max_tries:
                    try:
                        chunk.pre_root = rootID_lookup.get_roots(chunk.pre_SV.values, cv)
                        chunk.post_root = rootID_lookup.get_roots(chunk.post_SV.values, cv)
                        chunk.to_csv(temp, mode='a', index=False, header=False)
                        tries = max_tries+1
                    except Exception as e2:
                        print(e2)
                        tries+=1
                        print('Fail at:',chunksize*idx,' Try:',tries)
                    if tries == max_tries:
                        return 'Incomplete',idx*chunksize
            else:      
                return 'Incomplete',idx 
        idx+=1
        
        header = False
        
    os.replace(temp,synapse_csv_fname)
    return 'Complete',None 


def init_table(filename):
        fileEmpty =  os.path.exists(filename)
        if not fileEmpty:
            df = pd.DataFrame(data = None, columns={'pre_SV','post_SV','pre_pt','post_pt','source','pre_root','post_root'})
            df.to_csv(filename,index=False)
            print('table created')
        else:
            print('table exists')


def write_table(table_name,
                source_name,
                threshold=12,
                include_source=True,
                drop_duplicates=True):
    
    ''' Write a synapse csv file from synaptic links stored locally
    args:
    
    table_name: str, path to table
    source_name: str, path to folder containing synapse files
    threshold: int, thresholding value to use when selecting synapses. Thresholding is based on "sum" value. This currently only works with link files that have the sum score as the 10th value in the array. 
    include_source: bool, include the filename of the synaptic link. Can be useful for tracking issues, but increases file size substantially. 
    drop_duplicates: bool, drop duplicate synapses between the same pair of supervoxels. 
    '''
    
    if not isinstance(source_name,str):
        source_name = str(source_name)
    
    links_formatted = load(source_name,threshold=threshold).reshape([-1,3])
    
    if links_formatted is not None:
        sv_ids = rootID_lookup.segIDs_from_pts_service(links_formatted,return_roots=False)
        if isinstance(sv_ids,list):
            pre_ids = sv_ids[0::2]
            post_ids = sv_ids[1::2]
            cols = {'pre_SV','post_SV','pre_pt','post_pt','source','pre_root','post_root'}
            df = pd.DataFrame(columns=cols)

            df.pre_SV = pre_ids
            df.post_SV = post_ids
            df.pre_pt = list(links_formatted[0::2])
            df.post_pt = list(links_formatted[1::2])
            if include_source is True:
                df.source = source_name
            # Remove 0 value SV ids
            df =df[(df.pre_SV != 0) & (df.post_SV !=0)]
            
            if drop_duplicates is True:    
                df.drop_duplicates(subset=['pre_SV', 'post_SV'], inplace=True)
                

            df.to_csv(table_name, mode='a', header=False,index=False, encoding = 'utf-8',columns=cols)
            return True
        else:
            return True
            
    return False