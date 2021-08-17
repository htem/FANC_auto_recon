import numpy as np
import sys
import os
import pandas as pd
from numpy.random.mtrand import f
from tqdm import tqdm
from glob import glob
import argparse
import random
import sqlite3

from cloudvolume import CloudVolume, view, Bbox
sys.path.append(os.path.abspath("../segmentation"))
# to import rootID_lookup and authentication_utils like below

import rootID_lookup as IDlook
import authentication_utils as auth

# function - path


def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f


# function - cloud-volume


def mip0_to_mip2(x,y,z):
  xyz_mip2 = np.array([(x/(2**2)),(y/(2**2)), z])
  xyz_mip2 = xyz_mip2.astype('int64')

  return xyz_mip2[0], xyz_mip2[1], xyz_mip2[2]


def mip0_to_mip2_array(array):
  X, Y, Z = mip0_to_mip2(array[0], array[1], array[2])
  result = np.array([X, Y, Z])
  return result


def mip0_to_mip4(x,y,z):
  xyz_mip4 = np.array([(x/(2**4)),(y/(2**4)), z])
  xyz_mip4 = xyz_mip4.astype('int64')

  return xyz_mip4[0], xyz_mip4[1], xyz_mip4[2]


def mip0_to_mip4_array(array):
  X, Y, Z = mip0_to_mip4(array[0], array[1], array[2])
  result = np.array([X, Y, Z])
  return result


def mip2_to_mip0(x,y,z, img):
  origin = img.bounds.minpt
  xyz_mip2 = np.add(np.array([x,y,z]), origin)
  xyz_mip0 = np.array([(xyz_mip2[0] * 2**2),(xyz_mip2[1] * 2**2), xyz_mip2[2]])
  xyz_mip0 = xyz_mip0.astype('int64')

  return xyz_mip0[0], xyz_mip0[1], xyz_mip0[2]


def mip2_to_mip0_array(array, img):
  X, Y, Z = mip2_to_mip0(array[0], array[1], array[2], img)
  result = np.array([X, Y, Z])
  return result


def mip4_to_mip0(x,y,z, img):
    origin = img.bounds.minpt
    xyz_mip4 = np.add(np.array([x,y,z]), origin)
    xyz_mip0 = np.array([(xyz_mip4[0] * 2**4),(xyz_mip4[1] * 2**4), xyz_mip4[2]])
    xyz_mip0 = xyz_mip0.astype('int64')

    return xyz_mip0[0], xyz_mip0[1], xyz_mip0[2]


def mip4_to_mip0_array(array, img):
    X, Y, Z = mip4_to_mip0(array[0], array[1], array[2], img)
    result = np.array([X, Y, Z])
    return result


# function - seg/svID


def find_most_frequent_ID(array):
    uniqueID, count = np.unique(array, return_counts=True)
    unsorted_max_indices = np.argsort(-count)
    topIDs1 = uniqueID[unsorted_max_indices] 
    topIDs2 = topIDs1[~(topIDs1 == 0)] # no zero
    if topIDs2.size == 0:
        topID = np.zeros(1, dtype = 'int64') # empty then zero
    else:
        topID = topIDs2.astype('int64')[0]

    return topID


def segID_to_svID(segID, ID_array, location_array_mip0, reverse=False):
    indices = np.where(ID_array == segID)[0]
    pts = location_array_mip0[indices]
    if reverse == False:
      for j in range(len(pts)):
          ptsj = pts[j]
          svID = IDlook.segIDs_from_pts_service(ptsj, return_roots=False)
          if svID is None:
            svID = [0]
          if (svID[0] > 0) & (segID != 0):
              break
    else: # reverse == True
      for j in reversed(range(len(pts))):
          ptsj = pts[j]
          svID = IDlook.segIDs_from_pts_service(ptsj, return_roots=False)
          if svID is None:
            svID = [0]
          if (svID[0] > 0) & (segID != 0):
              break

    return svID[0],ptsj

    
def update_soma_table(dir, input_table_name, output_table_name, cv=None, retry=True, max_tries=10, chunksize = 20000):
  if cv is None:
    cv = auth.get_cv()

  temp = dir + '/' + str(random.randint(111111,999999)) + '.csv'
  header = True
  idx = 0
  for chunk in pd.read_csv(dir + '/{}.csv'.format(input_table_name), chunksize=chunksize): 
    try:
      chunk.loc[:,'nuc_rootID'] = cv.get_roots(chunk.nuc_svID.values)
      chunk.loc[:,'body_rootID'] = cv.get_roots(chunk.body_svID.values)
      chunk.to_csv(temp, mode='a', index=False, header=header)
      
    except Exception as e:
      print(e)
      if retry is True:
        tries = 0
        while tries < max_tries:
          try:
            chunk.loc[:,'nuc_rootID'] = cv.get_roots(chunk.nuc_svID.values)
            chunk.loc[:,'body_rootID'] = cv.get_roots(chunk.body_svID.values)
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
      
  os.replace(temp, dir + '/' + output_table_name)
  return 'Complete',None 