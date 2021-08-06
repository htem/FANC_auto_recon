import numpy as np
import sys
import os
import pandas as pd
from numpy.random.mtrand import f
from tqdm import tqdm
from glob import glob
import argparse

from cloudvolume import CloudVolume, view, Bbox
import cc3d
from taskqueue import TaskQueue, queueable, LocalTaskQueue
from functools import partial
sys.path.append(os.path.abspath("../segmentation"))
# to import rootID_lookup and authentication_utils like below

import rootID_lookup as IDlook
import authentication_utils as auth


def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

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


def mip2_to_mip0(x,y,z, img):
  origin = img.bounds.minpt
  xyz_mip2 = np.add(np.array([x,y,z]), origin)
  xyz_mip0 = np.array([(xyz_mip2[0] * 2**2),(xyz_mip2[1] * 2**2), xyz_mip2[2]])
  xyz_mip0 = xyz_mip0.astype('int64')

  return xyz_mip0[0], xyz_mip0[1], xyz_mip0[2]
