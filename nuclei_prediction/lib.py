import numpy as np
import os
import argparse
from pathlib import Path
import json
from fanc import rootID_lookup as IDlook

# function - path


def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def get_cv_path(version=None):
    fname = Path.home() / '.cloudvolume' / 'segmentations.json'
    with open(fname) as f:
        paths = json.load(f)
    
    if version is None:
        return(paths)
    else:
        return(paths[version])

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


def Bbox2cloud(bbox):
    cloud = [bbox.minpt.x, bbox.minpt.y, bbox.minpt.z, bbox.maxpt.x, bbox.maxpt.y, bbox.maxpt.z]
    return cloud
    

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


def segID_to_svID(segID, ID_array, location_array_mip0, cv, reverse=False):
    indices = np.where(ID_array == segID)[0]
    pts = location_array_mip0[indices]
    if reverse == True:
      pts = pts[::-1]
    for j in range(len(pts)):
      ptsj = pts[j]
      svID = IDlook.segIDs_from_pts_service(ptsj, return_roots=False)
      if svID is None:
        svID = [0]
      if (svID[0] > 0) & (segID != 0):
        break

    # else: # reverse == True
    #   for j in reversed(range(len(pts))):
    #     ptsj = pts[j]
    #     svID = IDlook.segIDs_from_pts_service(ptsj, return_roots=False)
    #     if svID is None:
    #       svID = [0]
    #     if (svID[0] > 0) & (segID != 0):
    #       break

    # if reverse == False:
    #   svID = IDlook.segIDs_from_pts_cv(pts=pts, cv=cv, return_roots=False, progress=False)
    #   ptsj = pts[0]
    # else:
    #   pts = pts[::-1]
    #   svID = IDlook.segIDs_from_pts_cv(pts=pts, cv=cv, return_roots=False, progress=False)
    #   for j in range(len(pts)):
    #     ptsj = pts[0]

    return svID[0],ptsj
