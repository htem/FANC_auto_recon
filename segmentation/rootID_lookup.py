import pandas as pd
import numpy as np
from pathlib import Path
import json
from cloudvolume import CloudVolume
import cloudvolume
import collections
import numpy as np
import os
import requests
import tqdm
from concurrent import futures


class GSPointLoader(object):
    """Build up a list of points, then load them batched by chunk.

    This code is based on an implementation by
    `Peter Li<https://gist.github.com/chinasaur/5429ef3e0a60aa7a1c38801b0cbfe9bb>_.
    """

    def __init__(self, cloud_volume):
        """Initialize with zero points.

        See add_points to queue some.

        Parameters
        ----------
        cloud_volume :  cloudvolume.CloudVolume (SET AGGLOMERATE = FALSE for the cloudvolume object.)

        """
        CVtype = cloudvolume.frontends.precomputed.CloudVolumePrecomputed
        if not isinstance(cloud_volume, CVtype):
            raise TypeError('Expected CloudVolume, got "{}"'.format(type(cloud_volume)))

        self._volume = cloud_volume
        self._image_res = np.array([4.3,4.3,45])
        self._chunk_map = collections.defaultdict(set)
        self._points = None

    def add_points(self, points):
        """Add more points to be loaded.

        Parameters
        ----------
        points:     iterable of XYZ iterables
                    E.g. Nx3 ndarray.  Assumed to be in absolute units relative
                    to volume.scale['resolution'].

        """
        pts_array = np.zeros([len(points),3])
        for i in range(len(pts_array)):
            pts_array[i,:] = points[i]
        
        points = pts_array

        if isinstance(self._points, type(None)):
            self._points = points
        else:
            self._points = np.concatenate(self._points, points)

        resolution = np.array(self._volume.scale['resolution']) / self._image_res
        chunk_size = np.array(self._volume.scale['chunk_sizes'])
        chunk_starts = (points // resolution).astype(int) // chunk_size * chunk_size
        for point, chunk_start in zip(points, chunk_starts):
            self._chunk_map[tuple(chunk_start)].add(tuple(point))

    def _load_chunk(self, chunk_start, chunk_end):
        # (No validation that this is a valid chunk_start.)
        return self._volume[chunk_start[0]:chunk_end[0],
                            chunk_start[1]:chunk_end[1],
                            chunk_start[2]:chunk_end[2]]

    def _load_points(self, chunk_map_key):
        chunk_start = np.array(chunk_map_key)
        points = np.array(list(self._chunk_map[chunk_map_key]))

        resolution = np.array(self._volume.scale['resolution']) / self._image_res
        indices = (points // resolution).astype(int) - chunk_start

        # We don't really need to load the whole chunk here:
        # Instead, we subset the chunk to the part that contains our points
        # This should at the very least save memory
        mn, mx = indices.min(axis=0), indices.max(axis=0)

        chunk_end = chunk_start + mx + 1
        chunk_start += mn
        indices -= mn

        chunk = self._load_chunk(chunk_start, chunk_end)
        return points, chunk[indices[:, 0], indices[:, 1], indices[:, 2]]

    def load_all(self, max_workers=4, return_sorted=True, progress=True):
        """Load all points in current list, batching by storage chunk.

        Parameters
        ----------
        max_workers :   int, optional
                        The max number of workers for parallel chunk requests.
        return_sorted : bool, optional
                        If True, will order the returned data to match the order
                        of the points as they were added.
        progress :      bool, optional
                        Whether to show progress bar.

        Returns
        -------
        points :        np.ndarray
        data :          np.ndarray
                        Parallel Numpy arrays of the requested points from all
                        cumulative calls to add_points, and the corresponding
                        data loaded from volume.

        """
        progress_state = self._volume.progress
        self._volume.progress = False
        pbar = tqdm.tqdm(total=len(self._chunk_map),
                         desc='Segmentation IDs',
                         disable=not progress)
        with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            point_futures = [ex.submit(self._load_points, k) for k in self._chunk_map]
            for f in futures.as_completed(point_futures):
                pbar.update(1)
        self._volume.progress = progress_state
        pbar.close()

        results = [f.result() for f in point_futures]

        if return_sorted:
            points_dict = dict(zip([tuple(p) for result in results for p in result[0]],
                                   [i for result in results for i in result[1]]))

            data = np.array([points_dict[tuple(p)] for p in self._points])
            points = self._points
        else:
            points = np.concatenate([result[0] for result in results])
            data = np.concatenate([result[1] for result in results])

        return points, data
    

def segIDs_from_pts(cv,coords,n=100000,max_tries = 3):
    seg_ids = []
    failed = []
    bins = np.array_split(np.arange(0,len(coords)),np.ceil(len(coords)/10000))
    
    for i in bins: 
        pt_loader = GSPointLoader(cv)
        pt_loader.add_points(coords[i])
        try:
            chunk_ids = pt_loader.load_all()[1].reshape(len(coords[i]),)
            seg_ids.append(chunk_ids)
        except:
            print('Failed, retrying')
            fail_check = 1
            while fail_check < max_tries:
                try:
                    chunk_ids = pt_loader.load_all()[1].reshape(len(coords[i]),)
                    seg_ids.append(chunk_ids)
                    fail_check = max_tries + 1
                except:
                    print('Fail: {}'.format(fail_check))
                    fail_check+=1
            
            if fail_check == max_tries:
                failed.append(i)       
    
    
    return cv.get_roots(np.concatenate(seg_ids))
    
    
def batch_roots(cv,df,n=100000):
    groups = int(np.ceil(len(df)/n))
    full = []
    df = df.join(pd.DataFrame(np.ones([len(df),2]),columns={'pre_roots','post_roots'},dtype='int64'))
    for i in range(groups+1):
        start = n*i
        stop = n*(i+1)
        pre = cv.get_roots(df.pre_id.values[start:stop])
        post = cv.get_roots(df.post_id.values[start:stop])
        try:
            df.loc[start:stop-1,'pre_roots'] = pre
            df.loc[start:stop-1,'post_roots'] = post
        except:
            print('Failed at:',start)
            return(df)
        print(start,stop)
    return(df)

def get_links(filename): 
    return(np.fromfile(filename,dtype=np.int32).reshape(-1, 6))
    
def flip_xyz_zyx_convention(array):
    assert array.shape[1] == 6
    array[:, 0:3] = array[:, 2::-1]
    array[:, 3:6] = array[:, 5:2:-1]


def flip_pre_post_order(array):
    assert array.shape[1] == 6
    tmp = array[:, 0:3].copy()
    array[:, 0:3] = array[:, 3:6]
    array[:, 3:6] = tmp


def seg_from_pt(pts,vol,image_res=np.array([4.3,4.3,45]),max_workers=4):
    ''' Get segment ID at a point. Default volume is the static segmentation layer for now. 
    Args:
        pts (list): list of 3-element np.arrays of MIP0 coordinates
        vol_url (str): cloud volume url
    Returns:
        list, segment_ID at specified point '''
    
    vol.progress = False
    seg_mip = vol.scale['resolution']
    res = seg_mip / image_res

    pts_scaled = [pt // res for pt in pts]
    results = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        point_futures = [ex.submit(lambda pt,vol: vol[list(pt)][0][0][0][0], k,vol) for k in pts_scaled]
        
        for f in futures.as_completed(point_futures):
            results=[f.result() for f in point_futures]
       
    return results

def init_table(filename):
        fileEmpty =  os.path.exists(filename)
        if not fileEmpty:
            df = pd.DataFrame(data = None, columns={'pre_id','post_id','pre_pt','post_pt','source'})
            df.to_csv(filename,index=False)
            print('table created')
        else:
            print('table exists')


def format_links(links):
    if len(links) > 0:
        flip_pre_post_order(links)
        links = links * (2, 2, 1, 2, 2, 1) + np.array([1,1,0,1,1,0])
        links_formatted = links.reshape(len(links)*2,3)
    else:
        links_formatted = None
    return(links_formatted)
            
def write_table(table_name,source_name,gs):
    links= get_links(source_name)
    links_formatted = format_links(links)
    if links_formatted is not None:
        gs.add_points(links_formatted)
        root_ids = gs.load_all()[1]
        root_ids = root_ids.reshape([len(root_ids)])
        pre_ids = root_ids[0::2]
        post_ids = root_ids[1::2]
        cols = {'pre_id','post_id','pre_pt','post_pt','source'}
        df = pd.DataFrame(columns=cols)

        df.pre_id = pre_ids
        df.post_id = post_ids
        df.pre_pt = list(links_formatted[0::2])
        df.post_pt = list(links_formatted[1::2])
        df.source = source_name.name
        df.to_csv(table_name, mode='a', header=False,index=False, encoding = 'utf-8')
