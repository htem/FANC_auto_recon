#!/usr/bin/env python3

import collections
from concurrent import futures
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import tqdm
import cloudvolume

from . import auth

default_proofreading_tables = ['proofread_first_pass', 'proofread_second_pass']
default_svid_lookup_url = 'https://services.itanna.io/app/transform-service/query/dataset/fanc_v4/s/2/values_array_string_response/'


def proofreading_status(segid: int or list[int],
                        table_names: str or list[str] = default_proofreading_tables,
                        timestamp='now') -> None or str or tuple(str, list):
    """
    Determine whether a segment has been marked as proofread.

    Arguments
    ---------
    segid: int
      The ID of the segment to query

    table_names: str, or list of str
      The name(s) of the CAVE proofreading table(s) to query

    timestamp : datetime, str, or None (default 'now')
      The timestamp at which to query the segment's proofreading status.
      If None, use the timestamp of the latest materialization.
      If 'now', use the current time.
      If datetime, use the time specified by the user.

    Returns
    -------
    None: The given segment ID has not been marked as proofread in any of the
      given tables.
    str: The name of the proofreading table in which the segment is marked as
      proofread.
    2-tuple of (str, list): The name of the proofreading table in which an earlier
      version of this the segment was marked as proofread, and a list
      containing the previous segment ID(s) that were marked as proofread.
    """
    if isinstance(segid, (int, np.integer)):
        return proofreading_status([segid], table_names=table_names, timestamp=timestamp)[0]

    client = auth.get_caveclient()
    if timestamp in ['now', 'live']:
        timestamp = datetime.utcnow()
    elif timestamp is None:
        timestamp = client.materialize.get_timestamp()

    if not all(client.chunkedgraph.is_latest_roots(segid, timestamp=timestamp)):
        raise KeyError('A given ID(s) is not valid at the given timestamp.'
                       ' Use updated IDs or provide the timestamp where'
                       ' the ID(s) is valid.')

    if isinstance(table_names, str):
        table_names = [table_names]

    results = pd.Series(index=segid, data=None, dtype=object)
    for table_name in table_names[::-1]:
        table = client.materialize.live_live_query(table_name, timestamp)
        results.loc[results.isna() & results.index.isin(table.valid_id)] = table_name
        if results.notna().all():
            return results.loc[segid].to_list()
        results.loc[results.isna()] = table.groupby('pt_root_id')['valid_id'].apply(lambda x: (table_name, list(x)))
        if results.notna().all():
            return results.loc[segid].to_list()

    results.loc[results.isna()] = None
    return results.loc[segid].to_list()


def annotations(segid: int or list[int],
                table_name: str = 'neuron_information',
                return_as='list',
                slow_mode=False) -> list or pd.DataFrame:
    """
    Get a cell's annotations from a CAVE table.

    Arguments
    ---------
    segid: int
      The segment ID to query

    table_name: str (default 'neuron_information')
      The name of the CAVE table to query. If `return_as` is set to
      'list', the table must have a column named 'tag'.

    return_as: str
      'list' (default)
        Return the `tag` column of the CAVE table, as a list of strings
      'dataframe'
        Return the entire dataframe from CAVE

    Returns
    -------
    list of strings OR pd.DataFrame, depending on `return_as`
    """
    if isinstance(segid, (int, np.integer)) and return_as == 'list':
        return annotations([segid], table_name=table_name, return_as='list',
                           slow_mode=slow_mode)[0]
    elif isinstance(segid, (int, np.integer)):
        segid = [segid]

    client = auth.get_caveclient()
    now = datetime.utcnow()
    try:
        if slow_mode:
            table = client.materialize.live_live_query(table_name, now)
            table = table.loc[table.pt_root_id.isin(segid)]
        else:
            table = client.materialize.live_live_query(
                table_name,
                now,
                filter_in_dict={table_name: {'pt_root_id': segid}},
            )
    except requests.exceptions.HTTPError as e:
        if 'returned no results' not in e.args[0]:
            raise e

    if return_as == 'dataframe':
        return table
    elif return_as != 'list':
        raise ValueError(f"'return_as' must be 'list' or 'dataframe'"
                         f" but was '{return_as}'")
    try:
        # Return tags as list
        return [[anno for anno in table.loc[table.pt_root_id == s, 'tag']]
                 for s in segid]
    except KeyError:
        raise KeyError(f"Table '{table_name}' has no column named 'tag'")


def svids_from_pts(pts, service_url=default_svid_lookup_url):
    """
    Return the supervoxel IDs for a set of points.

    This function relies on an external service hosted on services.itanna.io,
    created and maintained by Eric Perlman, which provides very fast svid
    lookups. If this service is down, you can try the slower version
    instead, `fanc.lookup.segids_from_pts_cv()`.

    Arguments
    ---------
    pts: Nx3 iterable (list / tuple / np.ndarray / pd.Series)
      Points to query. Provide these in xyz order and in mip0 voxel coordinates.

    --- Returns ---
    The requested supervoxel IDs as a list of ints.
    Order is preserved - the svid corresponding to the Nth point in the
    argument will be the Nth value in the returned list.
    """
    if isinstance(pts, pd.Series):
        pts = np.vstack(pts)

    pts = np.array(pts, dtype=np.uint32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 3)
    r = requests.post(service_url, json={
        'x': list(pts[:, 0].astype(str)),
        'y': list(pts[:, 1].astype(str)),
        'z': list(pts[:, 2].astype(str))
    })
    r = r.json()['values'][0]
    return [int(i) for i in r]


def segids_from_pts(pts,
                    timestamp='now',
                    service_url=default_svid_lookup_url,
                    **kwargs):
    """
    Return the segment IDs (also called root IDs) for a set of points
    at a specified timestamp.

    --- Arguments ---
    pts: Nx3 iterable (list / tuple / np.ndarray / pd.Series)
      Points to query. Provide these in xyz order and in mip0 voxel coordinates.
    timestamp: 'now' or None or datetime
      If 'now' or None, look up the current rootID corresponding to the point
      location. Otherwise, look up the rootID for the point location at the
      specified time in the past.

    --- Additional kwargs ---
    cv: cloudvolume.CloudVolume
      If provided, lookup rootIDs using the given cloudvolume instead of the
      default one. Not common to need this.

    --- Returns ---
    The requested segIDs as a numpy array of uint64 values
    Order is preserved - the segID corresponding to the Nth point in
    the argument will be the Nth value in the returned array.
    """
    if timestamp == 'now':
        # cv.get_roots interprets timestamp=None as requesting the latest root
        timestamp = None

    svids = svids_from_pts(pts, service_url=service_url)

    if 'cv' in kwargs:
        cv = kwargs['cv']
    else:
        cv = auth.get_cloudvolume()

    return cv.get_roots(svids, timestamp=timestamp).astype(np.int64)


anchor_point_sources = ['somas_dec2022', 'peripheral_nerves', 'neck_connective']
def anchor_point(segid, source_tables=anchor_point_sources,
                 timestamp='now', resolve_duplicates=False,
                 select_nth_duplicate: int = 0, slow_mode=False) -> np.ndarray:
    """
    Return a representative "anchor" point for each of the given
    segment ID(s).

    The returned point is expected to be a stable identifier for the
    segment, meaning that correct proofreading operations should not
    disconnect this point from the main body of the segment.

    Arguments
    ---------
    segid: int, or iterable of ints
      The ID(s) of the segment(s) to look up anchor points for.

    source_tables: iterable of str
      An list of names of CAVE tables to search for anchor points. This list
      must be ordered by priority, as the first table that contains a point for
      this segment will have its point used.

    timestamp: None or 'now' or datetime (default 'now')
      The timestamp to use to query the CAVE tables.
      If None, use the timestamp of the latest materialization.
      If 'now', use the current time.
      If datetime, use that time.

    resolve_duplicates: bool (default False)
    select_nth_duplicate: int (default 0)
      If multiple anchor points are found within a single table for a
      single segment ID, raise an error (if resolve_duplicates==False) or
      select one of the points (if resolve_duplicates==True) by sorting the
      points by x coordinate and then picking the Nth one, with N given by
      select_nth_duplicate. (N is zero-indexed, so the default value of
      select_nth_duplicate=0 results in selection of the point with the
      smallest x coordinate). select_nth_duplicate is ignored if
      resolve_duplicates==False.

    Returns
    -------
    points: np.ndarray (3-length vector or Nx3 array)
      If segid is an int, returns a single xyz point coordinate with shape (3,).
      If segid is iterable, returns an Nx3 array of xyz point coordinates.
        Order is preserved (that is, points[n] is the  point for segid[n]).
    """
    try: iter(segid)
    except:
        return anchor_point(
            [segid], source_tables=source_tables, timestamp=timestamp,
            resolve_duplicates=resolve_duplicates
        )[0]

    client = auth.get_caveclient()
    if timestamp in ['now', 'live']:
        timestamp = datetime.utcnow()
    elif timestamp is None:
        timestamp = client.materialize.get_timestamp()

    if not all(client.chunkedgraph.is_latest_roots(segid, timestamp=timestamp)):
        raise KeyError('A given ID(s) is not valid at the given timestamp.'
                       ' Use updated IDs or provide the timestamp where'
                       ' the ID(s) is valid.')

    anchor_points = pd.Series(index=set(segid), dtype=object)

    for table in source_tables:
        unanchored_ids = anchor_points[anchor_points.isna()].index.values
        if slow_mode:
            points = client.materialize.live_live_query(table,
                                                        timestamp=timestamp)
            points = points.loc[points.pt_root_id.isin(unanchored_ids)]
        else:
            points = client.materialize.live_live_query(
                table,
                filter_in_dict={table: {'pt_root_id': unanchored_ids}},
                timestamp=timestamp
            )
        for seg, point in points.groupby('pt_root_id'):
            if len(point) > 1:
                # Sort points by x coordinate
                point = point.sort_values('pt_position', key=lambda x: x.str[0])
                if table == 'somas_dec2022':
                    raise ValueError('Multiple somas points found for segid'
                                     f' {seg} in table "{table}".')
                elif not resolve_duplicates:
                    raise ValueError('Multiple anchor points found for segid'
                                     f' {seg} in table "{table}":\n'
                                     f'{np.vstack(point.pt_position)}.\nSet'
                                     ' resolve_duplicates to choose one.')
                if select_nth_duplicate >= len(point):
                    raise ValueError('select_nth_duplicate is too large given'
                                     f' that {len(point)} points were found'
                                     f' for segid {seg} in table "{table}".')
                anchor_points.loc[seg] = point['pt_position'].iloc[select_nth_duplicate]
            else:
                anchor_points.loc[seg] = point['pt_position'].iloc[0]
        if not any(anchor_points.isna()):
            break

    if any(anchor_points.isna()):
        raise ValueError(f'No anchor point found for segid(s)'
                         f' {anchor_points[anchor_points.isna()].index.values}'
                         f' in tables {source_tables}')
    return np.vstack(anchor_points[segid])


# TODO implement the raise kwargs
def somas_from_segids(segid,
                      table='default_soma_table',
                      select_columns=['id', 'volume', 'pt_root_id', 'pt_position'],
                      timestamp='now',
                      raise_not_found=True,
                      raise_multiple=True):
    """
    Given a segID (ID of an object from the full dataset segmentation),
    return information about its soma listed in the soma table.

    Arguments
    ---------

    segid: int or iterable of ints
      The ID(s) of the segment(s) to look up soma information for.

    table: str (default 'default_soma_table')
      The name or nickname of the soma table to query. Available nicknames:
        'default_soma_table' or 'neuron' or 'neurons' -> neuron_somas_dec2022
        'all' or 'somas' -> somas_dec2022
        'glia' -> glia_somas_dec2022

    select_columns: list of str (default ['id', 'volume', 'pt_root_id', 'pt_position'])
      The columns to get from the soma table.

    timestamp: None or 'now' or datetime (default 'now')
      The timestamp to use to query the CAVE tables.
      If None, use the timestamp of the latest materialization.
      If 'now', use the current time.
      If datetime, use that time.

    raise_not_found: bool (default True)
      If no entry is found in the soma table for the given segID, raise an
      exception. Otherwise, return None.

    raise_not_found: bool (default True)
      If multiple entries are found in the soma table for the given segID,
      raise an exception. Otherwise, return all soma table entries

    Returns
    -------
    pd.DataFrame containing the soma table entries for the given segID(s).
    """
    try: iter(segid)
    except: segid = [segid]

    client = auth.get_caveclient()
    if timestamp in ['now', 'live']:
        timestamp = datetime.utcnow()
    elif timestamp is None:
        timestamp = client.materialize.get_timestamp()

    if not all(client.chunkedgraph.is_latest_roots(segid, timestamp=timestamp)):
        raise KeyError('A given ID(s) is not valid at the given timestamp.'
                       ' Use updated IDs or provide the timestamp where'
                       ' the ID(s) is valid.')

    if table in [None, 'default_soma_table']:
        table = client.info.get_datastack_info()['soma_table']
        select_columns = None  # Feature not currently supported on reference tables
    elif table in ['all', 'somas']:
        table = 'somas_dec2022'
    elif table in ['neurons', 'neuron']:
        table = 'neuron_somas_dec2022'
        select_columns = None  # Feature not currently supported on reference tables
    elif table == 'glia':
        table = 'glia_somas_dec2022'
        select_columns = None  # Feature not currently supported on reference tables
    somas = client.materialize.query_table(table,
                                           select_columns=select_columns,
                                           timestamp=timestamp)
    return somas.loc[somas.pt_root_id.isin(segid)]



# The code below implements a slower version of segIDs_from_pts_service that
# you should never need to run as long as the service is operational. If the
# service goes down, try using segIDs_from_pts_cv instead
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
        self._image_res = np.array([4.3, 4.3, 45])
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
        pts_array = np.zeros([len(points), 3])
        for i in range(len(pts_array)):
            pts_array[i, :] = points[i]

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


def segids_from_pts_cv(pts,
                       cv=auth.get_cloudvolume(),
                       n=100000,
                       max_tries=3,
                       return_roots=True,
                       max_workers=4,
                       progress=True,
                       timestamp=None):
    """
    Query cloudvolume object for root or supervoxel IDs.

    This method is slower than segIDs_from_pts, but does not depend on
    the supervoxel ID lookup service created by Eric Perlman and hosted on
    services.itanna.io. As such, this function might be useful if that service
    is not available for some reason.

    This function may not actually work as-is (it threw errors when tested in
    December 2022) but presumably it's close to working, because it did work in
    the past, so we can try to revive it if the need arises.

    --- Arguments ---
    pts: Nx3 numpy array or pandas Series
      Points to query, in xyz order and in mip0 coordinates.
    n: int (default 100,000)
      number of coordinates to query in a single batch. Default is 100000,
      which seems to prevent server errors.
    max_tries: int (default 3)
      number of attempts per batch. Usually if it fails 3 times, something is
      wrong and more attempts won't work.
    return_roots: bool (detault True)
      If true, will look up root ids from supervoxel ids. Otherwise, supervoxel
      ids will be returned.

    --- Returns ---
    root IDs or supervoxel IDs for queried coordinates as int64
    """

    if hasattr(cv, 'agglomerate'):
        cv.agglomerate = False

    # Reshape from list entries if dataframe column is passed
    if isinstance(pts, pd.Series):
        pts = pts.reset_index(drop=True)
        pts = np.concatenate(pts).reshape(-1, 3)

    sv_ids = []
    failed = []
    bins = np.array_split(np.arange(0, len(pts)), np.ceil(len(pts) / 10000))

    for i in bins:
        pt_loader = GSPointLoader(cv)
        pt_loader.add_points(pts[i])
        try:
            chunk_ids = pt_loader.load_all(max_workers=max_workers, progress=progress)[1].reshape(len(pts[i]), )
            sv_ids.append(chunk_ids)
        except:
            print('Failed, retrying')
            fail_check = 1
            while fail_check < max_tries:
                try:
                    chunk_ids = pt_loader.load_all(max_workers=max_workers, progress=progress)[1].reshape(len(pts[i]), )
                    sv_ids.append(chunk_ids)
                    fail_check = max_tries + 1
                except:
                    print('Fail: {}'.format(fail_check))
                    fail_check += 1

            if fail_check == max_tries:
                failed.append(i)

    sv_id_full = np.concatenate(sv_ids)

    if return_roots:
        root_ids = cv.get_roots(sv_id_full, timestamp=timestamp)
        return root_ids
    else:
        return sv_id_full
