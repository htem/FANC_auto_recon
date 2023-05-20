#!/usr/bin/env python3
"""
Upload data to CAVE tables

See some examples at https://github.com/htem/FANC_auto_recon/blob/main/example_notebooks/update_cave_tables.ipynb
"""

from datetime import datetime
from textwrap import dedent

import numpy as np
import pandas as pd
from requests.exceptions import HTTPError
from caveclient import CAVEclient
from caveclient.chunkedgraph import root_id_int_list_check
from cloudvolume import CloudVolume
from cloudvolume.lib import green, red

from . import annotations, auth, lookup, statebuilder


def annotate_neuron(neuron: 'segID or point',
                    annotation_class: str,
                    annotation: str,
                    user_id: int,
                    table_name='neuron_information',
                    resolve_duplicate_anchor_points=False) -> dict:
    """
    Upload information about a neuron to a CAVE table.

    This function will validate that `annotation` is a valid annotation for the
    given `annotation_class`, according to the rules described at
    https://github.com/htem/FANC_auto_recon/wiki/Neuron-annotations#neuron_information
    and then post the `annotation_class, annotation` pair to the specified CAVE
    table.

    Arguments
    ---------
    neuron: int OR 3-length iterable of ints/floats
        Segment ID or point coordinate of a neuron to upload information about

    annotation_class: str
        Class of the annotation

    annotation: str
        Term to annotate the neuron with

    user_id: int
        The CAVE user ID number to associate with this annotation

    table_name: str
        Name of the CAVE table to upload information to. Only works
        with tables of schema "bound_double_tag_user".

    resolve_duplicate_anchor_points: bool or int
        This argument is passed to `lookup.anchor_point`, see its
        docstring for details.

    Returns
    -------
    dict: Response from server containing information about the
          success or failure of the upload
    """
    client = auth.get_caveclient()
    if isinstance(neuron, int):
        if not client.chunkedgraph.is_latest_roots(neuron):
            raise ValueError(f'{neuron} is not a current segment ID')
        segid = neuron
        point = lookup.anchor_point(neuron, resolve_duplicates=resolve_duplicate_anchor_points)
    else:
        try:
            iter(neuron)
            segid = lookup.segids_from_pts(neuron)[0]
            if segid == 0:
                raise ValueError(f'Point {neuron} is a location with no segmentation')
            point = lookup.anchor_point(segid, resolve_duplicates=resolve_duplicate_anchor_points)
            print(f'Found segID {segid} with anchor point {point}.')
        except TypeError:
            raise TypeError('First argument must be a segID or a point coordinate')

    assert annotations.is_allowed_to_post(segid, annotation_class, annotation, raise_errors=True)

    stage = client.annotation.stage_annotations(table_name)
    stage.add(
        pt_position=point,
        tag=annotation,
        tag2=annotation_class,
        user_id=user_id
    )
    response = client.annotation.upload_staged_annotations(stage)

    return response


def xyz_StringSeries2List(StringSeries: pd.Series):
    pts = StringSeries.str.strip('()').str.split(',',expand=True)
    return pts.astype(int).values.tolist()


class SomaTableOrganizer(object):
    def __init__(
        self,
        client=None
    ):
        self._client = client      
        
    def initialize(self,
                   soma_table_name="",
                   subset_table_name=""):
        """
        Set soma table and subset table you want to edit.

        ---Arguments---
        soma_table_name: string
            Name of soma table you want to edit. If "" (default), this will
            be set to the most recent soma table on CAVE: 'soma_mmmYYYY'
        subset_table_name: string
            Name of subset table you want to edit. You can choose from either
            the name of cell types ("neuron", "glia") or the name of subset tables
            on CAVE: 'celltype_soma_mmmYYYY'
        """
        if soma_table_name != "":
            self._soma_table_name = soma_table_name
        else:
            self._soma_table_name = self._client.info.get_datastack_info()['soma_table'].split("neuron_")[-1]

        self._subset_table_dict = {"neuron": "neuron_" + self._soma_table_name, "glia": "glia_" + self._soma_table_name}

        if subset_table_name in list(self._subset_table_dict.keys()):
            self._subset_table_name = self._subset_table_dict[subset_table_name]
        elif subset_table_name in list(self._subset_table_dict.values()):
            self._subset_table_name = subset_table_name
        else:
            raise ValueError("Need name of subset or subset table name. Choose from {}.".format(self._subset_table_dict))

        self.update_tables()
        txt_msg = """\
            Ready to update soma table: {} and subset soma table: {}
            Please make sure you have separate soma in each annotation and have all information required: {}.
            When you upload somas locating outside the dataset, use the coordinates on the final slice.""".format(self._soma_table_name, self._subset_table_name, self._required_props())
        print(dedent(txt_msg))

    @property
    def soma_table_name(self):
        return self._soma_table_name

    @soma_table_name.setter
    def soma_table_name(self, value):
        self._soma_table_name = value

    @property
    def subset_table_name(self):
        return self._subset_table_name

    @subset_table_name.setter
    def subset_table_name(self, value):
        self._subset_table_name = value

    @property
    def soma_table(self):
        return self._soma_table

    @property
    def subset_table(self):
        return self._subset_table

    @property
    def subset_table_dict(self):    
        return self._subset_table_dict

    @subset_table_dict.setter
    def subset_table_dict(self, value):
        self._subset_table_dict = value

    def _required_props(self):
        return self._client.schema.schema_definition('nucleus_detection')['definitions']['NucleusDetection']['required']

    @staticmethod
    def add_radius_column(soma_table):
        if ('bb_start_position' not in soma_table.columns) or ('bb_end_position' not in soma_table.columns):
            raise ValueError("Need 'bb_start_position' and 'bb_end_position' columns.")
        
        soma_table['radius'] = 0
        for i, r in soma_table.iterrows():
            if soma_table['bb_start_position'][i] is not np.nan and soma_table['bb_end_position'][i] is not np.nan:
                dist = np.array(soma_table['bb_end_position'][i]) - np.array(soma_table['bb_start_position'][i])
                soma_table.at[i,'radius'] = np.linalg.norm(dist)/2 # distance in voxel
            else:
                soma_table.at[i,'radius'] = 10 # np.nan
        return soma_table

    @staticmethod
    def find_manual_ids(existing_ids, initial_digit=1):
        initials = [int(str(i)[0]) for i in existing_ids]
        idx = np.where(np.array(initials)==initial_digit)
        return existing_ids[idx]

    def _get_man_id(self, initial_digit=1, digit=17):
        ExistingID = self.find_manual_ids(self._soma_table.id.values, initial_digit=1)
        if len(ExistingID) >0:
            MaxExistingID = np.max(ExistingID)
        else:
            MaxExistingID = initial_digit*10**(digit-1)
        return MaxExistingID + 1

    def _get_man_id_column(self, length, initial_digit=1, digit=17):
        ExistingID = self.find_manual_ids(self._soma_table.id.values, initial_digit=1)
        if len(ExistingID) >0:
            MaxExistingID = np.max(ExistingID)
        else:
            MaxExistingID =initial_digit*10**(digit-1)
        return [MaxExistingID + i for i in range(1, 1+length)]
    
    def _get_nuc_ids(self, pts, nucleus_segmentation_layer=None):
        info = self._client.info.get_datastack_info()
        if nucleus_segmentation_layer==None:
            nucleus_segmentation_layer = self._client.annotation.get_table_metadata(info['soma_table'])['flat_segmentation_source']
        nuclei_cv = CloudVolume( # mip4
            nucleus_segmentation_layer,
            progress=False,
            cache=False, # to avoid conflicts with LocalTaskQueue
            use_https=True,
            autocrop=True, # crop exceeded volumes of request
            bounded=False
        )
        nid = lookup.segids_from_pts_cv(pts, nuclei_cv, return_roots=False, progress=False)
        return nid

    def _check_change(self, table_name, timestamp=datetime.utcnow()):
        try:
            return self._client.materialize.live_live_query(table_name, timestamp, allow_missing_lookups=False).reset_index(level=0)
        except HTTPError as e:
            raise UpdateUnsuccessful(red(e.response.text + "Please check after 1 hour. Only works between 10 AM - 12 AM PST."))

    def update_tables(self, timestamp=datetime.utcnow()):
        """
        Update both soma table and subset table to the most recent version. Will give 
        an error if newly added points are not ingested yet.
        """
        self._soma_table = self._check_change(self._soma_table_name, timestamp)
        self._subset_table = self._check_change(self._subset_table_name, timestamp)

    def _validate(self, df):
        stage = self._client.annotation.stage_annotations(self._soma_table_name, schema_name="nucleus_detection", id_field=True)
        stage.add_dataframe(df) # check whether the df has necessary columns
        stage.clear_annotations()

        svIDs = lookup.svids_from_pts(df.pt_position)
        rIDs = root_id_int_list_check(self._client.chunkedgraph.get_roots(svIDs))

        overlap = np.isin(rIDs, root_id_int_list_check(self._soma_table.pt_root_id.values))
        if sum(overlap) == 0:
            pass
        else:
            raise UploadUnsuccessful("The information below has root ids that are already registed in soma table. \n {}".format(df.drop(columns=['id'])[overlap].to_string()))

    def join_table(self):
        return self._subset_table.join(self._soma_table.set_index('id'), on='target_id', lsuffix='_subset', rsuffix='_soma')
    
    def preview(self, asPoint=True, asSphere=False):
        """
        Generate Neuroglancer link to inspect somata on the subset table.

        ---Arguments---
        asPoint: bool (default True)
            If True, produce an annotation layer that has all the subset somas as points
        asSphere: bool (default True)
            If True, produce an annotation layer that has all the subset somas as sheperes.
            Their radius are estimated based on their bounding boxes.

        ---Returns---
        Neuroglancer url (as a string)
        """

        joined = self.join_table()
        st = joined.reindex(columns=['id', 'pt_root_id', 'pt_position', 'bb_start_position', 'bb_end_position'])

        annotations = []
        if asPoint == True and asSphere == False:
            LayerName = '{}_pt_{}'.format(self.subset_table_name, datetime.now().strftime("%Y%m%d"))
            annotations.append({'name':LayerName,'type':'points','data': st})
        elif asPoint == False and asSphere == True:
            LayerName = '{}_sp_{}'.format(self.subset_table_name, datetime.now().strftime("%Y%m%d"))
            st_r = self.add_radius_column(st)
            annotations.append({'name':LayerName,'type':'spheres','data': st_r})
        else:
            raise ValueError("Either asPoint or asSphere should be True")

        print(statebuilder.render_scene(annotations=annotations, nuclei=st.id.values, client=self._client))

    def add_dataframe(self, df: pd.DataFrame, batch_size=10000):
        """
        Add dataframe to both soma table and subset table.

        ---Arguments---
        df: pandas.DataFrame
            A dataframe with coordinates of new somata that you want to upload
        batch_size: int (default 10000)
            How many annotations you upload in each batch. The CAVE server does not
            allow users to upload more than 10000 annotations at a time.
        """
        print("Checking the format of your table...")
        self.update_tables()
        df_i = df.reset_index(drop=True)
        id_missing_idx = np.where(df_i['id'].isna())[0]

        new_man_ids = self._get_man_id_column(len(id_missing_idx))
        new_nuc_ids = self._get_nuc_ids(df_i['pt_position'].loc[id_missing_idx])
        j=0
        for i, idx in enumerate(id_missing_idx):
            if new_nuc_ids[i] != 0:
                df_i.at[idx, 'id'] = new_nuc_ids[i]
            else:
                df_i.at[idx, 'id'] = new_man_ids[j]
                j += 1
        df_i["id"] = pd.to_numeric(df_i["id"])
        self._validate(df_i)
        print("Ready to upload...")

        if len(df_i) <= 10000:
            stage = self._client.annotation.stage_annotations(self._soma_table_name, schema_name="nucleus_detection", id_field=True)
            stage.add_dataframe(df_i)
            self._client.annotation.upload_staged_annotations(stage)
            stage.clear_annotations()

            df_is = df_i.reindex(columns=['id'])
            df_is['target_id'] = df_is['id']
            # df_is = df_is.rename(columns={'id': 'target_id'})
            stage = self._client.annotation.stage_annotations(self._subset_table_name, schema_name="simple_reference", id_field=True)
            stage.add_dataframe(df_is)
            self._client.annotation.upload_staged_annotations(stage)
            stage.clear_annotations()

        else:
            minidfs = [df_i.loc[i:i+batch_size-1, :] for i in range(0, len(df_i), batch_size)]
            stage = self._client.annotation.stage_annotations(self._soma_table_name, schema_name="nucleus_detection", id_field=True)
            for dftmp in minidfs:
                stage.add_dataframe(dftmp)
                self._client.annotation.upload_staged_annotations(stage)
                stage.clear_annotations()

            df_is = df_i.reindex(columns=['id'])
            df_is['target_id'] = df_is['id']
            # df_is = df_is.rename(columns={'id': 'target_id'})
            minidfs = [df_is.loc[i:i+batch_size-1, :] for i in range(0, len(df_is), batch_size)]
            stage = self._client.annotation.stage_annotations(self._subset_table_name, schema_name="simple_reference", id_field=True)
            for dftmp in minidfs:
                stage.add_dataframe(dftmp)
                self._client.annotation.upload_staged_annotations(stage)
                stage.clear_annotations()

        # self.update_tables()
        print(green("Successfully uploaded!"))


def transfer_segmentation(from_layer, to_layer):
    # copy and paste segmentations 
    pass


def add_soma(points=None, is_neuron=True, nucleus_id=None):
    """
    Upload one new soma to a corresponding CAVE table.

    Arguments
    ---------
    points: list OR np.array
        A point coordinate of a soma to upload

    is_neuron: bool
        Whether the soma belongs to a neuron (True) or a glia (False).

    nucleus_id: int or np.uint64
        A nucleus ID that you want to use to annotate the soma. If you don't
        have preference, you can use None (by default). The code then will
        check the nucleus ID by looking up the same coordinate on the nucleus
        segmentation, and use the nucleus ID that it finds. If it cannot find
        anything, then the code will generate a "meaningless" artificial
        annotation ID for this soma.
    """
    sto = SomaTableOrganizer(client=auth.get_caveclient())
    if is_neuron:
        sto.initialize(subset_table_name='neuron')
    else:
        sto.initialize(subset_table_name='glia')
    upload_df = pd.DataFrame(columns=['pt_position', 'id'])
    if nucleus_id is not None:
        upload_df.loc[0] = [points, nucleus_id]
    else:
        upload_df.loc[0] = [points, np.nan]
    sto.add_dataframe(upload_df)
    # transfer_segmentation()


def add_soma_df(points: pd.DataFrame, is_neuron=True, pt_position_column='pt_position', id_column='id'):
    """
    Upload multiple new soma to a corresponding CAVE table.

    Arguments
    ---------
    points: pd.DataFrame
        Point coordinates (with or without IDs that you want to use) of somata to upload

    is_neuron: bool
        True, if it is neuron

    pt_position_column: str
        If points has a column of point coordinates and use a non-standarized name (i.e., not 'pt_position'), 
        you need to tell the name of the column to this code.

    id_column: str
        If points has a column of IDs and use a non-standarized name (i.e., not 'id'),  you need to tell the 
        name of the column to this code. If you don't have preference, you can use np.nan (by default). 
        The code then will check the nucleus IDs by looking up the same coordinates on the nucleus segmentation, 
        and use the nucleus IDs that it finds. If it cannot find anything, then the code will generate 
        "meaningless" artificial annotation IDs for this soma.
    """
    sto = SomaTableOrganizer(client=auth.get_caveclient())
    if is_neuron:
        sto.initialize(subset_table_name='neuron')
    else:
        sto.initialize(subset_table_name='glia')
    
    if len(points.columns)<2:
        points['id'] = np.nan

    points = points.rename(columns={pt_position_column: 'pt_position', id_column: 'id'})
    sto.add_dataframe(points)
    # transfer_segmentation()


class UploadUnsuccessful(Exception):
    def __init__(self, message):      
        super().__init__(red("Uploading failed. ") + message)


class UpdateUnsuccessful(Exception):
    def __init__(self, message):      
        super().__init__(red("Updating failed. ") + message)
