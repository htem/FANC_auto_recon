#!/usr/bin/env python3

import numpy as np
import pandas as pd
from datetime import datetime
from caveclient import CAVEclient
from caveclient.chunkedgraph import root_id_int_list_check
from cloudvolume.lib import green, red
from requests.exceptions import HTTPError
from textwrap import dedent
from .statebuilder import render_scene
from .lookup import segids_from_pts

CAVE_DATASETS = {'production': 'fanc_production_mar2021',
                 'sandbox': 'fanc_sandbox'}

class CAVEorganizer(object):
    """
    A manager for all the functions that interact with data on CAVE.
    You need a client to instantiate a organizer:

    client = CAVEclient(datastack_name='my_datastack')
    organizer = CAVEorganizer(client)

    Then,
    * organizer.update_soma allows you to upload new manually-detected somas to the soma table
    """
    def __init__(
        self,
        client,
        datastack_name=None,
        server_address=None,
        auth_token_file=None,
        auth_token_key=None,
        auth_token=None,
        global_only=False,
        pool_maxsize=None,
        pool_block=None,
        desired_resolution=None,
        info_cache=None
    ):
        if client is None:
            self.client = CAVEclient(datastack_name=datastack_name,
                                     server_address=server_address,
                                     auth_token_file=auth_token_file,
                                     auth_token_key=auth_token_key,
                                     auth_token=auth_token,
                                     global_only=global_only,
                                     pool_maxsize=pool_maxsize,
                                     pool_block=pool_block,
                                     desired_resolution=desired_resolution,
                                     info_cache=info_cache)
        else:
            self._client = client
        self._reset_services()

    def _reset_services(self):
        self._update_soma = None

    @property
    def client(self):
        return self._client

    def get_tables(self, datastack_name=None, version=None):
        return self._client.materialize.get_tables(datastack_name=datastack_name, 
                                                   version=version)

    def get_info(self, datastack_name=None):
        return self._client.info.get_datastack_info(datastack_name=datastack_name)

    @property
    def update_soma(self):
        if self._update_soma is None:
            self._update_soma = SomaTableOrganizer(
                client=self._client
            )
        return self._update_soma


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
            When you upload somas locating slightly outside the dataset and dorsal, use the coordinates on z=10 slice.""".format(self._soma_table_name, self._subset_table_name, self._required_props())
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

        svIDs = svids_from_pts(df.pt_position)
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

        print(render_scene(annotations=annotations, nuclei=st.id.values, client=self._client))

    def add_dataframe(self, df: pd.DataFrame, bath_size=10000):
        """
        Add dataframe to both soma table and subset table.

        ---Arguments---
        df: pandas.DataFrame
            A dataframe with coordinates of new somata that you want to upload
        bath_size: int (default 10000)
            How many annotations you upload in each batch. The CAVE server does not
            allow users to upload more than 10000 annotations at a time.
        """
        print("Checking the format of your table...")
        self.update_tables()
        df_i = df.reset_index(drop=True)
        df_i['id'] = self._get_man_id_column(len(df_i))
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
            minidfs = [df_i.loc[i:i+bath_size-1, :] for i in range(0, len(df_i), bath_size)]
            stage = self._client.annotation.stage_annotations(self._soma_table_name, schema_name="nucleus_detection", id_field=True)
            for dftmp in minidfs:
                stage.add_dataframe(dftmp)
                self._client.annotation.upload_staged_annotations(stage)
                stage.clear_annotations()

            df_is = df_i.reindex(columns=['id'])
            df_is['target_id'] = df_is['id']
            # df_is = df_is.rename(columns={'id': 'target_id'})
            minidfs = [df_is.loc[i:i+bath_size-1, :] for i in range(0, len(df_is), bath_size)]
            stage = self._client.annotation.stage_annotations(self._subset_table_name, schema_name="simple_reference", id_field=True)
            for dftmp in minidfs:
                stage.add_dataframe(dftmp)
                self._client.annotation.upload_staged_annotations(stage)
                stage.clear_annotations()

        # self.update_tables()
        print(green("Successfully uploaded!"))


class UploadUnsuccessful(Exception):
    def __init__(self, message):      
        super().__init__(red("Uploading failed. ") + message)


class UpdateUnsuccessful(Exception):
    def __init__(self, message):      
        super().__init__(red("Updating failed. ") + message)
