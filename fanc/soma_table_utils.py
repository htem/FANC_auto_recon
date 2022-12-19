import numpy as np
import pandas as pd
from datetime import datetime
from caveclient.chunkedgraph import root_id_int_list_check
from cloudvolume.lib import green, red
from requests.exceptions import HTTPError
from textwrap import dedent
from .statebuilder import render_scene
from .rootID_lookup import segIDs_from_pts_service

def xyz_StringSeries2List(StringSeries: pd.Series):
    pts = StringSeries.str.strip('()').str.split(',',expand=True)
    return pts.astype(int).values.tolist()

def which_label(id, label, dfs):
    label_list = list(label.values())
    for i in range(len(label)):
        if sum(np.isin(dfs[i]["old_nucID"], np.array(id, dtype=np.uint64))) == 1:
            return label_list[i]

class SomaTableOrganizer(object):
    def __init__(
        self,
        client=None
    ):
        self._client = client      
        
    def initialize(self,
                   soma_table_name="",
                   subset_table_name=""):
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
            Please make sure you have separate soma in each annotation and have all information required: {}.""".format(self._soma_table_name, self._subset_table_name, self._required_props())
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
        self._soma_table = self._check_change(self._soma_table_name, timestamp)
        self._subset_table = self._check_change(self._subset_table_name, timestamp)

    def _validate(self, df):
        stage = self._client.annotation.stage_annotations(self._soma_table_name, schema_name="nucleus_detection", id_field=True)
        stage.add_dataframe(df) # check whether the df has necessary columns
        stage.clear_annotations()

        svIDs = segIDs_from_pts_service(df.pt_position, return_roots=False) 
        rIDs = root_id_int_list_check(self._client.chunkedgraph.get_roots(svIDs))

        overlap = np.isin(rIDs, root_id_int_list_check(self._soma_table.pt_root_id.values))
        if sum(overlap) == 0:
            pass
        else:
            raise UploadUnsuccessful("The information below has root ids that are already registed in soma table. \n {}".format(df.drop(columns=['id'])[overlap].to_string()))
    
    def preview(self, asPoint=True, asSphere=False):
        st = self._soma_table.reindex(columns=['id', 'pt_root_id', 'pt_position', 'bb_start_position', 'bb_end_position'])

        annotations = []
        if asPoint:
            LayerName = '{}_pt_{}'.format(self.subset_table_name, datetime.now().strftime("%Y%m%d"))
            annotations.append({'name':LayerName,'type':'points','data': st})
        if asSphere:
            LayerName = '{}_sp_{}'.format(self.subset_table_name, datetime.now().strftime("%Y%m%d"))
            st_r = self.add_radius_column(st)
            annotations.append({'name':LayerName,'type':'sphere','data': st_r})

        print(render_scene(annotations=annotations, client=self._client))

    def add_dataframe(self, df: pd.DataFrame, bath_size=10000):
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