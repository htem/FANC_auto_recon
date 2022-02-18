import time
import copy
import numpy as np
from urllib.error import HTTPError
from caveclient import CAVEclient
from cloudfiles import CloudFiles
from taskqueue import RegisteredTask


class PerformMergeTask(RegisteredTask):
    """
    CAUTION: Large scale merging operation massively affects materialization of CAVE tables. 
             Contact Cave team before running this code.
    This code is adapted from Forrest Collman's work.
    """

    def __init__(
        self,
        pcg_source="",
        bucket_save_location="",
        A_sv_id=0,
        B_sv_id=0,
        A_sv_id_loc=(0, 0, 0),
        B_sv_id_loc=(0, 0, 0),
        resolution=(4.3, 4.3, 45),
        datastack_name="fanc_production_mar2021",
    ):
        """[summary]

        Args:
            pcg_source (str, optional): [description]. Defaults to "".
            bucket_save_location (str, optional): [description]. Defaults to "".
            A_sv_id (int, optional): [description]. Defaults to 0.
            B_sv_id (int, optional): [description]. Defaults to 0.
            A_sv_id_loc (tuple, optional): [description]. Defaults to (0,0,0).
            B_sv_id_loc (tuple, optional): [description]. Defaults to (0,0,0).
            resolution (tuple, optional): [description]. Defaults to (4.3,4.3,45).
            datastack_name (str, optional): [description]. Defaults to "fanc_production_mar2021".
        """
        super().__init__(
            pcg_source,
            bucket_save_location,
            A_sv_id,
            B_sv_id,
            A_sv_id_loc,
            B_sv_id_loc,
            resolution,
            datastack_name,
        )

        self.pcg_source = pcg_source
        self.bucket_save_location = bucket_save_location
        self.A_sv_id = A_sv_id
        self.B_sv_id = B_sv_id
        self.A_sv_id_loc = A_sv_id_loc
        self.B_sv_id_loc = B_sv_id_loc
        self.resolution = resolution
        self.datastack_name = datastack_name

    def execute(self):
        client = CAVEclient(self.datastack_name)
        roots = client.chunkedgraph.get_roots([self.A_sv_id, self.B_sv_id])
        r = {}

        if roots[0] != roots[1]:
            try:
                client.chunkedgraph.do_merge(
                    [self.A_sv_id, self.B_sv_id],
                    [np.array(self.A_sv_id_loc), np.array(self.B_sv_id_loc)],
                    resolution=self.resolution
                )
                r["did_merge"] = True
            except (Exception, HTTPError) as e:
                r = {}
                t = 10
                print("timeout error.. ")
                for i in range(t):
                    print("sleeping..")
                    time.sleep(30)
                    root_test = client.chunkedgraph.get_roots(
                        [self.A_sv_id, self.B_sv_id]
                    )
                    if root_test[0] == root_test[1]:

                        print("merge test found completion")
                        r["did_merge"] = True
                        r["new_root_id"] = root_test[1]
                        break
                if "did_merge" not in r.keys():
                    r["did_merge"] = "Merge appeared to have failed after {} minutes of waiting. {}".format(0.5*t, e)
        
        else:
            r["did_merge"] = False

        r["A_sv_id"] = self.A_sv_id
        r["B_sv_id"] = self.B_sv_id
        r["A_sv_id_loc"] = self.A_sv_id_loc
        r["B_sv_id_loc"] = self.B_sv_id_loc
        r["A_root"] = roots[0]
        r["B_root"] = roots[1]
        cf = CloudFiles(self.bucket_save_location)
        cf.put_json(f"{self.B_sv_id}.json", r)
        print(r)
        return


def create_nuc_merge_tasks(df, pcg_source, bucket_save_location, resolution=(4.3, 4.3, 45)):
    class PerformNucMergeTaskIterator(object):
        def __init__(self, df, pcg_source, bucket_save_location, resolution=(4.3, 4.3, 45)):
            self.pcg_source = pcg_source
            self.bucket_save_location = bucket_save_location
            self.df = df
            self.resolution = resolution

        def __len__(self):
            return len(self.df)

        def __getitem__(self, slc):
            itr = copy.deepcopy(self)
            itr.df = df.iloc[slc]
            return itr

        def __iter__(self):

            for num, row in self.df.iterrows():
                yield PerformMergeTask(
                    self.pcg_source,
                    self.bucket_save_location,
                    row.nuc_sv_id,
                    row.cell_sv_id,
                    row.nuc_sv_id_loc,
                    row.cell_sv_id_loc,
                    self.resolution,
                )

    return PerformNucMergeTaskIterator(df, pcg_source, bucket_save_location, resolution)