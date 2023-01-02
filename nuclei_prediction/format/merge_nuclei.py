from taskqueue import LocalTaskQueue
import pandas as pd
import copy
from fanc.merge_operation import PerformMergeTask
from fanc import ngl_info

parallel_cpu = 1
merge_file = "/Users/sumiya/git/FANC_auto_recon/Output/proofread_soma_temp.csv"
seg_source = ngl_info.seg['path']


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

def main():
    df = pd.read_csv(merge_file, header=0) # read csv file and select pairs within the same cells
    df = df[(df.is_neuron=='y') & (df.is_inside=='y')]

    df_to_merge = df.reindex(columns=['nuc_svID','soma_svID', 'nuc_xyz', 'soma_xyz'])
    df_to_merge.columns =['nuc_sv_id', 'cell_sv_id', 'nuc_sv_id_loc', 'cell_sv_id_loc']
    nuc_xyz_df = df['nuc_xyz'].str.strip('()').str.split(',',expand=True)
    soma_xyz_df = df['soma_xyz'].str.strip('()').str.split(',',expand=True)
    df_to_merge['nuc_sv_id_loc'] = nuc_xyz_df.astype(int).values.tolist()
    df_to_merge['cell_sv_id_loc'] = soma_xyz_df.astype(int).values.tolist()
    return df_to_merge.reset_index(drop=True, inplace=True)

if __name__ == "__main__":
    df_to_merge = main()
    tq = LocalTaskQueue(parallel=parallel_cpu)
    tq.insert(create_nuc_merge_tasks(df_to_merge,
                                     seg_source,
                                     'file:///Users/sumiya/git/FANC_auto_recon/Output/log_merge', # path in cloudfiles format
                                     resolution=(4.3,4.3,45)))
    tq.execute(progress=True)
    print('Done')