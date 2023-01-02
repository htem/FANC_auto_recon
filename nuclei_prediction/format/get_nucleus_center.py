import numpy as np
import pandas as pd
from cloudvolume import CloudVolume, Bbox
from taskqueue import queueable, LocalTaskQueue
from functools import partial
from tqdm import tqdm
from ..lib import *
from fanc import rootID_lookup, ngl_info

# -----------------------------------------------------------------------------
path_to_df = '/Users/sumiya/git/FANC_auto_recon/Output/soma_info_Aug2021ver5_newcave.csv'
colnames = ['nucID','center_xyz','nuc_xyz','nuc_svID','nuc_rootID','soma_xyz','soma_svID','soma_rootID','vol','voxel_size','bbx_min','bbx_max','center_for_CAVE']
seg = CloudVolume(ngl_info.seg['path'], use_https=True, agglomerate=False, cache=True, progress=False)
parallel_cpu = 1

def argwhere_from_outside(volume, value, bbox_size):
  ind = np.argwhere(volume == value)
  center = bbox_size/2

  distance = np.apply_along_axis(np.linalg.norm, 1, ind-center)
  sorted_indice = np.argsort(-distance)
  result = ind[sorted_indice]
  
  return result

@queueable
def task_get_new_center_for_CAVE(hoge):
    df = pd.read_csv(path_to_df, header=0, names=colnames, dtype={'center_for_CAVE': object})
    rowi = df.iloc[hoge,:].values
    loc_cn = pd.Series(np.array([rowi[1], rowi[2]]))
    loc_cn_np = loc_cn.str.strip("'()'").str.split(',',expand=True).astype(int).to_numpy()
    r_cn = rootID_lookup.segIDs_from_pts_cv(loc_cn_np,seg,progress=False)

    if r_cn[0] != r_cn[1]:
      # this means center_xyz is inside different seg or void
      try:
        bbox = pd.Series(np.array([rowi[10], rowi[11]]))
        bbox_np = bbox.str.strip("'()'").str.split(',',expand=True).astype(int).to_numpy()

        bbox_mip2 = Bbox(mip0_to_mip2_array(bbox_np[1]), mip0_to_mip2_array(bbox_np[0]))
        bbox_size_mip2 = mip0_to_mip2_array(bbox_np[1]) - mip0_to_mip2_array(bbox_np[0])

        seg_nuc = seg.download(bbox_mip2, mip=0, segids=r_cn[1], renumber=False) # mip 0 : 'resolution': [17.2, 17.2, 45.0]
        vol_temp = seg_nuc[:,:,:]
        vol_temp2 = np.where(vol_temp == r_cn[1], 1, 0)
        vol = np.squeeze(vol_temp2)
        location_one_from_outside = argwhere_from_outside(vol, value=1, bbox_size=bbox_size_mip2)
        location_one_from_inside = location_one_from_outside[::-1] # loc with the same root id of nuc_loc
        lchosen_mip0 = np.apply_along_axis(mip2_to_mip0_array, 1, location_one_from_inside, seg_nuc)
        
        k = 10
        for i in tqdm(range(0, len(lchosen_mip0), k)):
          lchosen_temp = lchosen_mip0[i:i+k-1, :] 
          r_newc = rootID_lookup.segIDs_from_pts_cv(lchosen_temp,seg,progress=False)
          for j in range(i,i+k):
            new_center_loc = lchosen_mip0[i+j]
            if r_cn[1] == r_newc[i+j]:
              break
          else:
              continue
          break

        newl_aslist = new_center_loc.tolist()
        df.at[hoge, 'center_for_CAVE'] = tuple(newl_aslist)

      except Exception as e:
        print('Failed. Start retrying {}'.format(e))
        max_tries = 10000
        fail_check = 1
        while fail_check < max_tries:
          try:
            bbox = pd.Series(np.array([rowi[10], rowi[11]]))
            bbox_np = bbox.str.strip("'()'").str.split(',',expand=True).astype(int).to_numpy()

            bbox_mip2 = Bbox(mip0_to_mip2_array(bbox_np[1]), mip0_to_mip2_array(bbox_np[0]))
            bbox_size_mip2 = mip0_to_mip2_array(bbox_np[1]) - mip0_to_mip2_array(bbox_np[0])

            seg_nuc = seg.download(bbox_mip2, mip=0, segids=r_cn[1], renumber=False) # mip 0 : 'resolution': [17.2, 17.2, 45.0]
            vol_temp = seg_nuc[:,:,:]
            vol_temp2 = np.where(vol_temp == r_cn[1], 1, 0)
            vol = np.squeeze(vol_temp2)
            location_one_from_outside = argwhere_from_outside(vol, value=1, bbox_size=bbox_size_mip2)
            location_one_from_inside = location_one_from_outside[::-1] # loc with the same root id of nuc_loc
            lchosen_mip0 = np.apply_along_axis(mip2_to_mip0_array, 1, location_one_from_inside, seg_nuc)
            
            k = 10
            for i in tqdm(range(0, len(lchosen_mip0), k)):
              lchosen_temp = lchosen_mip0[i:i+k-1, :] 
              r_newc = rootID_lookup.segIDs_from_pts_cv(lchosen_temp,seg,progress=False)
              for j in range(i,i+k):
                new_center_loc = lchosen_mip0[i+j]
                if r_cn[1] == r_newc[i+j]:
                  break
              else:
                  continue
              break

            newl_aslist = new_center_loc.tolist()
            df.at[hoge, 'center_for_CAVE'] = tuple(newl_aslist)

          except Exception as e2:
            print('Still failing: {}'.format(e2, fail_check))
            fail_check+=1    
    
    else:
      df.at[hoge, 'center_for_CAVE'] = loc_cn[0]

    df.to_csv(path_to_df, index=False)

if __name__ == "__main__":
  df = pd.read_csv(path_to_df, header=0)
  tq = LocalTaskQueue(parallel=parallel_cpu)
  tq.insert(( partial(task_get_new_center_for_CAVE, i) for i in range(0,len(df)) ))
  tq.execute(progress=True)
  print('Done')