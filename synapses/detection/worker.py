from cloudvolume import CloudVolume
from cloudfiles import CloudFiles

import detection
import numpy as np
import os

def process(msg):
    cv_pos = msg["cv_pos"]
    cv_vec = msg["cv_vec"]
    cv_out = msg["cv_out"]
    voxel_size = msg["voxel_size"]
    scaling_factor = msg["scaling_factor"]
    data_bbox = msg["data_bbox"]
    bbox = msg["bbox"]
    padding = msg["padding"]
    param = msg["param"]

    vol_pos = CloudVolume(cv_pos, mip=voxel_size)
    vol_vec = CloudVolume(cv_vec, mip=voxel_size)


    start_pos = [max(d, b - p) for d, b, p in zip(data_bbox[0:3], bbox[0:3], padding)]
    end_pos = [min(d, b + p) for d, b, p in zip(data_bbox[3:6], bbox[3:6], padding)]
    print(start_pos)
    print(end_pos)

    pos_data = np.squeeze(vol_pos[start_pos[0]:end_pos[0],start_pos[1]:end_pos[1],start_pos[2]:end_pos[2], :]).astype(np.float32)/255

    parameters = detection.SynapseExtractionParameters(
            extract_type=param['extract_type'],
            cc_threshold=param['cc_threshold'],
            loc_type=param['loc_type'],
            score_thr=param['score_thr'],
            score_type=param['score_type'],
            nms_radius=param['nms_radius']
    )

    predicted_syns, scores = detection.find_locations(pos_data, parameters, voxel_size=voxel_size)

    new_scorelist = []
    filtered_list = []
    print(predicted_syns[0])
    for ii, loc in enumerate(predicted_syns):
        if (all(b1 <= p < b2 for b1, p, b2 in zip(bbox[0:3], loc+start_pos, bbox[3:6]))):
            score = scores[ii]
            if parameters.score_thr is not None:
                if score > parameters.score_thr:
                    filtered_list.append(loc)
                    new_scorelist.append(score)
            else:
                filtered_list.append(loc)
                new_scorelist.append(score)

    predicted_syns = filtered_list
    scores = new_scorelist

    vec_data = np.squeeze(vol_vec[start_pos[0]:end_pos[0],start_pos[1]:end_pos[1],start_pos[2]:end_pos[2], :])
    target_sites = detection.find_targets(predicted_syns, vec_data, voxel_size=scaling_factor)

    pairs = []
    for post, pre in zip(predicted_syns, target_sites):
        pairs.append(np.concatenate((post+start_pos, pre+start_pos)))

    out = np.stack(pairs).astype(np.int32)

    filename = f"{bbox[0]}-{bbox[3]}_{bbox[1]}-{bbox[4]}_{bbox[2]}-{bbox[5]}"
    folder = "_".join(str(x) for x in voxel_size)
    path = os.path.join(cv_out, folder)
    cf = CloudFiles(path)
    cf.put(filename, out.tobytes())


if __name__ == "__main__":
    parameter_dic = {
            "extract_type": "cc",
            "cc_threshold": 0.98,
            "loc_type": "edt",
            "score_thr": 12,
            "score_type": "sum",
            "nms_radius": None
    }


    msg = {
        "voxel_size" : [8.6, 8.6, 45],
        "scaling_factor" : [8, 8, 40],
        #"data_bbox" : [2500, 43000, 1200, 34500, 60000, 4399],
        "data_bbox" : [24000, 50000, 2200, 26000, 52000, 2600],
        #"bbox" : [14000, 52500, 2100, 14512, 53012, 2228],
        "bbox" : [24000, 50000, 2200, 26000, 52000, 2600],
        "padding" : [128,128,16],
        "param": parameter_dic,
        "cv_pos" : "gs://ranl_scratch_zetta_30/seg_test/synful/200520/743d23ede8b5f08c1b2979f7d2be846b",
        "cv_vec" : "gs://ranl_scratch_zetta_30/seg_test/synful_vec/200523/b64dfdc1721676c424c73e8b901f692a",
        "cv_out" : "gs://ranl_scratch_zetta_30/seg_test/synful_out_scratch",
    }

    #msg = {
    #    "voxel_size" : [8, 8, 45],
    #    "scaling_factor" : [8, 8, 40],
    #    "data_bbox" : [0, 0, 0, 896, 896, 175],
    #    "bbox" : [0, 0, 0, 896, 896, 175],
    #    "padding" : [128,128,16],
    #    "param": parameter_dic,
    #    "cv_pos" : "gs://zetta_lee_fly_vnc_001_synapse_cutout/synapse_cutout9/synapse_average",
    #    "cv_vec" : "gs://zetta_lee_fly_vnc_001_synapse_cutout/synapse_cutout9/synapse_vec_200523_average",
    #}

    process(msg)
