import numpy as np
import numpy.ma as ma
from scipy import ndimage
from skimage import measure
import edt
import os
import json
from cloudvolume import CloudVolume
from cloudfiles import CloudFiles


class SynapseExtractionParameters(object):
    '''

    Args:

        extract_type (``string``, optional):

            How to detect synapse candidate locations. CC --> For each connected
            component of the thresholded input array, one location is extracted.

        cc_threshold (``float``, optional):

            Where to threshold input array to extract connected component.

        loc_type (``string``, optional):

            How to extract location from connected component:
            edt --> euclidean distance transform.

        score_thr (``string``, optional):

            Only consider location with a score value greater
            than this threshold.

        score_type (``string``, optional):

            How to calculate the score. Possible options: sum, mean, max, count.


    '''

    def __init__(
            self,
            extract_type='cc',
            cc_threshold=None,
            loc_type='edt',
            # How to extract location from blob: edt --> euclidean distance transform
            score_thr=None,  # If locs should be filtered with threshold
            score_type=None,  # What kind of score to use.
            nms_radius=None
    ):
        # assert extract_type == 'cc', 'Synapse Detection currently only ' \
        #                              'implemented with option cc'  # TODO: Implement nms
        if extract_type == 'nms':
            assert nms_radius is not None
        self.extract_type = extract_type
        self.cc_threshold = cc_threshold if extract_type == 'cc' else None
        self.loc_type = loc_type if extract_type == 'cc' else None
        self.score_type = score_type if extract_type == 'cc' else None
        self.score_thr = score_thr
        self.nms_radius = nms_radius if extract_type == 'nms' else None


def __from_labels_to_edt(labels, voxel_size):
    boundaries = __find_boundaries(labels)
    boundaries = 1.0 - boundaries
    distances = ndimage.distance_transform_edt(
        boundaries,
        sampling=tuple(float(v) / 2 for v in voxel_size))
    distances = distances.astype(np.float32)

    # restore original shape
    downsample = (slice(None, None, 2),) * len(voxel_size)
    distances = distances[downsample]
    return distances


def __find_boundaries(labels):
    # labels: 1 1 1 1 0 0 2 2 2 2 3 3       n
    # shift :   1 1 1 1 0 0 2 2 2 2 3       n - 1
    # diff  :   0 0 0 1 0 1 0 0 0 1 0       n - 1
    # bound.: 00000001000100000001000      2n - 1

    dims = len(labels.shape)
    in_shape = labels.shape
    out_shape = tuple(2 * s - 1 for s in in_shape)

    boundaries = np.zeros(out_shape, dtype=np.bool)

    for d in range(dims):
        shift_p = [slice(None)] * dims
        shift_p[d] = slice(1, in_shape[d])

        shift_n = [slice(None)] * dims
        shift_n[d] = slice(0, in_shape[d] - 1)

        diff = (labels[tuple(shift_p)] - labels[tuple(shift_n)]) != 0

        target = [slice(None, None, 2)] * dims
        target[d] = slice(1, out_shape[d], 2)

        boundaries[tuple(target)] = diff

    return boundaries


def __from_probmap_to_labels(probmap, threshold):
    """Thresholds an intensity map and find connected components.

    Args:
        probmap (np.array): The original array with probabilities.
        threshold (int/float): threshold

    Returns:
        regions:
        res: numpy array in which each disconnected region has a unique ID.

    """
    res = np.zeros_like(probmap, dtype=np.uint8)
    res[probmap > threshold] = 1
    res, num_labels = ndimage.label(res)
    regions = measure.regionprops(res, probmap)
    return regions, res


def __from_labels_to_locs(labels, regions, voxel_size,
                          intensity_vol=None,
                          score_vol=None,
                          score_type=None):
    """Function that extracts locations from connected components.

    Args:
        labels (np.array): The array with connecected components (each marked
        with an unique ID).

        regions (regionsprops.regions): The regionsprops extracted from labels.

        voxel_size (np.array): voxel size

        intensity_vol (np.array): an array with the same shape as labels.
        If given, the maxima of this array represent the locations. If this
        is set to None, edt is calculated for the connected component itself
        and used for location extraction.

        score_vol (np.array): array to use to calculate the score from.

        score_type (str): how to combine the score values.

    Returns:
        locs: list of locations in world units.

    """

    locs = []
    scores = []
    for reg in regions:
        label_id = reg['label']
        z1, y1, x1, z2, y2, x2 = reg['bbox']
        crop = labels[z1:z2, y1:y2, x1:x2]
        if intensity_vol is None:
            lab_edt = __from_labels_to_edt(crop, voxel_size)
        else:
            lab_edt = intensity_vol[z1:z2, y1:y2, x1:x2]
        reg_mask = crop != label_id
        crop = ma.masked_array(lab_edt, reg_mask)
        loc_local = np.unravel_index(crop.argmax(), crop.shape)
        loc_abs = np.array(loc_local) + np.array([z1, y1, x1])
        # Obtain score based on score_vol.
        if score_vol is not None:
            if score_type == 'sum':
                score = reg.mean_intensity * reg.area
            elif score_type == 'mean':
                score = reg.mean_intensity
            elif score_type == 'max':
                score = reg.max_intensity
            elif score_type == 'count':
                score = reg['area']
            else:
                raise RuntimeError('score not defined')
            scores.append(score)
        locs.append(loc_abs)
    if score_vol is not None:
        assert len(locs) == len(scores)
        return locs, scores
    else:
        return locs


def find_locations(probmap, parameters,
                   voxel_size=(1, 1, 1)):
    """Function that extracts locations from an intensity / probability map.

    Args:
        probmap (np.array): Intensity array, higher value indicates presence
        of objects to be found.

        parameters (regionsprops.regions): synapse parameters

        voxel_size (np.array): voxel size

    Returns:

        locs: list of locations in world units.

    """
    voxel_size = np.array(voxel_size)
    if parameters.extract_type == 'cc':
        regions, pred_labels = __from_probmap_to_labels(probmap,
                                                        parameters.cc_threshold)
    else:
        raise RuntimeError(
            'unknown extract_type option set: {}'.format(parameters.loc_type))

    if parameters.extract_type == 'cc':
        assert parameters.loc_type == 'edt', 'unknown loc_type option set: {}'.format(parameters.loc_type)
        pred_locs, scorelist = __from_labels_to_locs(pred_labels,
                                                     regions,
                                                     voxel_size,
                                                     intensity_vol=edt.edt(pred_labels, anisotropy=voxel_size,
                                                     score_vol=probmap,
                                                     score_type=parameters.score_type)
    pred_locs = [loc.astype(np.int64) for loc in pred_locs]
    return pred_locs, scorelist


def find_targets(source_locs, dirvectors,
                 voxel_size=[1., 1., 1.], min_dist=0):
    """Function that finds target position based on a direction vector map.

    Args:
        source_locs (list): list with source locations in world units.
        dirvectors (np.array): map with [dim, source_locs.shape]
        voxel_size (np.array): voxel size
        min_dist (float/int): threshold to filter target locations based on
        the distance of dir vector. This modifies the source_locs input.
    Returns:
        locs: List of locations in world units.

    """
    target_locs = []
    distances = []
    for loc in source_locs:
        loc_voxel = loc.astype(np.uint32)
        dirvector = dirvectors[loc_voxel[0], loc_voxel[1], loc_voxel[2], :]
        target_loc = loc + dirvector / voxel_size
        target_locs.append(target_loc)
        dist = np.linalg.norm(np.array(list(loc)) - np.array(list(target_loc)))
        distances.append(dist)
    to_remove = []
    for ii, dist in enumerate(distances):
        if dist < min_dist:
            to_remove.append(ii)

    # Delete items in reversed order such that the indeces stay correct
    for index in sorted(to_remove, reverse=True):
        del source_locs[index]
        del target_locs[index]

    target_locs = [loc.astype(np.int64) for loc in target_locs]
    return target_locs


def process_task(msg):
    msg = json.loads(msg)

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

    parameters = SynapseExtractionParameters(
            extract_type=param['extract_type'],
            cc_threshold=param['cc_threshold'],
            loc_type=param['loc_type'],
            score_thr=param['score_thr'],
            score_type=param['score_type'],
            nms_radius=param['nms_radius']
    )

    predicted_syns, scores = find_locations(pos_data, parameters, voxel_size=voxel_size)

    new_scorelist = []
    filtered_list = []
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
    target_sites = find_targets(predicted_syns, vec_data, voxel_size=scaling_factor)

    pairs = []
    for post, pre in zip(predicted_syns, target_sites):
        pairs.append(np.concatenate((post+start_pos, pre+start_pos)))

    out = np.stack(pairs).astype(np.int32)

    filename = f"{bbox[0]}-{bbox[3]}_{bbox[1]}-{bbox[4]}_{bbox[2]}-{bbox[5]}"
    folder = "_".join(str(x) for x in voxel_size)
    path = os.path.join(cv_out, folder)
    cf = CloudFiles(path)
    cf.put(filename, out.tobytes())


def submit_tasks():
    from secrets import token_hex
    from slack_message import slack_message
    from copy import deepcopy

    parameter_dic = {
            "extract_type": "cc",
            "cc_threshold": 0.97,
            "loc_type": "edt",
            "score_thr": None,
            "score_type": "sum",
            "nms_radius": None
    }


    msg = {
        "voxel_size" : [8.6, 8.6, 45],
        "scaling_factor" : [8, 8, 40],
        "param": parameter_dic,
        "cv_pos" : "gs://ranl_scratch_zetta_30/seg_test/synful/200520/743d23ede8b5f08c1b2979f7d2be846b",
        "cv_vec" : "gs://ranl_scratch_zetta_30/seg_test/synful_vec/200523/b64dfdc1721676c424c73e8b901f692a",
    }

    run_name = token_hex(16)
    output_path = "gs://ranl_scratch_zetta_30/seg_test/{}".format(run_name)
    slack_message("output path `{}`".format(output_path))

    #data_bbox = [2500, 43000, 1200, 34500, 60000, 4399]
    data_bbox = [24000, 50000, 2200, 26000, 52000, 2600]
    chunk_size = [512, 512, 128]
    padding = [128,128,16]

    msg['data_bbox'] = data_bbox
    msg['padding'] = padding
    msg['cv_out'] = output_path

    tasks = []

    for x in range(data_bbox[0], data_bbox[3], chunk_size[0]):
        for y in range(data_bbox[1], data_bbox[4], chunk_size[1]):
            for z in range(data_bbox[2], data_bbox[5], chunk_size[2]):
                payload = deepcopy(msg)
                payload['bbox'] = [x, y, z,
                        min(data_bbox[3],x+chunk_size[0]),
                        min(data_bbox[4],y+chunk_size[1]),
                        min(data_bbox[5],z+chunk_size[2])]
                tasks.append(json.dumps(payload))

    return tasks


if __name__ == "__main__":
    parameter_dic = {
            "extract_type": "cc",
            "cc_threshold": 0.97,
            "loc_type": "edt",
            "score_thr": None,
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

    process_task(json.dumps(msg))
