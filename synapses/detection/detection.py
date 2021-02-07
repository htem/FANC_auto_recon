import logging

import numpy as np
import numpy.ma as ma
from scipy import ndimage
from skimage import measure


logger = logging.getLogger(__name__)


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
    res = np.zeros_like(probmap)
    res[probmap > threshold] = 1
    res, num_labels = ndimage.label(res)
    regions = measure.regionprops(res)
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
            score_vol_crop = score_vol[z1:z2, y1:y2, x1:x2]
            score_crop = ma.masked_array(score_vol_crop, reg_mask)
            if score_type == 'sum':
                score = score_crop.sum()
            elif score_type == 'mean':
                score = score_crop.mean()
            elif score_type == 'max':
                score = score_crop.max()
            elif score_type == 'count':
                score = reg['area']
            else:
                raise RuntimeError('score not defined')
            scores.append(score)
        locs.append(loc_abs * voxel_size)
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
        loc_voxel = (loc / voxel_size).astype(np.uint32)
        dirvector = dirvectors[:, loc_voxel[0], loc_voxel[1], loc_voxel[2]]
        target_loc = loc + dirvector
        target_loc = np.round(target_loc / voxel_size) # snap to voxel grid,
        # assuming MSE trained direction vector models.
        target_loc *= voxel_size
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
    if len(distances) > 0:
        logger.debug('Average distance of synapses %0.2f' % np.mean(distances))
    logger.debug('Removed {} synapses because distance '
                 'smaller than {}'.format(len(to_remove), min_dist))
    target_locs = [loc.astype(np.int64) for loc in target_locs]
    return target_locs
