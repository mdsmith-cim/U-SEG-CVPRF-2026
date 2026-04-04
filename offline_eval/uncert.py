# Based on https://github.com/IML-DKFZ/values/blob/main/evaluation/uncertainty_aggregation/aggregate_uncertainties.py
from scipy.signal import convolve
import numpy as np
import h5py

def patch_level_aggregation(image, patch_size = 10, mean=False):
    if type(patch_size) == int:
        patch_size = len(image.shape) * [patch_size]
    kernel = np.ones(patch_size)
    patch_aggragated = convolve(image, kernel, mode="valid")
    if mean:
        patch_aggragated = patch_aggragated / (np.prod(patch_size))
    all_max_indices = np.where(np.isclose(patch_aggragated, np.max(patch_aggragated)))
    max_indices = []
    for indices in all_max_indices:
        max_indices.append(indices[0])

    max_indices_slice = []
    for idx, index in enumerate(max_indices):
        max_indices_slice.append((int(index), int(index + patch_size[idx])))
    return {"max_score": float(np.max(patch_aggragated)), "bounding_box": max_indices_slice}


def image_level_aggregation(image, mean=False):
    if mean:
        return float(np.sum(image) / image.size)
    return {"max_score": float(np.sum(image))}

def getConfidence(h5data: h5py.Group, mode='aurc'):
    possible_uncert_types = ('pred_entropy_mask', 'expected_entropy_mask', 'mutual_information_mask', 'expected_entropy_class', 'expected_variance_mask', 'pred_entropy_class', 'mutual_information_class', 'softmax_cls_score', 'norm_sigmoid_mask_score', 'pred_variance_mask', 'class_mask_combined_score')

    assert mode in ('aurc', 'ood'), "Mode must be either 'aurc' or 'ood'"

    data = {}

    for u in possible_uncert_types:

        # Calculate mutual information if possible
        if u == 'mutual_information_mask':
            if 'pred_entropy_mask' in h5data and 'expected_entropy_mask' in h5data:
                uncert = np.asarray(h5data['pred_entropy_mask']) - np.asarray(h5data['expected_entropy_mask'])
            else:
                continue
        elif u == 'mutual_information_class':
            if 'pred_entropy_class' in h5data and 'expected_entropy_class' in h5data:
                uncert = np.asarray(h5data['pred_entropy_class']) - np.asarray(h5data['expected_entropy_class'])
            else:
                continue
        elif u in h5data:
            uncert = np.asarray(h5data[u])
        else:
            continue

        # AURC is for failure detection
        # Use negative uncertainty as confidence scoring function
        # If already a confidence measure w.r.t. correctness of prediction leave alone
        if mode == 'aurc':

            patch_level = patch_level_aggregation(uncert)['max_score']
            image_level = image_level_aggregation(uncert)['max_score']

            if u in ('pred_entropy_mask', 'expected_entropy_mask', 'mutual_information_mask', 'expected_entropy_class', 'expected_variance_mask', 'pred_entropy_class', 'mutual_information_class', 'pred_variance_mask'):
                patch_level = -patch_level
                image_level = -image_level

        # OOD requires a confidence measure in the image containing OOD objects
        # So with uncertainty measures like entropy - bigger is more confident
        # With confidence measures like softmax the score is for the predicted class so we do 1 - score
        elif mode == 'ood':
            if u in ('softmax_cls_score', 'norm_sigmoid_mask_score', 'class_mask_combined_score'):
                uncert = 1 - uncert

            patch_level = patch_level_aggregation(uncert)['max_score']
            image_level = image_level_aggregation(uncert)['max_score']

        # noinspection PyUnboundLocalVariable
        data[u] = {'Patch Level': patch_level, 'Image Level': image_level}

    return data
