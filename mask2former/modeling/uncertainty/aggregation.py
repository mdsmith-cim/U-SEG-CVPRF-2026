import torch
from torch.nn import functional as F
import logging
import math
from detectron2.structures import Boxes, Instances
from mask2former.utils.misc import getIoU
from .prediction_modeling import calculate_uncertainty
logger = logging.getLogger(__name__)


def nonBackgroundSuppression(multisample_model_output: list, num_classes: int):
    """
    Get non-background panoptic prediction for each sample for >1 samples from a DETR-style network.

    @param multisample_model_output: list of dict with keys "mask_cls_results" and "mask_pred_results". mask_cls_results key: class logits for each sample. mask_pred_results key: mask predictions for each sample.
    @param num_classes: Number of classes in the dataset/produced by the model.
    @return: List of length # samples, each entry a dict with keys 'cur_scores_all' (softmax scores of each detection) and 'cur_masks' (masks of each detection).
    """
    numSamples = len(multisample_model_output)

    perSamplePredictions = []
    for i in range(numSamples):
        masks = multisample_model_output[i]["mask_pred_results"]
        logits = multisample_model_output[i]["mask_cls_results"]

        scores_all = F.softmax(logits, dim=-1)
        scores, labels = scores_all.max(-1)
        masks = masks.sigmoid()

        # Avoid background classes - this is quite important actually it turns out
        keep = labels.ne(num_classes)
        cur_scores_all = scores_all[keep]
        cur_masks = masks[keep]

        perSamplePredictions.append({'cur_scores_all': cur_scores_all, 'cur_masks': cur_masks})
    return perSamplePredictions

def getMeanConfidenceAndInstances(reducedPredictions: list, numClasses: int, metadata, overlap_threshold: float):
    """
    Using softmax scores and masks from each sample, get the mean softmax vector for each pixel as well as per-sample instance segments.
    @param reducedPredictions: List of length # samples, each entry a dict with keys 'cur_scores_all' (softmax scores of each detection) and 'cur_masks' (masks of each detection)
    @param numClasses: Number of classes in the dataset/produced by the model
    @param metadata: Dataset metadata from detectron2 metadata catalog
    @param overlap_threshold: Area threshold that determines if the score-weighted mask response is strong enough to be considered a valid instance
    """
    device = reducedPredictions[0]['cur_masks'].device
    mask_size = reducedPredictions[0]['cur_masks'].shape[-2:]
    meanConfidence = torch.zeros((*mask_size, numClasses+1), device=device, dtype=torch.float32)
    numSamples = len(reducedPredictions)

    current_segment_id = 0
    instance_seg_per_sample = []
    instance_info_per_sample = []
    for s in range(numSamples):
        cur_scores_all = reducedPredictions[s]['cur_scores_all']
        cur_masks = reducedPredictions[s]['cur_masks']
        # When using timeseries, some pixels may be marked as NaN as they were not able to be mapped
        # We exclude these pixels from any predictions for a given sample
        # If no sample provides any prediction for those pixels the region will be marked as background
        is_nan_region = cur_masks.isnan().any(dim=0)
        cur_scores, cur_classes = cur_scores_all.max(-1)
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
        instance_seg = torch.zeros(mask_size, dtype=torch.int32, device=device)
        instance_info = []

        numDetections = cur_scores.shape[0]

        if numDetections != 0:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            cur_mask_ids[is_nan_region] = -1 # -1 ensures will never match to anything
            for k in range(numDetections):
                # Idea seems to be take areas weighted by softmax score against area of strong mask response
                # and only keep those that do well
                pred_class = cur_classes[k].item()
                pred_softmaxdist = cur_scores_all[k]
                isthing = pred_class in metadata.thing_dataset_id_to_contiguous_id.values()

                weighted_mask = cur_mask_ids == k
                original_mask = cur_masks[k] >= 0.5
                weighted_mask_area = weighted_mask.sum().item()
                original_area = original_mask.sum().item()
                mask = weighted_mask & original_mask
                mask_area = mask.sum().item()

                #print(f'For sample {s} and detection {k}, mask area is {mask_area}, original area is {original_area}, weighted mask area is {weighted_mask_area}')
                if weighted_mask_area > 0 and original_area > 0 and mask_area > 0:
                    if mask_area / original_area < overlap_threshold:
                        continue

                    meanConfidence[mask] += pred_softmaxdist

                    if isthing:
                        current_segment_id += 1
                        instance_seg[mask] = current_segment_id
                        instance_info.append(
                            {
                                "id": current_segment_id,
                                'softmax_dist': pred_softmaxdist.tolist()
                            }
                        )
        instance_seg_per_sample.append(instance_seg)
        instance_info_per_sample.append(instance_info)

    meanConfidence /= meanConfidence.sum(-1, keepdim=True)

    # Set any unassigned areas to background
    meanConfidence[meanConfidence.isnan().any(-1), :] = torch.tensor([0 if i < numClasses else 1 for i in range(numClasses + 1)], dtype=meanConfidence.dtype, device=device)

    return meanConfidence, instance_seg_per_sample, instance_info_per_sample

def panopticStuffAggregation(meanConfidence: torch.Tensor, metadata, numClasses: int):
    """
    Assign stuff classes to each pixel for panoptic segmentation using mean confidence scores.
    @param meanConfidence: Tensor of shape (h, w, # classes) with mean softmax scores for each pixel
    @param metadata: Dataset metadata from detectron2 metadata catalog
    @return: Tuple (stuff_segment_ids, stuff_segments_info) where stuff_segment_ids a tensor of shape (h, w) with segment ID for each pixel and stuff_segments_info a list of dicts with keys 'id', 'isthing', 'category_id', 'softmax_dist'
    """
    # Assign classes to each pixel
    scores, labels = meanConfidence.max(-1)

    stuff_segment_ids = torch.zeros(scores.shape, device=scores.device, dtype=torch.int32)
    stuff_segments_info = []
    current_segment_id = 0
    for c in labels.unique():
        # This is the same behaviour as in the original network processing
        # Likely to occur if class predictions change with the average being e.g. [0.32,0.32,0.34]
        # Then this serves as a basic uncertainty rejection
        if c == numClasses:
            continue
        match = labels == c
        c = c.item()
        is_thing = c in metadata.thing_dataset_id_to_contiguous_id.values()
        area = match.sum().item()
        # Note: multiple different stuff objects can be merged here so not the most useful
        # If pixel level confidence is desired just use meanConfidence
        segment_confidence = meanConfidence[match].mean(0)
        if area > 0:
            if is_thing:
                stuff_segment_ids[match] = -1
            else:
                current_segment_id += 1
                stuff_segment_ids[match] = current_segment_id

                stuff_segments_info.append({"id": current_segment_id, "isthing": bool(is_thing), "category_id": int(c), "score": segment_confidence.max().item(), "instance_data": {"softmax_dist": segment_confidence.tolist()}})
    return stuff_segment_ids, stuff_segments_info


def panopticInstanceAggregator(segment_ids: torch.Tensor, segments_info: list, instance_seg_per_sample: list, instance_info_per_sample: list, metadata, numClasses: int, overlap_threshold: float, voteThreshold: float):
    """
    Accumulate votes for instance segmentation, inspired by BSAS with IoU as a similarity metric.
    :param segment_ids: Tensor of shape (h, w) with segment ID for each pixel. -1 value = thing class yet to be assigned
    :param segments_info: List of dicts corresponding to segment IDs with keys 'id', 'isthing', 'category_id', 'softmax_dist'
    :param instance_seg_per_sample: List of length (# samples) with tensors of shape (h, w) with instance segment ID for each pixel
    :param instance_info_per_sample: List of length (# samples) with sublists containing dicts with keys 'id', 'softmax_dist'
    :param metadata: Dataset metadata from detectron2 metadata catalog
    :param overlap_threshold: IoU threshold used to determine if two instance segments are sufficiently similar across samples as to be considered the same instance and merged
    :param voteThreshold: Minimum # of matches (~= samples) required to create a thing segment. Specified as percentage of number of samples, with the floor operator applied.
    :return:segment_ids, segments_info: Final segmentation with both stuff and things: ID mapping for each pixel and associated information on each segment
    """
    # Reduce segments to only those with pixels where the consensus is that it is a category with a thing
    # This may include non-thing segments or segments with the wrong class in some samples,
    # as long as the final assigned class was determined to be a thing
    stuffArea = segment_ids != -1
    for seg in instance_seg_per_sample:
        seg[stuffArea] = -1
    del stuffArea
    numSamples = len(instance_info_per_sample)

    if voteThreshold is None or voteThreshold < 0:
        voteThreshold = 0
    elif voteThreshold > 1:
        voteThreshold = 1

    voteThreshold = math.floor(voteThreshold * numSamples)

    # Should never happen
    assert voteThreshold <= numSamples, f"Vote threshold {voteThreshold} is greater than number of samples {numSamples}."

    # Initialize tracking of segments by votes with first sample
    segmentCounter = {}
    for i in instance_info_per_sample[0]:
        seg_id = i['id']
        segmentCounter[seg_id] = {'count': 1,
                                  'samples': [0],
                                  'region': (instance_seg_per_sample[0] == seg_id).to(torch.int32),
                                  'softmax_dist': torch.tensor(i['softmax_dist'], device=segment_ids.device, dtype=torch.float32)}

    # *** Voting for instances***
    # Can't really do something like BSAS here - although this is inspired by it - as we are tracking potentially
    # multiple overlapping segments, rather than creating a final segmentation as we go
    # Our goal here is really to get the instance segments (borders), as we have already found the softmax vector
    # assigned to each pixel in a previous step
    # Note that instances here could be merged even if they represent different classes, although this is not too common
    for s in range(1, numSamples):
        segmentIDs = instance_seg_per_sample[s]
        # For each instance from a new sample...
        for inst in instance_info_per_sample[s]:
            i = inst['id']
            potent_match = segmentIDs == i # instance region
            # Nothing here: move on
            if potent_match.sum().item() == 0:
                continue
            # Compare this instance to existing (j) instances
            match_found = False
            for j, j_info in segmentCounter.items():
                current_match = j_info['region']
                iou = getIoU(potent_match, current_match.to(torch.bool))
                if iou >= overlap_threshold: # Match could even be of a different class perhaps; we don't want to exclude anything at this stage
                    # At least one match, maybe more
                    # We merge the segments and consider it a vote of sorts
                    match_found = True
                    j_info['count'] += 1
                    j_info['samples'].append(s)
                    j_info['region'] = potent_match + current_match
                    j_info['softmax_dist'] += torch.tensor(inst['softmax_dist'], device=segment_ids.device, dtype=torch.float32)

            # New segment - could be merged later with others
            # Note that by the nature of this segment voting the region covered by each segment may overlap with others
            if not match_found:
                segmentCounter[i] = {
                    'count': 1,
                    'samples': [s],
                    'region': potent_match.to(torch.int32),
                    'softmax_dist': torch.tensor(inst['softmax_dist'], device=segment_ids.device, dtype=torch.float32)
                }

    # Copy over thing segments with sufficient votes to the final segmentation, in order of # of votes
    # Segments with most votes are assigned first
    # Once assigned, pixels can not be taken by another segment, so some pixels may go unassigned
    # and some lower count segments may have portions that overlap with a greater vote count segment removed
    sortedSegments = sorted(segmentCounter.items(), key=lambda a: a[1]['count'], reverse=True)
    # Start at last index of stuff segment..apparently sometimes there can be literally no stuff segments!
    try:
        idx = segments_info[-1]['id'] + 1
    except IndexError:
        idx = 1
    for k, v in sortedSegments:
        # We use only pixels where we meet the required # of votes
        match = (v['region'] >= voteThreshold) & (segment_ids == -1)
        area = match.sum().item()
        if area > 0:
            # Use mean of instance softmaxes for score/category
            # Note that we could use the meanConfidence here but that would introduce noise that we may reject here e.g. by the vote threshold
            confidence = v['softmax_dist'] / v['count']
            score, cat = confidence.max(dim=0)
            cat = cat.item()
            # Background class due to confusion -> discard
            if cat == numClasses:
                print('Background class detected in instance segmentation! Discarding it...')
                continue
            is_thing = cat in metadata.thing_dataset_id_to_contiguous_id.values()

            # This shouldn't trigger under typical circumstances but can happen e.g. very high uncertainty (multiple thing classes nearly equally likely for several pixels,
            # calculating the mean across all pixels results in stuff class)
            if not is_thing:
                print('Not thing detected in instance segmentation! Discarding it...')
                continue

            segment_ids[match] = idx
            to_append = {"id": idx, "isthing": is_thing, "category_id": cat, "score": score.item(),
                         "instance_data": {"num_votes": v['count'],
                                             "samples": v['samples'],
                                             "source_id": [k],
                                             "softmax_dist": confidence.tolist()}}
            segments_info.append(to_append)
            idx += 1

    # Assign any remaining pixels to background/void
    segment_ids[segment_ids == -1] = 0
    return segment_ids, segments_info

@torch.no_grad()
def getIndividualInstances(batch_out, metadata, panoptic_on: bool):
    """
    For instance segmentation, generate instance segmentation for each sample.
    @param batch_out: List of length # samples, each entry a dict with keys unprocessed_logits & unprocessed_masks
    @param metadata: Dataset metadata from detectron2 metadata catalog
    @param panoptic_on: Whether panoptic segmentation is enabled
    @return: List of instance segmentation results, length # of samples
    """
    # Default instance segmentation uses top-n instances but this can result in multiple copies of the same instance
    # if the softmax distribution has a few roughly equal classes
    # This is not ideal as almost by definition one is going to be wrong, plus it really complicated matching instances
    # with > 1 sample as they are the same mask and (original) softmax distribution
    # So we treat it here more like object detection - force non background, but use max softmax for each "object"

    numSamples = len(batch_out)
    instances_per_sample = []
    for i in range(numSamples):
        mask_pred = batch_out[i]['unprocessed_masks']
        mask_cls = batch_out[i]['unprocessed_logits']

        image_size = mask_pred.shape[-2:]

        # BEGIN Not allowing duplicates approach
        # Force non-background scores
        scores_all = F.softmax(mask_cls, dim=-1)[:, :-1]
        scores, labels = scores_all.max(-1)
        # if this is panoptic segmentation, we only keep the "thing" classes
        if panoptic_on:

            keep = torch.zeros_like(labels, dtype=torch.bool)
            for i, lab in enumerate(labels):
                keep[i] = lab.item() in metadata.thing_dataset_id_to_contiguous_id.values()

            mask_pred = mask_pred[keep]
            scores_all = scores_all[keep]
            scores = scores[keep]
            labels = labels[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.region = mask_pred > 0
        pred_masks = result.region.float()
        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores_all = scores_all * mask_scores_per_image.view(-1,1) # Note: these are not normalized

        # We maintain scores and pred_classes for compatibility with concat and other non-averaging approaches
        result.scores = scores * mask_scores_per_image
        result.pred_classes = labels
        # END Not allowing duplicates approach

        instances_per_sample.append(result)
    return instances_per_sample

@torch.no_grad()
def instanceAggregator(instances_per_sample: list, overlap_threshold: float, voteThreshold: int, panoptic_on: bool, metadata):
    """
    Instance segmentation aggregation, given individual instance segmentation results.
    @param instances_per_sample: List of instance segmentation results, length # of samples
    @param overlap_threshold: IoU threshold used to determine if two instance segments are sufficiently similar across samples as to be considered the same instance and merged
    @param voteThreshold: Minimum # of matches (~= samples) required to create a thing segment.
    @return: Instances object with aggregated instance segmentation
    """
    numSamples = len(instances_per_sample)

    if voteThreshold is None:
        voteThreshold = 0
    # Can't have more votes than samples -> force to # samples
    # Occurs in timeseries case at beginning of sequence
    # General checks done in main.py
    if voteThreshold > numSamples:
        logger.warning(f"Vote threshold {voteThreshold} is greater than number of samples {numSamples}. Setting vote threshold to number of samples.")
        voteThreshold = numSamples

    # Initialize tracking of segments by votes with first sample
    segmentCounter = []

    # *** Voting for instances***
    # Same procedure as for panoptic segmentation using modified BSAS
    # for s in range(1, numSamples):
    for s in range(0, numSamples):
        # segmentIDs = instance_seg_per_sample[s]
        instances = instances_per_sample[s]
        # For each instance from a new sample...
        for i in range(len(instances)):
            potent_match = instances.region[i] # instance region
            potent_label = instances.scores_all[i].argmax().item()
            # Nothing here: move on
            if potent_match.sum().item() == 0:
                continue
            # Compare this instance to existing (j) instances
            match_found = False
            for j_info in segmentCounter:
                current_match = j_info['region']
                iou = getIoU(potent_match, current_match)
                current_label = j_info['softmax_dist'].argmax().item()
                if iou >= overlap_threshold and (potent_label == current_label): # Match could even be of a different class perhaps; we don't want to exclude anything at this stage
                    # At least one match, maybe more
                    # We merge the segments and consider it a vote of sorts
                    match_found = True
                    j_info['count'] += 1
                    j_info['samples'].append(s)
                    j_info['region'] = potent_match | current_match
                    j_info['instance_ids'].append(i)
                    j_info['softmax_dist'] += instances.scores_all[i]

            # New segment - could be merged later with others
            # Note that by the nature of this segment voting the region covered by each segment may overlap with others
            if not match_found:
                segmentCounter.append({
                    'count': 1,
                    'samples': [s],
                    'region': potent_match,
                    'instance_ids': [i],
                    'softmax_dist': instances.scores_all[i].detach().clone()
                })

    # Copy over thing segments with sufficient votes to the final segmentation, in order of # of votes
    # Segments with most votes are assigned first
    # Once assigned, pixels can not be taken by another segment, so some pixels may go unassigned
    # and some lower count segments may have portions that overlap with a greater vote count segment removed
    sortedSegments = sorted(segmentCounter, key=lambda a: a['count'], reverse=True)

    pred_masks = []
    scores = []
    pred_classes = []
    for seg in sortedSegments:
        match = seg['region']
        area = match.sum().item()
        if seg['count'] >= voteThreshold and area > 0:
            confidence = seg['softmax_dist'] / seg['count']
            score, cat = confidence.max(dim=0)

            # Case where due to merging and split softmax distributions we get a stuff class -> ignore
            if panoptic_on and (cat.item() not in metadata.thing_dataset_id_to_contiguous_id.values()):
                continue

            pred_masks.append(match)
            scores.append(score)
            pred_classes.append(cat)

    # Standard case
    if len(pred_masks) != 0:
        pred_masks = torch.stack(pred_masks)
        scores = torch.stack(scores)
        pred_classes = torch.stack(pred_classes)
    # Handle case of no predictions (i.e. very weak response from network)
    else:
        # As this only occurs if all "detections" are effectively 0 we take the easy way out and just keep one detection
        pred_masks = sortedSegments[0]['region'].unsqueeze(0)
        scores = sortedSegments[0]['softmax_dist'].max().unsqueeze(0) # Should just be 0
        pred_classes = sortedSegments[0]['softmax_dist'].argmax().unsqueeze(0) # Should just be 0

    result = Instances(pred_masks.shape[-2:])
    # mask (before sigmoid)
    result.pred_masks = pred_masks.float()
    result.pred_boxes = Boxes(torch.zeros(pred_masks.size(0), 4))
    result.scores = scores
    result.pred_classes = pred_classes

    return result

def naiveInstanceMean(batch_out, metadata, panoptic_on: bool, test_topk_per_image: int):

    numSamples = len(batch_out)
    mask_pred = torch.stack([batch_out[i]['unprocessed_masks'] for i in range(numSamples)]).mean(0)
    mask_cls = torch.stack([batch_out[i]['unprocessed_logits'] for i in range(numSamples)]).mean(0)

    # ***
    # Rest of this is default instance code
    # ***
    # mask_pred is already processed to have the same shape as original input
    image_size = mask_pred.shape[-2:]

    # [Q, K]
    scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    labels = torch.arange(mask_cls.shape[-1] - 1, device=mask_cls.device).unsqueeze(0).repeat(mask_cls.shape[0],
                                                                                                 1).flatten(0, 1)
    # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
    scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
    labels_per_image = labels[topk_indices]

    topk_indices = topk_indices // (mask_cls.shape[-1] - 1)
    # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
    mask_pred = mask_pred[topk_indices]

    # if this is panoptic segmentation, we only keep the "thing" classes
    if panoptic_on:
        keep = torch.zeros_like(scores_per_image).bool()
        for i, lab in enumerate(labels_per_image):
            keep[i] = lab.item() in metadata.thing_dataset_id_to_contiguous_id.values()

        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]
        mask_pred = mask_pred[keep]

    result = Instances(image_size)
    # mask (before sigmoid)
    result.pred_masks = (mask_pred > 0).float()
    result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
    # Uncomment the following to get boxes from masks (this is slow)
    # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

    # calculate average mask prob
    mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                result.pred_masks.flatten(1).sum(1) + 1e-6)
    result.scores = scores_per_image * mask_scores_per_image
    result.pred_classes = labels_per_image
    return result

def instanceConcatenate(instances_per_sample: list):
    """
    Straightforward concatenation of instance segmentation results.
    @param instances_per_sample: List of instance segmentation results, length # of samples
    @return: Instances object with aggregated instance segmentation
    """
    numSamples = len(instances_per_sample)
    pred_masks = []
    scores = []
    pred_classes = []
    for s in range(numSamples):
        pred_masks.append(instances_per_sample[s].region.float())
        scores.append(instances_per_sample[s].scores)
        pred_classes.append(instances_per_sample[s].pred_classes)

    pred_masks = torch.concatenate(pred_masks, dim=0)
    scores = torch.concatenate(scores, dim=0)
    pred_classes = torch.concatenate(pred_classes, dim=0)

    result = Instances(pred_masks.shape[-2:])
    result.pred_masks = pred_masks
    result.pred_boxes = Boxes(torch.zeros(pred_masks.size(0), 4))
    result.scores = scores
    result.pred_classes = pred_classes
    return result

@torch.no_grad()
def aggregateSemantic(results: list, semantic_aggregation_method='averaging'):
    """
    Aggregate semantic segmentation predictions.

    Args:
        results: list of dict with keys "mask_cls_results" and "mask_pred_results", each of which is a torch.tensor containing
        logit scores and masks for N objects
        semantic_aggregation_method (str): Method to aggregate semantic predictions. Default is 'averaging'.

    Returns:
        dict: Aggregated semantic segmentation predictions and uncertainty.
    """
    output = {}
    num_samples = len(results)

    # Averaging method is per-pixel average softmax
    # We get all per-sample segmentations and do basic mean
    if semantic_aggregation_method == 'averaging':
        samples = []
        for s in range(num_samples):
            mask_cls, mask_pred = results[s]["mask_cls_results"], results[s]["mask_pred_results"]
            mask_cls = F.softmax(mask_cls, dim=-1)
            mask_pred = mask_pred.sigmoid()
            # conf = (C, H, W)
            conf = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            conf /= conf.sum(0) # Mask weighting means does not add to 0 -> we normalize
            samples.append(conf)

        samples = torch.stack(samples)
        semseg = samples.nanmean(dim=0)
        uncert = calculate_uncertainty(samples)

        # Remove background
        semseg = semseg[:-1, ...]
        output['sem_seg'] = semseg
        output.update(uncert)

    else:
        raise ValueError(f"Unsupported semantic aggregation method: {semantic_aggregation_method}")

    return output


@torch.no_grad()
def aggregatePanoptic(multisample_model_output: list, num_classes: int, metadata, overlap_threshold: float, aggregation_overlap_threshold: float, aggregation_vote_threshold: float, panoptic_aggregation_method='averaging'):
    """
        Aggregate panoptic segmentation predictions.

        Args:
            multisample_model_output: list of dict with keys "mask_cls_results" and "mask_pred_results". mask_cls_result key: class logits for each sample. mask_pred_result key: mask predictions for each sample.
            num_classes (int): Number of classes in the dataset/produced by the model.
            metadata: Dataset metadata from detectron2 metadata catalog.
            overlap_threshold (float): Area threshold that determines if the score-weighted mask response is strong enough to be considered a valid instance.
            aggregation_overlap_threshold (float): IoU threshold used to determine if two instance segments are sufficiently similar across samples as to be considered the same instance and merged.
            aggregation_vote_threshold (float): Minimum number of matches (~= samples) required to create a thing segment as a percentage of samples.
            panoptic_aggregation_method (str): Method to aggregate panoptic predictions. Default is 'averaging'.

        Returns:
            tuple: Aggregated panoptic predictions (segment_ids, segments_info).
    """
    if panoptic_aggregation_method == 'averaging':
        reducedPredictions = nonBackgroundSuppression(multisample_model_output, num_classes)
        meanConfidence, instance_seg_per_sample, instance_info_per_sample = getMeanConfidenceAndInstances(reducedPredictions, num_classes, metadata, overlap_threshold)
        del reducedPredictions
        segment_ids, segments_info = panopticStuffAggregation(meanConfidence, metadata, num_classes)
        del meanConfidence
        segment_ids, segments_info = panopticInstanceAggregator(segment_ids, segments_info, instance_seg_per_sample, instance_info_per_sample, metadata, num_classes, aggregation_overlap_threshold, aggregation_vote_threshold)

        return segment_ids, segments_info
    else:
        raise NotImplementedError(f"Unsupported panoptic aggregation method: {panoptic_aggregation_method}")
