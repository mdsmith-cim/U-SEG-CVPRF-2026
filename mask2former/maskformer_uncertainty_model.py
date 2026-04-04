# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Tuple

import torch
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from torch import nn
from torch.nn import functional as F

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.uncertainty.aggregation import aggregateSemantic, aggregatePanoptic
from .modeling.uncertainty.postprocessing import sem_seg_postprocess_v2
from .modeling.uncertainty.prediction_modeling import entropy
from .modeling.uncertainty.tta import getTTATransforms, scaleTransform
from .utils.memory import retry_if_cuda_oom
from .utils.opticalflow import resize as resize_flow_obj


@META_ARCH_REGISTRY.register()
class MaskFormerUncertainty(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    SUPPORTED_SEMANTIC_AGGREGATION_METHODS = ['averaging', 'pixel_decoder_averaging', 'mask_dist']
    SUPPORTED_INSTANCE_AGGREGATION_METHODS = ['averaging', 'naive_mean', 'concatenate', 'pixel_decoder_averaging', 'mask_dist']
    SUPPORTED_PANOPTIC_AGGREGATION_METHODS = ['averaging', 'pixel_decoder_averaging', 'mask_dist']

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            sem_seg_head: nn.Module,
            criterion: nn.Module,
            num_queries: int,
            object_mask_threshold: float,
            overlap_threshold: float,
            metadata,
            size_divisibility: int,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            # inference
            semantic_on: bool,
            panoptic_on: bool,
            instance_on: bool,
            test_topk_per_image: int,
            mc_dropout: bool,
            mc_dropout_samples: int,
            semantic_aggregation_method: str,
            instance_aggregation_method: str,
            panoptic_aggregation_method: str,
            aggregation_overlap_threshold: float,
            aggregation_vote_threshold: float,
            tta_enabled: bool,
            tta_transforms: Tuple[str],
            num_timeseries_prev_frames: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            mc_dropout: bool, whether to collect multiple samples using MC Dropout at inference time
            tta_enabled: bool, enables test-time augmentation to generate multiple samples
            tta_transforms: tuple(str) listing transforms to apply for test-time augmentation
            num_timeseries_prev_frames: int, number of previous frames to use as samples. 0 = disable
        """
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.mc_dropout = mc_dropout
        self.mc_dropout_samples = mc_dropout_samples

        if self.mc_dropout:
            assert self.mc_dropout_samples > 0, "With MC Dropout enabled, # of MC Dropout samples (cfg.UNCERTAINTY.MC_DROPOUT_SAMPLES) should be greater than 0."

            p_value = 0
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    p_value += m.p
            if p_value <= 0:
                self._logger.warning("Model should contain at least some dropout layers with p > 0.")
        else:
            self.mc_dropout_samples = 1

        self.semantic_aggregation_method = semantic_aggregation_method
        self.instance_aggregation_method = instance_aggregation_method
        self.panoptic_aggregation_method = panoptic_aggregation_method
        self.aggregation_overlap_threshold = aggregation_overlap_threshold
        self.aggregation_vote_threshold = aggregation_vote_threshold

        assert self.semantic_aggregation_method in self.SUPPORTED_SEMANTIC_AGGREGATION_METHODS, f"Unsupported semantic aggregation method: {self.semantic_aggregation_method}"
        assert self.instance_aggregation_method in self.SUPPORTED_INSTANCE_AGGREGATION_METHODS, f"Unsupported instance aggregation method: {self.instance_aggregation_method}"
        assert self.panoptic_aggregation_method in self.SUPPORTED_PANOPTIC_AGGREGATION_METHODS, f"Unsupported panoptic aggregation method: {self.panoptic_aggregation_method}"

        if (self.panoptic_aggregation_method == 'pixel_decoder_averaging') or (
                self.instance_aggregation_method == 'pixel_decoder_averaging') or (
                self.semantic_aggregation_method == 'pixel_decoder_averaging'):
            assert self.panoptic_aggregation_method == self.instance_aggregation_method == self.semantic_aggregation_method, 'With pixel_decoder_averaging semantic/instance/panoptic must be treated the same'

        self.tta = tta_enabled
        self.tta_transforms = getTTATransforms(self.tta, tta_transforms)

        if self.panoptic_aggregation_method == 'pixel_decoder_averaging':
            for t in self.tta_transforms[0]:
                assert not isinstance(t,
                                      scaleTransform), 'Cannot use any scaling transforms with pixel_decoder_averaging'

        assert num_timeseries_prev_frames >= 0, f'Cannot have negative number of previous frames!'
        self.num_timeseries_prev_frames = num_timeseries_prev_frames
        self.timeseries_enabled = num_timeseries_prev_frames > 0

        sampling_enabled = self.tta or self.mc_dropout or self.timeseries_enabled
        self._logger.info(f'Mask2Former Uncertainty sampling is {"ENABLED" if sampling_enabled else "DISABLED"}')
        self.num_samples = 1
        if self.tta:
            self._logger.info(f'Using {len(self.tta_transforms[0])} TTA samples')
            self.num_samples *= len(self.tta_transforms[0])
        if self.mc_dropout:
            self._logger.info(f'Using {self.mc_dropout_samples} MC samples')
            self.num_samples *= self.mc_dropout_samples
        if self.timeseries_enabled:
            self._logger.info(f'Using {self.num_timeseries_prev_frames} previous frames')
            self.num_samples *= (self.num_timeseries_prev_frames + 1)
        if sampling_enabled:
            self._logger.info(f'Total # of samples: {self.num_samples}')

    def train(self, mode: bool = True):
        r"""Set the module in training mode, excluding dropout layers which are forcibly enabled to allow for MC Dropout.
         See documentation of :class:`torch.nn.Module` for further details.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        # Based on https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py
        super().train(mode)
        if self.mc_dropout:
            for module in self.modules():
                # Enable MC Dropout layers if in inference mode
                if isinstance(module, nn.Dropout) and not mode:
                    module.train(True)
        return self

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        if cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE:
            cls._logger.warning(
                'MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE is not supported in uncertainty configuration; ignoring.')

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "mc_dropout": cfg.MODEL.MASK_FORMER.UNCERTAINTY.MC_DROPOUT_ENABLED,
            "mc_dropout_samples": cfg.MODEL.MASK_FORMER.UNCERTAINTY.MC_DROPOUT_SAMPLES,
            "semantic_aggregation_method": cfg.MODEL.MASK_FORMER.UNCERTAINTY.SEMANTIC_AGGREGATION_METHOD,
            "instance_aggregation_method": cfg.MODEL.MASK_FORMER.UNCERTAINTY.INSTANCE_AGGREGATION_METHOD,
            "panoptic_aggregation_method": cfg.MODEL.MASK_FORMER.UNCERTAINTY.PANOPTIC_AGGREGATION_METHOD,
            "aggregation_overlap_threshold": cfg.MODEL.MASK_FORMER.UNCERTAINTY.AGGREGATION_OVERLAP_THRESHOLD,
            "aggregation_vote_threshold": cfg.MODEL.MASK_FORMER.UNCERTAINTY.AGGREGATION_VOTE_THRESHOLD,
            "tta_enabled": cfg.MODEL.MASK_FORMER.UNCERTAINTY.TTA_ENABLED,
            "tta_transforms": cfg.MODEL.MASK_FORMER.UNCERTAINTY.TTA_TRANSFORMS,
            "num_timeseries_prev_frames": cfg.MODEL.MASK_FORMER.UNCERTAINTY.TIMESERIES_NUM_PREV_FRAMES,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        if self.training:
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)

            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        # Inference mode
        else:

            useMultiSample = self.mc_dropout or self.tta or self.timeseries_enabled

            # MC Dropout, TTA
            if useMultiSample:
                if self.panoptic_aggregation_method == 'pixel_decoder_averaging':
                    return self.multiSamplePixelDecoderAverage(batched_inputs)
                elif self.panoptic_aggregation_method == 'mask_dist':
                    return self.multiSampleMaskDist(batched_inputs)
                else:
                    multisample_model_output = self.multiSampleForward(batched_inputs)
                    return self.multiSampleProcess(multisample_model_output)
            else:
                # Baseline approach
                return self.baselineForward(batched_inputs)

    def multiSampleMaskDist(self, batched_inputs: list):
        """
        Use Euclidean distance between masks to determine correspondence between "objects" in each sample, then use the
        mean.
        :param batched_inputs: list of inputs to model. Must be length 1.
        :return: List of length 1, with dict containing panoptic/semantic/instance segmentation alongside uncertainty information.
        """
        tta_transforms, inverse_tta_transforms = self.tta_transforms

        batch_size = len(batched_inputs)
        assert batch_size == 1, "Batch size > 1 not supported for multi-sample"

        batched_inputs = batched_inputs[0]

        input_images = [(batched_inputs["image"], None)]  # Image, optical flow transform (None if N/A)

        # Add previous images to list for processing
        # Order is oldest -> most recent in time, but not relevant for our purposes as flow already combined
        # to map directly to base image
        if self.timeseries_enabled:
            assert "prev_frame_data" in batched_inputs, 'No previous frame data found! Check dataset and mapper.'
            for prev_f in batched_inputs["prev_frame_data"]:
                input_images.append((prev_f['image'], prev_f['flow']))

        mask_pred, cls_pred, mask_cls_pred = None, None, None
        expected_entropy_mask, expected_entropy_class = 0, 0 #aleatoric
        expected_variance_mask = 0
        pixel_level_mask_sample_count = None
        num_actual_samples = 0 # Might be different from self.num_samples especially with timeseries

        for selected_image, flow in input_images:
            image = selected_image.to(self.device)

            for transform, inverse_transform in zip(tta_transforms, inverse_tta_transforms):

                imageT = transform(image)  # Transform (test-time augmentation)
                imageT = (imageT - self.pixel_mean) / self.pixel_std  # Normalize
                imageT = ImageList.from_tensors([imageT], self.size_divisibility)  # Pad if necessary
                image_size = imageT.image_sizes[0]

                imageTensor = imageT.tensor

                if flow is not None and image_size != flow.shape[-2:]:
                    flow2 = resize_flow_obj(flow, image_size)
                else:
                    flow2 = flow

                for _ in range(self.mc_dropout_samples):
                    features = self.backbone(imageTensor)
                    outputs = self.sem_seg_head(features)

                    # Masks here are a fraction of original input size
                    mask_cls_results = outputs["pred_logits"][0]  # w/ typical COCO parameters: [100, 134]
                    mask_pred_results = outputs["pred_masks"][0]  # w/ typical COCO parameters: [100, 200, 304]

                    del features
                    del outputs

                    height = batched_inputs.get("height",
                                                   image_size[0])  # Original image sizes before dataloader processing
                    width = batched_inputs.get("width", image_size[1])

                    # Mask pred results = [N, H, W]
                    # N usually 100
                    mask_pred_results = sem_seg_postprocess_v2(mask_pred_results,
                                                                pre_padding_image_size=image_size,
                                                                post_padding_image_size=imageTensor.shape[-2:],
                                                                final_output_size=(height, width),
                                                                inverse_transform=inverse_transform,
                                                                flow=flow2)
                    mask_pred_results = mask_pred_results.sigmoid()
                    not_nan_region = ~mask_pred_results.isnan().any(0)
                    num_masks = mask_pred_results.shape[0]

                    # Mask-weighted softmax distribution
                    mask_cls_pred_results = torch.einsum("qc,qhw->chw", mask_cls_results.softmax(-1), mask_pred_results)
                    mask_cls_pred_results /= mask_cls_pred_results.sum(0)  # Mask weighting means does not add to 0 -> we normalize

                    # First sample
                    if mask_pred is None:
                        mask_pred = mask_pred_results.nan_to_num(nan=0)
                        cls_pred = mask_cls_results
                        mask_cls_pred = mask_cls_pred_results.nan_to_num(nan=0)
                        pixel_level_mask_sample_count = torch.zeros((height, width), dtype=torch.int32, device=self.device)
                        pixel_level_mask_sample_count[not_nan_region] = 1
                    # Follow on samples
                    else:
                        # Get euclidean distance between masks -> pick minimum, shuffle masks + class scores to match
                        flattened_mask = mask_pred_results.nan_to_num(nan=0).reshape(num_masks, -1)
                        dist_measure = torch.cdist(flattened_mask, (mask_pred / pixel_level_mask_sample_count).reshape(num_masks, -1))
                        del flattened_mask
                        assignments = dist_measure.argmin(dim=0)
                        mask_pred_results = mask_pred_results[assignments, ...]
                        mask_cls_results = mask_cls_results[assignments, ...]
                        mask_pred += mask_pred_results.nan_to_num(nan=0)
                        cls_pred += mask_cls_results
                        mask_cls_pred += mask_cls_pred_results.nan_to_num(nan=0)
                        pixel_level_mask_sample_count[not_nan_region] += 1

                    num_actual_samples += 1
                    # Note: entropy will normalize to 1
                    # Note2: nan to num will eliminate any invalid areas (such as due to missing time series pixels)
                    # but will also force any areas with effectively 0 entropy to 0 e.g. for entropy[0,0,1,0,0,0]
                    expected_entropy_mask += entropy(mask_pred_results, dim=0).nan_to_num(nan=0)
                    expected_entropy_class += entropy(mask_cls_pred_results, dim=0).nan_to_num(nan=0)
                    expected_variance_mask += mask_pred_results.var(dim=0).nan_to_num(nan=0)

                    del mask_pred_results
                    del mask_cls_results
                    del mask_cls_pred_results
                    del not_nan_region

        if num_actual_samples != self.num_samples:
            self._logger.warning(f'Expected {self.num_samples} samples, got {num_actual_samples} instead.')

        mask_pred /= pixel_level_mask_sample_count
        cls_pred /= num_actual_samples
        mask_cls_pred /= pixel_level_mask_sample_count
        expected_entropy_mask /= pixel_level_mask_sample_count
        expected_variance_mask /= pixel_level_mask_sample_count
        expected_entropy_class /= pixel_level_mask_sample_count
        # Force any NaNs from undefined entropy (e.g. due to 0 in input) to 0
        pred_entropy_mask = entropy(mask_pred, dim=0).nan_to_num(nan=0)

        # Ommitting to save space; easy to calculate from other two
        # Note: mutual information *may* be negative by ~epsilon of float32 in class case
        # May be negative by more than that in mask case due to numerical instability
        # mutual_information_mask = pred_entropy_mask - expected_entropy_mask
        # mutual_information_class = pred_entropy_class - expected_entropy_class

        results = {}

        results["pred_entropy_mask"] = pred_entropy_mask
        results["expected_entropy_mask"] = expected_entropy_mask
        results["expected_entropy_class"] = expected_entropy_class
        results["expected_variance_mask"] = expected_variance_mask

        # r = self.semantic_inference(cls_pred, mask_pred, use_sigmoid=True)
        results["pred_entropy_class"] = entropy(mask_cls_pred, dim=0)
        results["softmax_cls_score"] = mask_cls_pred[:-1, ...].max(0).values
        results["norm_sigmoid_mask_score"] = (mask_pred / mask_pred.sum(0)).max(0).values
        results["pred_variance_mask"] = mask_pred.var(dim=0).nan_to_num(nan=0)
        if self.semantic_on:
            # Remove background
            results["sem_seg"] = mask_cls_pred[:-1, ...]

        # panoptic segmentation inference
        if self.panoptic_on:
            panoptic_seg, segments_info, is_thing_region = self.panoptic_inference(cls_pred, mask_pred, use_sigmoid=False)
            results["panoptic_seg"] = (panoptic_seg, segments_info)
            class_mask_combined_score = results["softmax_cls_score"].detach().clone()
            class_mask_combined_score[is_thing_region] = (class_mask_combined_score[is_thing_region] + results["norm_sigmoid_mask_score"][is_thing_region])/2
            results["class_mask_combined_score"] = class_mask_combined_score

        # instance segmentation inference
        if self.instance_on:
            raise NotImplementedError('Instance segmentation not supported!')

        return [results]

    def multiSamplePixelDecoderAverage(self, batched_inputs: list):

        tta_transforms, inverse_tta_transforms = self.tta_transforms

        batch_size = len(batched_inputs)
        assert batch_size == 1, "Batch size > 1 not supported for multi-sample"

        batched_inputs = batched_inputs[0]

        input_images = [(batched_inputs["image"], None)]  # Image, optical flow transform (None if N/A)

        # Add previous images to list for processing
        # Order is oldest -> most recent in time, but not relevant for our purposes as flow already combined
        # to map directly to base image
        if self.timeseries_enabled:
            assert "prev_frame_data" in batched_inputs, 'No previous frame data found! Check dataset and mapper.'
            for prev_f in batched_inputs["prev_frame_data"]:
                input_images.append((prev_f['image'], prev_f['flow']))

        mask_features, multi_scale_features = 0, []
        pixel_level_mask_sample_count, pixel_level_multi_scale_sample_count = None, None

        image_sizes = []
        tensorShapes = []

        for selected_image, flow in input_images:
            image = selected_image.to(self.device)
            image = ImageList.from_tensors([image], self.size_divisibility)
            image_size = image.image_sizes[0]
            image_sizes.append(image_size)

            if flow is not None:
                flow = flow.to_device(self.device)

            for transform, inverse_transform in zip(tta_transforms, inverse_tta_transforms):

                # We apply the transform after padding so as to allow for the transform to be reversed
                # at the intermediate feature map level in the network without the complication of handling the padding
                # e.g. with horizontal flips
                # Note that because of the padding being applied first we can't support any changes to image size e.g. scale transforms
                imageTensor = transform(image.tensor)
                imageTensor = ((imageTensor - self.pixel_mean) / self.pixel_std)
                # Since we apply mean/std after padding, we fill those padded areas with 0
                imageTensor[..., image_size[0]:, :] = 0
                imageTensor[..., image_size[1]:] = 0

                tensorShapes.append(imageTensor.shape)

                for _ in range(self.mc_dropout_samples):
                    features = self.backbone(imageTensor)
                    # Get feature maps at intermediate stage in head
                    single_sample_mask_features, single_sample_multi_scale_features = self.sem_seg_head.forward_pixeldecoder(
                        features)

                    # Undo TTA transform
                    single_sample_mask_features = inverse_transform(single_sample_mask_features)
                    single_sample_multi_scale_features = [inverse_transform(l) for l in
                                                          single_sample_multi_scale_features]

                    # Track which pixels are valid for time series data - some regions will have invalid data due to occlusions etc.
                    if pixel_level_mask_sample_count is None:
                        pixel_level_mask_sample_count = torch.zeros(single_sample_mask_features.shape[-2:],
                                                                    dtype=torch.int32, device=self.device)
                    if pixel_level_multi_scale_sample_count is None:
                        pixel_level_multi_scale_sample_count = [
                            torch.zeros(l.shape[-2:], dtype=torch.int32, device=self.device) for l in
                            single_sample_multi_scale_features]

                    # For time series apply optical flow to map previous images(s) to base image
                    if flow is not None:
                        flow_resized = resize_flow_obj(flow, single_sample_mask_features.shape[-2:])
                        single_sample_mask_features, mask = flow_resized.apply(single_sample_mask_features,
                                                                               return_valid_area=True)
                        mask = mask.squeeze()
                        single_sample_mask_features[..., ~mask] = 0
                        pixel_level_mask_sample_count[mask] += 1
                        for i in range(len(single_sample_multi_scale_features)):
                            flow_resized = resize_flow_obj(flow, single_sample_multi_scale_features[i].shape[-2:])
                            l, mask = flow_resized.apply(single_sample_multi_scale_features[i], return_valid_area=True)
                            mask = mask.squeeze()
                            l[..., ~mask] = 0
                            pixel_level_multi_scale_sample_count[i][mask] += 1
                            single_sample_multi_scale_features[i] = l
                    else:
                        pixel_level_mask_sample_count += 1
                        for l in range(len(pixel_level_multi_scale_sample_count)):
                            pixel_level_multi_scale_sample_count[l] += 1

                    mask_features += single_sample_mask_features

                    if len(multi_scale_features) == 0:
                        multi_scale_features.extend(single_sample_multi_scale_features)
                    else:
                        for i in range(len(multi_scale_features)):
                            multi_scale_features[i] += single_sample_multi_scale_features[i]

        del imageTensor
        del features
        del single_sample_mask_features
        del single_sample_multi_scale_features

        # Calculate mean
        mask_features /= pixel_level_mask_sample_count
        for i in range(len(multi_scale_features)):
            multi_scale_features[i] /= pixel_level_multi_scale_sample_count[i]

        # Verify no size discrepancies are occurring
        tensorShape = tensorShapes[0]
        for i in range(1, len(tensorShapes)):
            assert tensorShape == tensorShapes[i]

        image_size = image_sizes[0]
        for i in range(1, len(image_sizes)):
            assert image_size == image_sizes[i]

        outputs = self.sem_seg_head.forward_prediction(mask_features, multi_scale_features)

        mask_cls_results = outputs["pred_logits"][0]
        mask_pred_results = outputs["pred_masks"][0]

        del outputs

        # Resize images back to model input shape, remove padding, and resize to final pre-dataloader size
        height = batched_inputs.get("height", image_size[0])
        width = batched_inputs.get("width", image_size[1])
        mask_pred_results = sem_seg_postprocess_v2(mask_pred_results, pre_padding_image_size=image_size,
                                                   post_padding_image_size=tensorShape[-2:],
                                                   final_output_size=(height, width))

        # [0] b/c forced batch size 1
        results = {}
        r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_results, mask_pred_results)
        results["pred_entropy_class"] = entropy(r, dim=0)
        mask_pred_sigmoid = mask_pred_results.sigmoid()
        results["pred_entropy_mask"] = entropy(mask_pred_sigmoid, dim=0).nan_to_num(nan=0)
        results["softmax_cls_score"] = r[:-1, ...].max(0).values
        results["norm_sigmoid_mask_score"] = (mask_pred_sigmoid / mask_pred_sigmoid.sum(0)).max(0).values
        results["pred_variance_mask"] = mask_pred_sigmoid.var(dim=0)

        # Remove background
        r = r[:-1, ...]

        if self.semantic_on:
            results["sem_seg"] = r

        # panoptic segmentation inference
        if self.panoptic_on:
            panoptic_seg, segments_info, is_thing_region = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_results, mask_pred_results)
            results["panoptic_seg"] = (panoptic_seg, segments_info)
            class_mask_combined_score = results["softmax_cls_score"].detach().clone()
            class_mask_combined_score[is_thing_region] = (class_mask_combined_score[is_thing_region] + results["norm_sigmoid_mask_score"][is_thing_region])/2
            results["class_mask_combined_score"] = class_mask_combined_score

        # instance segmentation inference
        if self.instance_on:
            instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_results, mask_pred_results)
            results["instances"] = instance_r

        return [results]

    def multiSampleForward(self, batched_inputs: list):

        tta_transforms, inverse_tta_transforms = self.tta_transforms

        batch_size = len(batched_inputs)
        assert batch_size == 1, "Batch size > 1 not supported for multi-sample"

        multisample_model_output = []

        input_images = [(batched_inputs[0]["image"], None)]  # Image, optical flow transform (None if N/A)

        # Add previous images to list for processing
        # Order is oldest -> most recent in time, but not relevant for our purposes as flow already combined
        # to map directly to base image
        if self.timeseries_enabled:
            assert "prev_frame_data" in batched_inputs[0], 'No previous frame data found! Check dataset and mapper.'
            for prev_f in batched_inputs[0]["prev_frame_data"]:
                input_images.append((prev_f['image'], prev_f['flow']))

        for selected_image, flow in input_images:
            image = selected_image.to(self.device)

            for transform, inverse_transform in zip(tta_transforms, inverse_tta_transforms):

                imageT = transform(image)  # Transform (test-time augmentation)
                imageT = (imageT - self.pixel_mean) / self.pixel_std  # Normalize
                imageT = ImageList.from_tensors([imageT], self.size_divisibility)  # Pad if necessary
                image_size = imageT.image_sizes[0]

                imageTensor = imageT.tensor

                if flow is not None and image_size != flow.shape[-2:]:
                    flow2 = resize_flow_obj(flow, image_size)
                else:
                    flow2 = flow

                for _ in range(self.mc_dropout_samples):
                    features = self.backbone(imageTensor)
                    outputs = self.sem_seg_head(features)

                    # Note: possibility here for moving to CPU and back to GPU later as needed
                    # Masks here are a fraction of original input size
                    mask_cls_results = outputs["pred_logits"][0]  # w/ typical COCO parameters: [100, 134]
                    mask_pred_results = outputs["pred_masks"][0]  # w/ typical COCO parameters: [100, 200, 304]

                    del features
                    del outputs

                    height = batched_inputs[0].get("height",
                                                   image_size[0])  # Original image sizes before dataloader processing
                    width = batched_inputs[0].get("width", image_size[1])

                    multisample_model_output.append(
                        {'mask_pred_results': mask_pred_results, 'mask_cls_results': mask_cls_results,
                         'inverse_transform': inverse_transform, 'tensor_size_padded': imageTensor.shape[-2:],
                         'tensor_size_nopadding': image_size, 'original_image_size': (height, width), 'flow': flow2})

                    del mask_pred_results
                    del mask_cls_results

        return multisample_model_output

    def multiSampleProcess(self, multisample_model_output: list):

        results = {}

        # Note: in case of time series this may not match self.num_samples due to missing data for some frames
        num_samples = len(multisample_model_output)
        if num_samples != self.num_samples:
            self._logger.warning(f'Expected {self.num_samples} samples, got {num_samples} instead.')

        # Restore sizing and undo the TTA transformations applied earlier on input image
        # Note: we assume transformations are not sensitive to the size of the image
        for i in multisample_model_output:
            i["mask_pred_results"] = sem_seg_postprocess_v2(i["mask_pred_results"],
                                                            pre_padding_image_size=i["tensor_size_nopadding"],
                                                            post_padding_image_size=i["tensor_size_padded"],
                                                            final_output_size=i["original_image_size"],
                                                            inverse_transform=i["inverse_transform"],
                                                            flow=i["flow"])

        # Verify size consistency
        for i in range(1, num_samples):
            cur_mask = multisample_model_output[i]["mask_pred_results"].shape
            prev_mask = multisample_model_output[i - 1]["mask_pred_results"].shape
            assert (torch.as_tensor(cur_mask) == torch.as_tensor(
                prev_mask)).all(), 'Size not consistent between samples!'

        # semantic segmentation inference
        # Note that we could potentially run this under the retry_if_cuda_oom context but in practice it seems that
        # running out of memory only then occurs somewhere else. It also is so slow as to be unusable.
        semantic_and_uncert_results = aggregateSemantic(multisample_model_output,
                                                        self.semantic_aggregation_method)

        sem_seg = semantic_and_uncert_results.pop('sem_seg')
        results["softmax_cls_score"] = sem_seg[:-1, ...].max(0).values

        if self.semantic_on:
            results['sem_seg'] = sem_seg

        # Add uncertainty information - always
        results.update(semantic_and_uncert_results)

        expected_entropy_mask, expected_variance_mask = [], []
        for i in multisample_model_output:
            mask_res = i["mask_pred_results"]
            # Use nan to num to force numerically unstable values = nan to 0 as they should be
            expected_entropy_mask.append(entropy(mask_res.sigmoid(), dim=0).nan_to_num(nan=0))
            expected_variance_mask.append(mask_res.sigmoid().var(dim=0).nan_to_num(nan=0))
        results['expected_entropy_mask'] = torch.stack(expected_entropy_mask).mean(0)
        results["expected_variance_mask"] = torch.stack(expected_variance_mask).mean(0)


        if self.panoptic_on:
            results['panoptic_seg'] = aggregatePanoptic(multisample_model_output,
                                                        self.sem_seg_head.num_classes, self.metadata,
                                                        self.overlap_threshold,
                                                        self.aggregation_overlap_threshold,
                                                        self.aggregation_vote_threshold,
                                                        self.panoptic_aggregation_method)

        if self.instance_on:
            raise NotImplementedError('Instance segmentation not implemented yet for multi-sample')

        # Model expects format as list length batch size
        return [results]

    def baselineForward(self, batched_inputs):

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        image_sizes = images.image_sizes
        imageTensor = images.tensor

        features = self.backbone(imageTensor)
        outputs = self.sem_seg_head(features)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(imageTensor.shape[-2], imageTensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs
        del features

        results = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            results.append({})

            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, image_size, height, width
            )
            mask_cls_result = mask_cls_result.to(mask_pred_result)

            r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)

            mask_pred_sigmoid = mask_pred_result.sigmoid()
            results[-1]["pred_entropy_class"] = entropy(r, dim=0)
            results[-1]["pred_entropy_mask"] = entropy(mask_pred_sigmoid, dim=0).nan_to_num(nan=0)
            results[-1]["softmax_cls_score"] = r[:-1, ...].max(0).values
            results[-1]["norm_sigmoid_mask_score"] = (mask_pred_sigmoid / mask_pred_sigmoid.sum(0)).max(0).values
            results[-1]["pred_variance_mask"] = mask_pred_sigmoid.var(dim=0)
            # Remove background class
            r = r[:-1, ...]

            if self.semantic_on:
                results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_seg, segments_info, is_thing_region = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                results[-1]["panoptic_seg"] = (panoptic_seg, segments_info)
                class_mask_combined_score = results[-1]["softmax_cls_score"].detach().clone()
                class_mask_combined_score[is_thing_region] = (class_mask_combined_score[is_thing_region] + results[-1]["norm_sigmoid_mask_score"][is_thing_region])/2
                results[-1]["class_mask_combined_score"] = class_mask_combined_score

            # instance segmentation inference
            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                results[-1]["instances"] = instance_r

        return results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred, use_sigmoid=True):
        mask_cls = F.softmax(mask_cls, dim=-1)
        if use_sigmoid:
            mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        semseg /= semseg.sum(0) # Normalize s.t. adds to 1
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred, use_sigmoid=True):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        if use_sigmoid:
            mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        is_thing_region = torch.zeros((h, w), dtype=torch.bool, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info, is_thing_region
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                matching_prob_mask = (cur_mask_ids == k)
                matching_original_mask = (cur_masks[k] >= 0.5)
                mask = matching_prob_mask & matching_original_mask
                mask_area = matching_prob_mask.sum().item()
                original_area = matching_original_mask.sum().item()

                if isthing:
                    is_thing_region[matching_prob_mask] = True

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                            "score": cur_prob_masks[k, mask].mean().item()
                        }
                    )

            return panoptic_seg, segments_info, is_thing_region

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries,
                                                                                                     1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab.item() in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
                result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
