import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from mask2former.utils.opticalflow import loadOpticalFlow
from mask2former.utils.opticalflow import resize as resize_flow_obj

"""
This file contains the dataset mapping that's applied when timeseries data is used.
Based on detectron2 default dataset mapper.
"""


class TimeseriesPanopticMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This particular version is based on the default mapper, and does the following:
    1. Load image data from "file_name"
    2. Load image data and optical flow for frames prior to the base frame
    3. Converts data to torch tensors
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        recompute_boxes: bool = False,
        num_prev_frames: int = 2,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
            num_prev_frames: number of frames to load image and optical flow data for (including base)
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.recompute_boxes        = recompute_boxes
        self.num_prev_frames        = num_prev_frames
        if self.num_prev_frames <= 0:
            raise ValueError('With timeseries panoptic mapper expect number of previous frames to be > 0')
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[TimeseriesPanopticMapper] Augmentations used in {mode}: {augmentations}")
        logger.info(f'[TimeseriesPanopticMapper] Using {self.num_prev_frames} previous frames')

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # Force inference mode; otherwise risk certain transforms like cropping being enabled
        assert not is_train, 'Cannot use timeseries mapper in training mode!'
        augs = utils.build_augmentation(cfg, is_train)
        recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "num_prev_frames": cfg.MODEL.MASK_FORMER.UNCERTAINTY.TIMESERIES_NUM_PREV_FRAMES,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            raise NotImplementedError('Precomputed proposals not supported with timeseries.')
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # With timeseries data we load images of prior frames as well as optical flow data
        assert 'prev_frame_data' in dataset_dict, 'With timeseries loader dataset is expected to have "prev_frame_data" field.'
        prev_frame_data = dataset_dict.pop("prev_frame_data")

        # Load image and flow for previous frames
        # Minimum is 1 previous frame
        loaded_files = []

        # Some images may be missing previous data due to errors in the dataset; in that case we have an emtpy list and skip
        if len(prev_frame_data) > 0:
            # Go through each previous frame, load image and flow
            for f in range(1, self.num_prev_frames + 1):
                # Get data of previous frame and following one
                frame = prev_frame_data[-f - 1]
                next_frame = prev_frame_data[-f]

                # Load RGB image and apply transforms as necessary
                frame_fn = frame["file_name"]
                frame_img = utils.read_image(frame_fn, format=self.image_format)
                aug_frame_img = T.AugInput(frame_img)
                transforms = self.augmentations(aug_frame_img)
                frame_img = aug_frame_img.image
                assert image_shape == frame_img.shape[:2], f'Image size {frame_img.shape[:-1]} for {frame_fn} does not match expected size {image_shape}!'
                frame_img = torch.as_tensor(np.ascontiguousarray(frame_img.transpose(2, 0, 1)))

                # Load optical flow
                # Forward flow uses forward flow on disk from current frame
                # Backward flow uses backward flow on disk from next frame
                # We multiply by -1 in loadOpticalFlow() to change the reference frame from source to target and thus
                # the backward flow data is actually used as forward
                opt_flow_filename_fw = frame["fw_flow_file_name"]
                opt_flow_filename_bw = next_frame["bw_flow_file_name"]
                flow = loadOpticalFlow(opt_flow_filename_fw, opt_flow_filename_bw, torch.device('cpu'))

                # Resize flow to match image
                flow = resize_flow_obj(flow, image_shape)

                # Combine flow with last flow to generate flow that directly maps to base frame
                if len(loaded_files) > 0:
                    next_flow = loaded_files[-1]['flow']
                    flow = flow.combine(next_flow, mode=3)

                assert flow.shape[1:] == image_shape, f'Flow shape {flow.shape} does not match image shape {image_shape}!'

                loaded_files.append({'flow': flow, 'image': frame_img})

        # Previous frame order matches original: oldest -> newest
        # i.e. with base frame f, [-1] will be f-1, [-2] will be f-2 etc.
        # Flows are already combined such that they map directly to base frame
        dataset_dict["prev_frame_data"] = loaded_files

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict
