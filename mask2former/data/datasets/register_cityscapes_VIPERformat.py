# Registers the existence of the Cityscapes (VIPER formatted) dataset in COCO format to detectron2/mask2former
# Based off of register_VIPER_panoptic_annos_semseg
# Much code is duplicated
import json
import logging
import os
from copy import deepcopy
from functools import cache
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager
from .register_VIPER_panoptic_annos_semseg import meta

logger = logging.getLogger(__name__)

def _convert_category_id(segment_info):
    if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
        segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
            segment_info["category_id"]
        ]
        segment_info["isthing"] = True
    else:
        segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
            segment_info["category_id"]
        ]
        segment_info["isthing"] = False
    return segment_info


def getPrevFrameData(image_fn: str, image_dir: str, fw_flow_root: str, bw_flow_root: str):

    # VIPER provides annotations for all images, but it is too much to use so we only use the "standard" option of
    # annotations for every 10 images. Thus, we have 10 previous images to use (including the base image itself, 9 without)
    # for timeseries sample generation
    parent_folder, bn = os.path.split(image_fn)
    name, ext = os.path.splitext(bn)
    city, seq, frame_id = name.split('_')
    seq_int, frame_id_int = int(seq), int(frame_id)
    prev_data = []
    try:
        for prev_frame_id in range(frame_id_int - 19, frame_id_int + 1):
            prev_name = os.path.join(parent_folder, f'{city}_{seq}_{prev_frame_id:06}')
            prev_fn = os.path.join(image_dir, prev_name + ext)
            assert PathManager.exists(prev_fn), f'Sequence image {prev_fn} does not exist!'
            # We do not provide forward/backward flow that would go beyond this set of images and touch on
            # other annotated images, and denote such files as None even though they likely do exist outside the start
            # and end of sequences
            prev_bw_flow = None
            if prev_frame_id != frame_id_int - 19:
                prev_bw_flow =  os.path.join(bw_flow_root, prev_name + '.npz')
                assert PathManager.exists(prev_bw_flow), f'Backward flow {prev_bw_flow} does not exist!'
            prev_fw_flow = None
            if prev_frame_id != frame_id_int:
                prev_fw_flow = os.path.join(fw_flow_root, prev_name + '.npz')

                assert PathManager.exists(prev_fw_flow), f'Forward flow {prev_fw_flow} does not exist!'
            prev_data.append({
                'file_name': prev_fn,
                'bw_flow_file_name': prev_bw_flow,
                'fw_flow_file_name': prev_fw_flow,
            })
    # Some sequences are missing data in the original dataset; authors state unable to fix per e-mail conversation
    # So we just ignore those sequences entirely (about two dozen)
    except AssertionError as e:
        logger.debug(f'Caught error {e} for {image_fn}; skipping previous data use.')
        return []

    # Final format is list of dict, including base image itself
    # List is in ascending order
    # First entry is furthest back possible; last entry is current image (needed for backward flow to image before)
    return prev_data

# Cache function call as some parts of Mask2Former/detectron will make repeat calls to get the dataset
@cache
def load_coco_panoptic_json(json_file, image_dir, gt_dir, semseg_dir, fw_flow_root, bw_flow_root, ood_root):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        semseg_dir: path to directory containing semantic segmentation annotations.
        fw_flow_root: path to directory containing forward optical flow data.
        bw_flow_root: path to directory containing backward optical flow data.
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
        ood_root: path to directory containing out of distribution RGB images.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    image_id_mapping = {i['id']: i for i in json_info["images"]}

    timeseries_enabled = (fw_flow_root is not None) and (bw_flow_root is not None)
    logger.info(f'Loading VIPER dataset; timeseries enabled: {timeseries_enabled}')
    if timeseries_enabled:
        logger.info(f'Loading and verifying timeseries data; this may take a while...')

    ret = []
    prev_data_amount = 0
    for ann in json_info["annotations"]:
        image_id = ann["image_id"]
        image_fn = image_id_mapping[image_id]["file_name"]
        image_file = os.path.join(image_dir, image_fn)
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x) for x in ann["segments_info"]]

        image_data = {
                "file_name": image_file,
                "is_ood": False,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        if timeseries_enabled:
            prev_data = getPrevFrameData(image_fn, image_dir, fw_flow_root, bw_flow_root)
            image_data["prev_frame_data"] = prev_data
            prev_data_amount += len(prev_data)

        ret.append(image_data)

        if ood_root is not None:
            image_data = deepcopy(image_data)
            image_file = os.path.join(ood_root, image_fn)
            assert os.path.exists(image_file), f'OOD image {image_file} does not exist!'
            image_data["is_ood"] = True
            image_data["file_name"] = image_file
            ret.append(image_data)

    assert len(ret), f"No images found in {image_dir}!"
    if timeseries_enabled:
        assert prev_data_amount > 0, f'Timeseries enabled but optical flow or RGB images seem to be missing?'
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret

_PREDEFINED_SPLITS_VIPER_PANOPTIC = {
    "cityscapes_VIPER_format_val_panoptic_with_semseg_time": (
        "cityscapes_VIPERformat/panoptic_val", # Panoptic png directory
        "cityscapes_VIPERformat/annotations/panoptic_coco_format_val.json", # Annotation JSON file - panoptic
        "cityscapes_VIPERformat/annotations/instances_coco_format_val.json", # Annotation JSON file - instances
        # Semantic annotations that are converted from panoptic annotations
        # See datasets/create_VIPER_dataset.py
        "cityscapes_VIPERformat/semseg_val",
        "cityscapes_VIPERformat/val", # Image directory
        "cityscapes_VIPERformat/optflow_extract/val/flow",  # Forward optical flow
        "cityscapes_VIPERformat/optflow_extract/val/flowbw",  # Backward optical flow
        None,
    ),
    "cityscapes_VIPER_format_val_panoptic_with_semseg_time_with_ood": (
        "cityscapes_VIPERformat/panoptic_val",  # Panoptic png directory
        "cityscapes_VIPERformat/annotations/panoptic_coco_format_val.json",  # Annotation JSON file - panoptic
        "cityscapes_VIPERformat/annotations/instances_coco_format_val.json",  # Annotation JSON file - instances
        # Semantic annotations that are converted from panoptic annotations
        # See datasets/create_VIPER_dataset.py
        "cityscapes_VIPERformat/semseg_val",
        "cityscapes_VIPERformat/val",  # Image directory
        "cityscapes_VIPERformat/optflow_extract/val/flow",  # Forward optical flow
        "cityscapes_VIPERformat/optflow_extract/val/flowbw",  # Backward optical flow
        "cityscapes_VIPERformat/val_ood", # OOD Image directory
    ),
    "cityscapes_VIPER_format_val_panoptic_with_semseg": (
        "cityscapes_VIPERformat/panoptic_val",
        "cityscapes_VIPERformat/annotations/panoptic_coco_format_val.json",
        "cityscapes_VIPERformat/annotations/instances_coco_format_val.json",
        "cityscapes_VIPERformat/semseg_val",
        "cityscapes_VIPERformat/val",
        None,
        None,
        None,
    ),
    "cityscapes_VIPER_format_val_panoptic_with_semseg_with_ood": (
        "cityscapes_VIPERformat/panoptic_val",
        "cityscapes_VIPERformat/annotations/panoptic_coco_format_val.json",
        "cityscapes_VIPERformat/annotations/instances_coco_format_val.json",
        "cityscapes_VIPERformat/semseg_val",
        "cityscapes_VIPERformat/val",
        None,
        None,
        "cityscapes_VIPERformat/val_ood", # OOD Image directory
    ),
    "cityscapes_VIPER_format_train_panoptic_with_semseg": (
        "cityscapes_VIPERformat/panoptic_train",
        "cityscapes_VIPERformat/annotations/panoptic_coco_format_train.json",
        "cityscapes_VIPERformat/annotations/instances_coco_format_train.json",
        "cityscapes_VIPERformat/semseg_train",
        "cityscapes_VIPERformat/train",
        None,
        None,
        None,
    ),
    "cityscapes_VIPER_format_test_panoptic_with_semseg": (
        "cityscapes_VIPERformat/panoptic_test",
        "cityscapes_VIPERformat/annotations/panoptic_coco_format_test.json",
        "cityscapes_VIPERformat/annotations/instances_coco_format_test.json",
        "cityscapes_VIPERformat/semseg_test",
        "cityscapes_VIPERformat/test",
        None,
        None,
        None,
    ),
}

def register_viper_panoptic_annos_sem_seg(name, image_root, panoptic_root, panoptic_json, sem_seg_root,
                                          fw_flow_root, bw_flow_root, instances_json, ood_root):
    # Register dataset for use
    DatasetCatalog.register(name,
                            lambda: load_coco_panoptic_json(panoptic_json, image_root, panoptic_root, sem_seg_root,
                                                            fw_flow_root, bw_flow_root, ood_root))
    # Register related metadata
    # (get function creates the metadata, then we set variables)
    MetadataCatalog.get(name).set(evaluator_type="coco_panoptic_seg", json_file=instances_json, image_root=image_root,
                                  ignore_label=255, label_divisor=1000, sem_seg_root=sem_seg_root, ood_img_root=ood_root,
                                  panoptic_root=panoptic_root,
                                  panoptic_json=panoptic_json, **meta)


def register_all_VIPER_panoptic_annos_sem_seg(root):
    for (
            prefix,
            (panoptic_root, panoptic_json, instances_json, semantic_root, image_root, fw_flow_root, bw_flow_root, ood_root),
    ) in _PREDEFINED_SPLITS_VIPER_PANOPTIC.items():
        register_viper_panoptic_annos_sem_seg(
            prefix,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            os.path.join(root, fw_flow_root) if fw_flow_root else None,
            os.path.join(root, bw_flow_root) if bw_flow_root else None,
            os.path.join(root, instances_json),
            os.path.join(root, ood_root) if ood_root else None,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_VIPER_panoptic_annos_sem_seg(_root)
