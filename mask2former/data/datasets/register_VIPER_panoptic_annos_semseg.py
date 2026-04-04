# Registers the existence of the VIPER dataset in COCO format to detectron2/mask2former
# Based off of register_coco_panoptic_annos_semseg
import json
import logging
import os
from functools import cache
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)

# Colors are taken from COCO panoptic api
# To create:
# import requests
# coco_categories = requests.get('https://raw.githubusercontent.com/cocodataset/panopticapi/refs/heads/master/panoptic_coco_categories.json').json()
# VIPER_CATEGORIES = []
# for i, c in enumerate(coco_format_class_list): # From VIPER JSON 'categories'
#     copied_c = c.copy()
#     copied_c.pop('supercategory')
#     copied_c['color'] = coco_categories[i]['color']
#     VIPER_CATEGORIES.append(copied_c)

def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    # Note: categories are expected to be sorted - things first, then stuff
    thing_classes = [k["name"] for k in VIPER_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in VIPER_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in VIPER_CATEGORIES]
    stuff_colors = [k["color"] for k in VIPER_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. We follow the detectron2 COCO setup and maintain the contiguous mapping
    #   giving us two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    #   Note that the above only applies to panoptic, the semantic segmentation PNGs use the contiguous IDs directly
    #   It is also expected that the stuff IDs encompass the entire dataset, while things consist of a subset and are the first
    #   classes in the category list
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(VIPER_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta

VIPER_CATEGORIES = [{'id': 1, 'name': 'trafficlight', 'isthing': 1, 'color': [220, 20, 60]},
                    {'id': 2, 'name': 'firehydrant', 'isthing': 1, 'color': [119, 11, 32]},
                    {'id': 3, 'name': 'chair', 'isthing': 1, 'color': [0, 0, 142]},
                    {'id': 4, 'name': 'trashcan', 'isthing': 1, 'color': [0, 0, 230]},
                    {'id': 5, 'name': 'person', 'isthing': 1, 'color': [106, 0, 228]},
                    {'id': 6, 'name': 'motorcycle', 'isthing': 1, 'color': [0, 60, 100]},
                    {'id': 7, 'name': 'car', 'isthing': 1, 'color': [0, 80, 100]},
                    {'id': 8, 'name': 'van', 'isthing': 1, 'color': [0, 0, 70]},
                    {'id': 9, 'name': 'bus', 'isthing': 1, 'color': [0, 0, 192]},
                    {'id': 10, 'name': 'truck', 'isthing': 1, 'color': [250, 170, 30]},
                    {'id': 11, 'name': 'sky', 'isthing': 0, 'color': [100, 170, 30]},
                    {'id': 12, 'name': 'road', 'isthing': 0, 'color': [220, 220, 0]},
                    {'id': 13, 'name': 'sidewalk', 'isthing': 0, 'color': [175, 116, 175]},
                    {'id': 14, 'name': 'terrain', 'isthing': 0, 'color': [250, 0, 30]},
                    {'id': 15, 'name': 'tree', 'isthing': 0, 'color': [165, 42, 42]},
                    {'id': 16, 'name': 'vegetation', 'isthing': 0, 'color': [255, 77, 255]},
                    {'id': 17, 'name': 'building', 'isthing': 0, 'color': [0, 226, 252]},
                    {'id': 18, 'name': 'infrastructure', 'isthing': 0, 'color': [182, 182, 255]},
                    {'id': 19, 'name': 'fence', 'isthing': 0, 'color': [0, 82, 0]},
                    {'id': 20, 'name': 'billboard', 'isthing': 0, 'color': [120, 166, 157]},
                    {'id': 21, 'name': 'trafficsign', 'isthing': 0, 'color': [110, 76, 0]},
                    {'id': 22, 'name': 'mobilebarrier', 'isthing': 0, 'color': [174, 57, 255]},
                    {'id': 23, 'name': 'trash', 'isthing': 0, 'color': [199, 100, 0]}]

meta = get_metadata()

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
    seq, frame_id = name.split('_')
    seq_int, frame_id_int = int(seq), int(frame_id)
    prev_data = []
    try:
        for prev_frame_id in range(frame_id_int - 9, frame_id_int + 1):
            prev_name = os.path.join(parent_folder, f'{seq}_{prev_frame_id:05}')
            prev_fn = os.path.join(image_dir, prev_name + ext)
            assert PathManager.exists(prev_fn), f'Sequence image {prev_fn} does not exist!'
            # We do not provide forward/backward flow that would go beyond this set of images and touch on
            # other annotated images, and denote such files as None even though they likely do exist outside the start
            # and end of sequences
            prev_bw_flow = None
            if prev_frame_id != frame_id_int - 9:
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
def load_coco_panoptic_json(json_file, image_dir, gt_dir, semseg_dir, fw_flow_root, bw_flow_root):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        semseg_dir: path to directory containing semantic segmentation annotations.
        fw_flow_root: path to directory containing forward optical flow data.
        bw_flow_root: path to directory containing backward optical flow data.
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
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
                "image_id": int(image_id),
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        if timeseries_enabled:
            prev_data = getPrevFrameData(image_fn, image_dir, fw_flow_root, bw_flow_root)
            image_data["prev_frame_data"] = prev_data
            prev_data_amount += len(prev_data)

        ret.append(image_data)

    assert len(ret), f"No images found in {image_dir}!"
    if timeseries_enabled:
        assert prev_data_amount > 0, f'Timeseries enabled but optical flow or RGB images seem to be missing?'
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret

_PREDEFINED_SPLITS_VIPER_PANOPTIC = {
    "viper_train_panoptic_with_semseg": (
        # Panoptic png directory
        "VIPER/panoptic_train",
        # Annotation JSON file - panoptic
        "VIPER/annotations/panoptic_coco_format_train.json",
        # Annotation JSON file - instances
        "VIPER/annotations/instances_coco_format_train.json",
        # Semantic annotations that are converted from panoptic annotations
        # See datasets/create_VIPER_dataset.py
        "VIPER/semseg_train",
        # Image directory
        "VIPER/train",
        None, # Forward optical flow
        None # Backward optical flow
    ),
    "viper_val_panoptic_with_semseg": (
        "VIPER/panoptic_val",
        "VIPER/annotations/panoptic_coco_format_val.json",
        "VIPER/annotations/instances_coco_format_val.json",
        "VIPER/semseg_val",
        "VIPER/val",
        None,
        None
    ),
    "viper_val_panoptic_with_semseg_and_time": (
        "VIPER/panoptic_val",
        "VIPER/annotations/panoptic_coco_format_val.json",
        "VIPER/annotations/instances_coco_format_val.json",
        "VIPER/semseg_val",
        "VIPER/val",
        "VIPER/optflow_extract/val/flow",  # Forward optical flow
        "VIPER/optflow_extract/val/flowbw",  # Backward optical flow
    ),
}

def register_viper_panoptic_annos_sem_seg(name, image_root, panoptic_root, panoptic_json, sem_seg_root,
                                          fw_flow_root, bw_flow_root, instances_json):
    # Register dataset for use
    DatasetCatalog.register(name,
                            lambda: load_coco_panoptic_json(panoptic_json, image_root, panoptic_root, sem_seg_root,
                                                            fw_flow_root, bw_flow_root))
    # Register related metadata
    # (get function creates the metadata, then we set variables)
    MetadataCatalog.get(name).set(evaluator_type="coco_panoptic_seg", json_file=instances_json, image_root=image_root,
                                  ignore_label=255, label_divisor=1000, sem_seg_root=sem_seg_root,
                                  panoptic_root=panoptic_root,
                                  panoptic_json=panoptic_json, **meta)


def register_all_VIPER_panoptic_annos_sem_seg(root):
    for (
            prefix,
            (panoptic_root, panoptic_json, instances_json, semantic_root, image_root, fw_flow_root, bw_flow_root),
    ) in _PREDEFINED_SPLITS_VIPER_PANOPTIC.items():
        register_viper_panoptic_annos_sem_seg(
            prefix,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            os.path.join(root, fw_flow_root) if fw_flow_root else None,
            os.path.join(root, bw_flow_root) if bw_flow_root else None,
            os.path.join(root, instances_json)
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_VIPER_panoptic_annos_sem_seg(_root)
