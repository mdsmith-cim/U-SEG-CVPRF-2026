import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES
from detectron2.utils.file_io import PathManager
from .register_cityscapes_timeseries import prev_files_json

"""
This file contains functions to register a timeseries VIPER dataset, using Cityscapes labels, to the DatasetCatalog.
Modified from register_cityscapes_timeseries, with much duplicated code due to being in a rush
"""

logger = logging.getLogger(__name__)

def get_cityscapes_panoptic_files(image_dir, gt_dir, fw_flow_dir, bw_flow_dir, json_info, timeseries_enabled: bool):
    files = []
    flow_suffix = '_leftImg8bit.npz'

    if timeseries_enabled:
        logger.info("Verifying sequence information for all cities. This may take a while...")
    else:
        logger.info('Skipping processing sequence information; time series not enabled.')

    image_dict = {}
    for image_info in json_info["images"]:
        image_filename = image_info['file_name']
        image_id = image_info['id']
        image_root, image_extension = os.path.splitext(image_filename)
        city_str, seq_str, frame_str, suffix = image_root.split('_')
        seq_int, frame_int = int(seq_str), int(frame_str)
        image_file = os.path.join(image_dir, city_str, image_filename)
        prev_files = []

        if timeseries_enabled:
            try:
                for prev_frame_int in range(frame_int - 9, frame_int + 1):
                    prev_img_file = os.path.join(image_dir, city_str, f'{city_str}_{seq_str}_{prev_frame_int:0>6}_{suffix}{image_extension}')
                    assert PathManager.exists(prev_img_file), f'Sequence image {prev_img_file} does not exist'

                    bw_flow_file = None
                    # No backward flow for first frame in a sequence - we set to None
                    if prev_frame_int != frame_int - 9:
                        bw_flow_file = os.path.join(bw_flow_dir, city_str,
                                                    f'{city_str}_{seq_str}_{prev_frame_int:0>6}{flow_suffix}')
                        assert PathManager.exists(bw_flow_file), f'Backward flow file {bw_flow_file} does not exist!'

                    fw_flow_file = None
                    if prev_frame_int != frame_int:
                        fw_flow_file = os.path.join(fw_flow_dir, city_str,
                                                    f'{city_str}_{seq_str}_{prev_frame_int:0>6}{flow_suffix}')
                        assert PathManager.exists(fw_flow_file), f'Forward flow file {fw_flow_file} does not exist!'
                    prev_files.append((prev_img_file, bw_flow_file, fw_flow_file))
            except AssertionError as e:
                # VIPER has some missing corrupted data; in that case, drop all previous frame data
                logger.debug(f'Encountered issue with previous frame data for {image_file}: {e}')
                prev_files = []

        image_dict[image_id] = (image_file, prev_files)

    for ann in json_info["annotations"]:
        try:
            image_file, prev_files = image_dict.get(ann["image_id"], None)
        except KeyError as e:
            raise FileNotFoundError(f"No image file for ID {ann['image_id']} (file {ann['file_name']}) found!") from e

        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = ann["segments_info"]

        assert PathManager.exists(image_file), f'Image file {image_file} does not exist!'
        assert PathManager.exists(label_file), f'Label file {label_file} does not exist!'

        files.append((image_file, prev_files, label_file, segments_info))

    assert len(files), "No images found in {}".format(image_dir)
    return files

def load_cityscapes_panoptic(image_dir, gt_dir, gt_json, fw_flow_dir, bw_flow_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train".
        gt_json (str): path to the json file. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train.json".
        meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id"
            and "stuff_dataset_id_to_contiguous_id" to map category ids to
            contiguous ids for training.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        return segment_info

    timeseries_enabled = False
    if (fw_flow_dir is not None) and (bw_flow_dir is not None):
        timeseries_enabled = True

    assert os.path.exists(
        gt_json
    ), "Please run `python cityscapesscripts/preparation/createPanopticImgs.py` to generate label files."  # noqa
    with open(gt_json) as f:
        json_info = json.load(f)
    files = get_cityscapes_panoptic_files(image_dir, gt_dir, fw_flow_dir, bw_flow_dir, json_info, timeseries_enabled)
    ret = []
    for image_file, prev_files, label_file, segments_info in files:
        sem_label_file = os.path.splitext(image_file.replace("leftImg8bit", "gtFine"))[0] + "_labelTrainIds.png"
        segments_info = [_convert_category_id(x, meta) for x in segments_info]

        ret_data = {
            "file_name": image_file,
            "image_id": "_".join(
                os.path.splitext(os.path.basename(image_file))[0].split("_")[:3]
            ),
            "sem_seg_file_name": sem_label_file,
            "pan_seg_file_name": label_file,
            "segments_info": segments_info,
        }
        if timeseries_enabled:
            ret_data["prev_frame_data"] = prev_files_json(prev_files)
        ret.append(ret_data)

    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    assert PathManager.isfile(
        ret[0]["pan_seg_file_name"]
    ), "Please generate panoptic annotation with python cityscapesscripts/preparation/createPanopticImgs.py"  # noqa
    return ret


CITYSCAPES_SPLITS = {
    # With timeseries
    "viper_cityscapes_format_panoptic_val_time": (
        "VIPER_cityscapesformat/leftImg8bit/val",
        "VIPER_cityscapesformat/gtFine/cityscapes_panoptic_val",
        "VIPER_cityscapesformat/gtFine/cityscapes_panoptic_val.json",
        "VIPER_cityscapesformat/optflow_fw/val",  # Forward optical flow
        "VIPER_cityscapesformat/optflow_bw/val",  # Backward optical flow
    ),
    # Standard split (except JPG instead of png for RGB)
    "viper_cityscapes_format_panoptic_train": (
        "VIPER_cityscapesformat/leftImg8bit/train",
        "VIPER_cityscapesformat/gtFine/cityscapes_panoptic_train",
        "VIPER_cityscapesformat/gtFine/cityscapes_panoptic_train.json",
        None,
        None,
    ),
    "viper_cityscapes_format_panoptic_val": (
        "VIPER_cityscapesformat/leftImg8bit/val",
        "VIPER_cityscapesformat/gtFine/cityscapes_panoptic_val",
        "VIPER_cityscapesformat/gtFine/cityscapes_panoptic_val.json",
        None,
        None,
    ),
}


def register_all_cityscapes_panoptic(root):
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]
    stuff_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # There are three types of ids in cityscapes panoptic segmentation:
    # (1) category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the classifier
    # (2) instance id: this id is used to differentiate different instances from
    #   the same category. For "stuff" classes, the instance id is always 0; for
    #   "thing" classes, the instance id starts from 1 and 0 is reserved for
    #   ignored instances (e.g. crowd annotation).
    # (3) panoptic id: this is the compact id that encode both category and
    #   instance id by: category_id * 1000 + instance_id.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in CITYSCAPES_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (image_dir, gt_dir, gt_json, fw_flow_folder, bw_flow_folder) in CITYSCAPES_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)
        fw_flow_folder = os.path.join(root, fw_flow_folder) if fw_flow_folder is not None else None
        bw_flow_folder = os.path.join(root, bw_flow_folder) if bw_flow_folder is not None else None

        DatasetCatalog.register(key, lambda a=image_dir, b=gt_dir, c=gt_json, d=fw_flow_folder, e=bw_flow_folder: load_cityscapes_panoptic(a, b, c, d, e, meta))
        MetadataCatalog.get(key).set(
            panoptic_root=gt_dir,
            image_root=image_dir,
            panoptic_json=gt_json,
            gt_dir=gt_dir.replace("cityscapes_panoptic_", ""),
            evaluator_type="cityscapes_panoptic_seg",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )


_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
register_all_cityscapes_panoptic(_root)
