# Based on register_coco_panoptic_annos_semseg.py and https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/builtin.py
import os

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from .register_coco_panoptic_annos_semseg import register_coco_panoptic_annos_sem_seg, get_metadata, load_coco_panoptic_json

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

_PREDEFINED_SPLITS_COCO_INSTANCE_DEBUG = {
    "coco_2017_val_debugsingle": (
        "coco/val2017",
        "coco/annotations/single_image_instance_debug_val2017.json",
    ),
    "coco_2017_val_debugmulti": (
        "coco/val2017",
        "coco/annotations/multi_image_instance_debug_val2017.json",
    ),
}

# Same format as _PREDEFINED_SPLITS_COCO_PANOPTIC in register_coco_panoptic_annos_semseg.py except
# first entry is name of already existing instance metadata entry
_PREDEFINED_SPLITS_COCO_PANOPTIC_DEBUG = {
    "coco_2017_val_panoptic_debugsingle": (
        "coco_2017_val",
        "coco/panoptic_val2017",
        "coco/annotations/single_image_panoptic_debug_val2017",
        "coco/panoptic_semseg_train2017",
    ),
    "coco_2017_val_panoptic_debugmulti": (
        "coco_2017_val",
        "coco/panoptic_val2017",
        "coco/annotations/multi_image_panoptic_debug_val2017.json",
        "coco/panoptic_semseg_val2017",
    ),
}

# Instance registration
for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_INSTANCE_DEBUG.items():
    register_coco_instances(
        key,
        _get_builtin_metadata("coco"),
        os.path.join(_root, json_file) if "://" not in json_file else json_file,
        os.path.join(_root, image_root),
    )

# Panoptic registration
for (prefix, (instance_name, panoptic_root, panoptic_json, semantic_root)) in _PREDEFINED_SPLITS_COCO_PANOPTIC_DEBUG.items():
    instances_meta = MetadataCatalog.get(instance_name)
    image_root, instances_json = instances_meta.image_root, instances_meta.json_file

    metadata = get_metadata()
    panoptic_root = os.path.join(_root, panoptic_root)
    panoptic_json = os.path.join(_root, panoptic_json)
    sem_seg_root = os.path.join(_root, semantic_root)

    DatasetCatalog.register(
        prefix,
        lambda a=panoptic_json, b=image_root, c=panoptic_root, d=sem_seg_root: load_coco_panoptic_json(a, b, c, d, metadata),
    )
    MetadataCatalog.get(prefix).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )

