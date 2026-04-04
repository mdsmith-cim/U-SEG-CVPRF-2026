# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling
from . import utils

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
from .data.dataset_mappers.timeseries_panoptic_mapper import TimeseriesPanopticMapper

# models
from .maskformer_model import MaskFormer
from .maskformer_uncertainty_model import MaskFormerUncertainty
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
from .evaluation.save_results import saveResults
from .evaluation.visualization import saveVisualizations
from .evaluation.PQ_perimage import PanopticQualityPerImage
from .evaluation.semsegv2 import SemSegEvaluatorV2
from .evaluation.PQv3 import PanopticQualityV3