# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set
from functools import partial

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    saveResults,
    TimeseriesPanopticMapper,
    PanopticQualityPerImage,
    SemSegEvaluatorV2,
    PanopticQualityV3,
    saveVisualizations
)

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        """

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        evaluator_list = []
        # Partial allows delayed calling of functions - particularily useful for some cases where instantiating
        # the evaluator requires optional libraries
        supported_evaluators = {"SemSegEvaluator": partial(SemSegEvaluator, dataset_name=dataset_name,distributed=True, output_dir=output_folder),
                                "COCOEvaluator": partial(COCOEvaluator, dataset_name=dataset_name, output_dir=output_folder),
                                "CityscapesInstanceEvaluator": partial(CityscapesInstanceEvaluator, dataset_name=dataset_name),
                                "COCOPanopticEvaluator": partial(COCOPanopticEvaluator, dataset_name=dataset_name, output_dir=output_folder),
                                "InstanceSegEvaluator": partial(InstanceSegEvaluator, dataset_name=dataset_name, output_dir=output_folder),
                                "CityscapesSemSegEvaluator": partial(CityscapesSemSegEvaluator, dataset_name=dataset_name),
                                "LVISEvaluator": partial(LVISEvaluator, dataset_name=dataset_name, output_dir=output_folder),
                                "saveResults": partial(saveResults, output_dir=output_folder),
                                "PanopticQualityPerImage": partial(PanopticQualityPerImage, dataset_name=dataset_name, output_dir=output_folder),
                                "PanopticQualityV3": partial(PanopticQualityV3, dataset_name=dataset_name, output_dir=output_folder, device=cfg.MODEL.DEVICE),
                                "SemSegEvaluatorV2": partial(SemSegEvaluatorV2, dataset_name=dataset_name, output_dir=output_folder, device=cfg.MODEL.DEVICE),
                                "saveVisualizations": partial(saveVisualizations, dataset_name=dataset_name, output_dir=output_folder, vis_cfg=cfg.TEST.VISUALIZER),}

        # Default behaviour - use hacky if-else logic inherited from mask2former/detectron that uses the special
        # metadata "evaluator_type" associated with each builtin dataset.
        if len(cfg.TEST.EVALUATORS) == 0:

            evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
            # semantic segmentation
            if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
                evaluator_list.append(supported_evaluators["SemSegEvaluator"]())
            # instance segmentation
            if evaluator_type == "coco":
                evaluator_list.append(supported_evaluators["COCOEvaluator"]())
            # panoptic segmentation
            if evaluator_type in [
                "coco_panoptic_seg",
                "ade20k_panoptic_seg",
                "cityscapes_panoptic_seg",
                "mapillary_vistas_panoptic_seg",
            ]:
                if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                    evaluator_list.append(supported_evaluators["COCOPanopticEvaluator"]())
            # COCO
            if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                evaluator_list.append(supported_evaluators["COCOEvaluator"]())
            if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                evaluator_list.append(supported_evaluators['SemSegEvaluator']())
            # Mapillary Vistas
            if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                evaluator_list.append(supported_evaluators['InstanceSegEvaluator']())
            if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                evaluator_list.append(supported_evaluators['SemSegEvaluator']())
            # Cityscapes
            if evaluator_type == "cityscapes_instance":
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(supported_evaluators['CityscapesInstanceEvaluator']())
            if evaluator_type == "cityscapes_sem_seg":
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(supported_evaluators['CityscapesSemSegEvaluator']())
            if evaluator_type == "cityscapes_panoptic_seg":
                if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                    assert (
                        torch.cuda.device_count() > comm.get_rank()
                    ), "CityscapesEvaluator currently do not work with multiple machines."
                    evaluator_list.append(supported_evaluators['CityscapesSemSegEvaluator']())
                if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                    assert (
                        torch.cuda.device_count() > comm.get_rank()
                    ), "CityscapesEvaluator currently do not work with multiple machines."
                    evaluator_list.append(supported_evaluators['CityscapesInstanceEvaluator']())
            # ADE20K
            if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                evaluator_list.append(supported_evaluators['InstanceSegEvaluator']())
            # LVIS
            if evaluator_type == "lvis":
                evaluator_list.append(supported_evaluators['LVISEvaluator']())

        else:
            for ev in cfg.TEST.EVALUATORS:
                evaluator_list.append(supported_evaluators[ev]())

        if len(evaluator_list) == 0:
            raise NotImplementedError(f"No Evaluator for the dataset {dataset_name} found!")
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """

        # Default case
        if cfg.INPUT.TEST_DATASET_MAPPER_NAME == "DatasetMapper":
            mapper = None
        elif cfg.INPUT.TEST_DATASET_MAPPER_NAME == "timeseries_panoptic":
            mapper = TimeseriesPanopticMapper(cfg, False)
        else:
            mapper = None
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    # This test-time augmentation is different from that used in uncertainty estimation
    # This is semantic segmentation only and provides the mean prediction when feeding the network horizontally flipped images
    # Goal here is to improve final prediction metric score, nothing more
    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
