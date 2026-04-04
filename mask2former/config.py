# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    cfg.INPUT.TEST_DATASET_MAPPER_NAME = "DatasetMapper"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Uncertainty evaluation
    cfg.MODEL.MASK_FORMER.UNCERTAINTY = CN()
    cfg.MODEL.MASK_FORMER.UNCERTAINTY.MC_DROPOUT_ENABLED = False
    cfg.MODEL.MASK_FORMER.UNCERTAINTY.MC_DROPOUT_SAMPLES = 0
    cfg.MODEL.MASK_FORMER.UNCERTAINTY.SEMANTIC_AGGREGATION_METHOD = 'averaging'
    cfg.MODEL.MASK_FORMER.UNCERTAINTY.INSTANCE_AGGREGATION_METHOD = 'averaging'
    cfg.MODEL.MASK_FORMER.UNCERTAINTY.PANOPTIC_AGGREGATION_METHOD = 'averaging'
    cfg.MODEL.MASK_FORMER.UNCERTAINTY.AGGREGATION_OVERLAP_THRESHOLD = 0.5
    cfg.MODEL.MASK_FORMER.UNCERTAINTY.AGGREGATION_VOTE_THRESHOLD = 0.0 # Range: [0,1], percentage of samples
    cfg.MODEL.MASK_FORMER.UNCERTAINTY.TTA_ENABLED = False
    cfg.MODEL.MASK_FORMER.UNCERTAINTY.TTA_TRANSFORMS = tuple() # expect tuple with entries 'horizontalFlip' and/or 'scale'
    cfg.MODEL.MASK_FORMER.UNCERTAINTY.TIMESERIES_NUM_PREV_FRAMES = 0 # 0 is disabled

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # timm backbone adapter
    cfg.MODEL.TIMMMODEL = CN()
    cfg.MODEL.TIMMMODEL.MODEL_NAME = 'resnet101'
    cfg.MODEL.TIMMMODEL.PRETRAINED = True
    cfg.MODEL.TIMMMODEL.DROP_RATE = None
    cfg.MODEL.TIMMMODEL.DROP_PATH_RATE = None
    cfg.MODEL.TIMMMODEL.FREEZE_MODEL = False
    cfg.MODEL.TIMMMODEL.THAW_SELECTED = [] # Selected modules to leave not frozen if FREEZE_MODEL = True
    cfg.MODEL.TIMMMODEL.USE_IMAGE_SIZE = False
    cfg.MODEL.TIMMMODEL.OUT_FEATURES = [1, 2, 3, 4]
    cfg.MODEL.TIMMMODEL.EXTRA_ARGS = CN()
    cfg.MODEL.TIMMMODEL.EXTRA_ARGS.set_new_allowed(True)

    # ViT-Adapter backbone
    cfg.MODEL.VITADAPTER = CN()
    cfg.MODEL.VITADAPTER.PRETRAIN_SIZE = 592
    cfg.MODEL.VITADAPTER.PATCH_SIZE = 16
    cfg.MODEL.VITADAPTER.EMBED_DIM = 768
    cfg.MODEL.VITADAPTER.DEPTH = 12
    cfg.MODEL.VITADAPTER.NUM_HEADS = 12
    cfg.MODEL.VITADAPTER.MLP_RATIO = 4
    cfg.MODEL.VITADAPTER.DROP_PATH_RATE = 0.3
    cfg.MODEL.VITADAPTER.CONV_INPLANE = 64
    cfg.MODEL.VITADAPTER.N_POINTS = 4
    cfg.MODEL.VITADAPTER.DEFORM_NUM_HEADS = 12
    cfg.MODEL.VITADAPTER.CFFN_RATIO = 0.25
    cfg.MODEL.VITADAPTER.WITH_CFFN = True
    cfg.MODEL.VITADAPTER.DEFORM_RATIO = 0.5
    cfg.MODEL.VITADAPTER.INTERACTION_INDEXES = [[0, 2], [3, 5], [6, 8], [9, 11]]
    cfg.MODEL.VITADAPTER.WINDOW_ATTN = [True, True, False, True, True, False, True, True, False, True, True, False]
    cfg.MODEL.VITADAPTER.WINDOW_SIZE = [14, 14, None, 14, 14, None, 14, 14, None, 14, 14, None]
    cfg.MODEL.VITADAPTER.FREEZE_VIT = True
    cfg.MODEL.VITADAPTER.OUT_FEATURES = [0, 1, 2, 3]

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0
    cfg.INPUT.ROTATION_AUG = False
    cfg.INPUT.GAUSSIAN_AUG = False

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parameter `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # Manual evaluator spec
    cfg.TEST.EVALUATORS = tuple()  # Default - none, use old if-else logic
    # For writing visualizations to disk via saveVisualizations
    cfg.TEST.VISUALIZER = CN()
    cfg.TEST.VISUALIZER.FORMAT = 'pdf' # File type to write. Specified as extension given to matplotlib.
    cfg.TEST.VISUALIZER.UNCERT_LIMITS = CN() # For visualization: min and max to show for various uncertainty/confidence measures
    # These are some rough general purpose values
    # Should be modified in custom config file
    # Format is (MIN, MAX)
    cfg.TEST.VISUALIZER.UNCERT_LIMITS.pred_entropy_mask = (0, 3)
    cfg.TEST.VISUALIZER.UNCERT_LIMITS.pred_entropy_class = (0, 3)
    cfg.TEST.VISUALIZER.UNCERT_LIMITS.expected_entropy_mask = (0, 3)
    cfg.TEST.VISUALIZER.UNCERT_LIMITS.expected_entropy_class = (0, 3)
    cfg.TEST.VISUALIZER.UNCERT_LIMITS.expected_variance_mask = (0, 0.15)
    cfg.TEST.VISUALIZER.UNCERT_LIMITS.pred_variance_mask = (0, 0.15)
    cfg.TEST.VISUALIZER.UNCERT_LIMITS.softmax_cls_score = (0, 1)
    cfg.TEST.VISUALIZER.UNCERT_LIMITS.norm_sigmoid_mask_score = (0, 1)
    cfg.TEST.VISUALIZER.UNCERT_LIMITS.class_mask_combined_score = (0, 1)
    cfg.TEST.VISUALIZER.UNCERT_LIMITS.set_new_allowed(True)