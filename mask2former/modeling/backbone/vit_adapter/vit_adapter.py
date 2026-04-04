# From https://github.com/czczup/ViT-Adapter/blob/main/segmentation/mmseg_custom/models/backbones/vit_adapter.py
# modified by Michael Smith, McGill University
# This is from the ViT-Adapter codebase, which in turn copied (for some reason) with modifications the ViT from TIMM.
# We reproduce that here for consistency. Ideally, would refactor to use the original TIMM codebase.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs
from .vit import TIMMVisionTransformer

_logger = logging.getLogger(__name__)


class ViTAdapter(TIMMVisionTransformer):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0., interaction_indexes=None, with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True,
                 use_extra_extractor=True, with_cp=False, freeze_vit=False, *args, **kwargs):

        super().__init__(num_heads=num_heads,
                         with_cp=with_cp, *args, **kwargs)
        if freeze_vit:
            for param in self.parameters():
                param.requires_grad = False

        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


@BACKBONE_REGISTRY.register()
class D2ViTAdapter(ViTAdapter, Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__(
            pretrain_size=cfg.MODEL.VITADAPTER.PRETRAIN_SIZE,
            img_size=cfg.INPUT.IMAGE_SIZE,
            patch_size=cfg.MODEL.VITADAPTER.PATCH_SIZE,
            num_heads=cfg.MODEL.VITADAPTER.NUM_HEADS,
            embed_dim=cfg.MODEL.VITADAPTER.EMBED_DIM,
            depth=cfg.MODEL.VITADAPTER.DEPTH,
            mlp_ratio=cfg.MODEL.VITADAPTER.MLP_RATIO,
            drop_path_rate=cfg.MODEL.VITADAPTER.DROP_PATH_RATE,
            conv_inplane=cfg.MODEL.VITADAPTER.CONV_INPLANE,
            n_points=cfg.MODEL.VITADAPTER.N_POINTS,
            deform_num_heads=cfg.MODEL.VITADAPTER.DEFORM_NUM_HEADS,
            cffn_ratio=cfg.MODEL.VITADAPTER.CFFN_RATIO,
            with_cffn=cfg.MODEL.VITADAPTER.WITH_CFFN,
            deform_ratio=cfg.MODEL.VITADAPTER.DEFORM_RATIO,
            interaction_indexes=cfg.MODEL.VITADAPTER.INTERACTION_INDEXES,
            window_attn=cfg.MODEL.VITADAPTER.WINDOW_ATTN,
            window_size=cfg.MODEL.VITADAPTER.WINDOW_SIZE,
            freeze_vit=cfg.MODEL.VITADAPTER.FREEZE_VIT
        )
        if cfg.MODEL.BACKBONE.FREEZE_AT > 0:
            _logger.warning(
                "cfg.MODEL.BACKBONE.FREEZE_AT is not applicable to ViT-Adapter models. Consider cfg.MODEL.VITADAPTER.FREEZE_VIT instead.")

        self._out_features = cfg.MODEL.VITADAPTER.OUT_FEATURES

        self._out_feature_strides = {
            0: 4,
            1: 8,
            2: 16,
            3: 32,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
                x.dim() == 4
        ), f"D2ViTAdapter takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for k in self._out_features:
            outputs[k] = y[k]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.embed_dim, stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return self.patch_size
