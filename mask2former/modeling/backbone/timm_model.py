# Michael Smith
# McGill University
# Allows for models from timm to be used as backbones
from typing import Mapping, Any

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from timm import create_model
from timm.utils.model import freeze, unfreeze
import logging
_logger = logging.getLogger(__name__)

@BACKBONE_REGISTRY.register()
class D2TimmModel(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        model_name = cfg.MODEL.TIMMMODEL.MODEL_NAME
        self.pretrained = cfg.MODEL.TIMMMODEL.PRETRAINED
        DROP_RATE = cfg.MODEL.TIMMMODEL.DROP_RATE
        DROP_PATH_RATE = cfg.MODEL.TIMMMODEL.DROP_PATH_RATE
        FREEZE_MODEL = cfg.MODEL.TIMMMODEL.FREEZE_MODEL
        THAW_SELECTED = cfg.MODEL.TIMMMODEL.THAW_SELECTED
        if not FREEZE_MODEL and len(THAW_SELECTED) > 0:
            raise ValueError("Can't specify modules to thaw while not freezing any!")
        USE_IMAGE_SIZE = cfg.MODEL.TIMMMODEL.USE_IMAGE_SIZE # Provide image size to model. Needed for some especially ViT-type ones
        IMG_SIZE = None
        if USE_IMAGE_SIZE:
            IMG_SIZE = cfg.INPUT.IMAGE_SIZE
        EXTRA_ARGS = cfg.MODEL.TIMMMODEL.EXTRA_ARGS
        if cfg.MODEL.BACKBONE.FREEZE_AT > 0:
            _logger.warning("cfg.MODEL.BACKBONE.FREEZE_AT is not applicable to timm models. Consider cfg.MODEL.TIMMMODEL.FREEZE_MODEL instead.")
        self._out_features = cfg.MODEL.TIMMMODEL.OUT_FEATURES
        self.model = create_model(model_name=model_name, pretrained=self.pretrained, drop_rate=DROP_RATE, drop_path_rate=DROP_PATH_RATE, features_only=True, out_indices=self._out_features, img_size=IMG_SIZE, **EXTRA_ARGS)
        if FREEZE_MODEL:
            freeze(self.model)
            if len(THAW_SELECTED) > 0:
                unfreeze(self.model, THAW_SELECTED)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"Timm model takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = self.model(x)
        for i, k in enumerate(self._out_features):
                outputs[k] = y[i]
        return outputs

    def output_shape(self):
        return {k: ShapeSpec(channels=self.model.feature_info.channels()[i], stride=self.model.feature_info.reduction()[i]) for i, k in enumerate(self._out_features)}

    @property
    def size_divisibility(self):
        return 0

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.pretrained:
            _logger.info('Note: backbone model loading is occuring while MODEL.TIMMMODEL.PRETRAINED is set! Make sure this is intended behaviour and that weights are not being overwritten.')
        return super()._load_from_state_dict(state_dict=state_dict, prefix=prefix, local_metadata=local_metadata, strict=strict, missing_keys=missing_keys, unexpected_keys=unexpected_keys, error_msgs=error_msgs)