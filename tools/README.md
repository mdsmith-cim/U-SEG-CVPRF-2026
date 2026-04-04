For tools that come with Mask2Former, see the [original documentation](https://github.com/facebookresearch/Mask2Former/blob/main/tools/README.md).

The ones we introduce are:
- `print-timm-model-info.py`: Prints information about a specified timm (pytorch-image-models) model. Useful to figure out what is what, as there are a lot of models and a lot of variants.
- `strip-backbone-weights.py`: Used to remove backbone weights from a model. Useful for when we want to start training with a pretrained backbone from one source(e.g. a pretrained DINO backbone on ImageNet via timm) and a pretrained head from another Mask2Former model.