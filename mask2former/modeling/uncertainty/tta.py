# Test time augmentation related functions
import logging
from typing import Any, Dict, List, Optional, Union

from torchvision.transforms.v2 import Transform, RandomHorizontalFlip, Compose, InterpolationMode
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2._utils import query_size

logger = logging.getLogger(__name__)


class noOpTransform(Transform):
    """
    Passes data through unchanged.
    """

    def forward(self, inputs: Any) -> Any:
        return inputs


class scaleTransform(Transform):
    """Based on torchvision.transforms.v2.ScaleJitter. Applies a fixed scaling factor to any input image.

    If the input is a :class:`torch.Tensor` or a ``TVTensor`` (e.g. :class:`~torchvision.tv_tensors.Image`,
    :class:`~torchvision.tv_tensors.Video`, :class:`~torchvision.tv_tensors.BoundingBoxes` etc.)
    it can have arbitrary number of leading batch dimensions. For example,
    the image can have ``[..., C, H, W]`` shape. A bounding box can have ``[..., 4]`` shape.

    Args:
        scale (float): Scaling factor to apply to image, as a function of input image size. Default: 1.0.
        interpolation (InterpolationMode, optional): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True`` (default): will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The default value changed from ``None`` to ``True`` in
            v0.17, for the PIL and Tensor backends to be consistent.
    """

    def __init__(self, scale: float = 1.0, interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
                 antialias: Optional[bool] = True):
        super().__init__()
        self.scale = scale
        self.interpolation = interpolation
        self.antialias = antialias

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_height, orig_width = query_size(flat_inputs)

        new_width = int(orig_width * self.scale)
        new_height = int(orig_height * self.scale)

        return dict(size=(new_height, new_width))

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.resize, inpt, size=params["size"], interpolation=self.interpolation,
                                 antialias=self.antialias)


# noinspection PyTypeChecker
def getTTATransforms(enabled: bool = True, transform_types: tuple = ('horizontalFlip',)) -> tuple:
    """
    Get all test-time transformations to apply. Note: we expect that image sizes will not be changed.
    @return: Tuple of transform to apply. [0]: input transforms. [1]: output transforms to restore spatial alignment
    """
    # By a sort of convention we always have the first transform be nothing
    transforms = [noOpTransform()]
    reversed_transforms = [noOpTransform()]

    if enabled:

        # VALUES paper uses Gaussian + Horizontal flipping
        # Note that Gaussian noise as used in VALUES does not work very well with Mask2Former however w.r.t PQ metric
        # Likely as unlike VALUES we do not train with Gaussian noise
        # gaussianTransform = Compose([ToDtype(torch.float32, scale=True),
        #                              GaussianNoise(clip=True),
        #                              ToDtype(torch.uint8, scale=True)])

        # **** Transforms Used ****
        # Scale Jitter - choice of scale based on Simple Copy-Paste Is a Strong Data Augmentation Method for Instance Segmentation CVPR 2021
        # Horizontal flips - as used in ValUES: A Framework for Systematic Validation of Uncertainty Estimation in Semantic Segmentation ICLR 2024
        # Confirmed both to work well with Mask2Former w.r.t. PQ metric

        if 'horizontalFlip' in transform_types:
            transforms.append(RandomHorizontalFlip(p=1))
            reversed_transforms.append(RandomHorizontalFlip(p=1))
        if 'scale' in transform_types:
            transforms.extend([scaleTransform(scale=0.8), scaleTransform(scale=1.25)])
            # Note: we don't invert the scale transforms as the output is already being resized as part of normal operation
            reversed_transforms.extend([noOpTransform(), noOpTransform()])
        if ('horizontalFlip' in transform_types) and ('scale' in transform_types):
            transforms.extend([Compose([RandomHorizontalFlip(p=1), scaleTransform(scale=0.8)]),
                               Compose([RandomHorizontalFlip(p=1), scaleTransform(scale=1.25)])])
            reversed_transforms.extend([RandomHorizontalFlip(p=1), RandomHorizontalFlip(p=1)])

        if len(transforms) == 1:
            raise ValueError(
                f'Expected at least one of "horizontalFlip" or "scale" as transform type. Got {transform_types}')

    return transforms, reversed_transforms
