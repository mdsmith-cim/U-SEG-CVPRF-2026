# Based on detectron2.modeling.postprocessing.sem_seg_postprocess
from torch.nn import functional as F
from typing import Optional
from oflibpytorch import Flow
import torch
def sem_seg_postprocess_v2(result, pre_padding_image_size, post_padding_image_size, final_output_size, mode="bilinear", inverse_transform=None, flow: Optional[Flow] = None):
    """
        Return semantic segmentation predictions in the original resolution and undo input TTA transform.

        The input images are often resized before being processed by the model, in addition to having padding added such
        that the image is divisible by maximum network stride.
        This function resizes the images back to original resolution and removes padding.

        Args:
            result (Tensor): A tensor of shape (B, C, H, W) with the first two dimensions being optional.
                B is the batch size, C is the number of classes, and H, W are the height and width of the prediction.
            pre_padding_image_size (tuple): (height, width) - image size that model took as input, but before padding for divisibility.
            post_padding_image_size (tuple): (height, width) - image size that model took as input, after padding for divisibility.
            final_output_size (tuple): (height, width) - the desired output resolution.
            mode: Passed directly to torch.nn.functional.interpolate.
            inverse_transform (callable): applied after resizing to pre_padding_image_size to reverse any input
            transforms.
            flow: oflibpytorch flow object to apply to image to map to base image. Set to None if not applicable.

        Returns:
            semantic segmentation prediction (Tensor): A tensor of the shape (B, C, output_height, output_width)
            resized to original resolution and with padding removed. First two dimensions are optional and only present
            if present in 'result' input tensor.
        """

    # Resize first to pre-padding size
    input_dims = result.dim()
    squeeze_dims = tuple()
    if input_dims < 3:
        result = result.expand(1, 1, -1, -1)
        squeeze_dims = (0, 1)
    elif input_dims < 4:
        result = result.expand(1, -1, -1, -1)
        squeeze_dims = (0,)

    # Interpolate expects Batch x Channels X Height x Width
    result = F.interpolate(result, size=post_padding_image_size, mode=mode)
    # Remove padding
    result = result[..., : pre_padding_image_size[0], : pre_padding_image_size[1]]
    # Apply inverse transform
    if inverse_transform is not None:
        result = inverse_transform(result)
    # Apply optical flow
    if flow is not None:
        flow = flow.to_device(result.device)
        result, mask = flow.apply(result, return_valid_area=True)
        result[..., ~mask.squeeze()] = torch.nan

    # Resize to original input size before any processing e.g. in dataloader
    result = F.interpolate(result, size=final_output_size, mode=mode).squeeze(dim=squeeze_dims)
    return result