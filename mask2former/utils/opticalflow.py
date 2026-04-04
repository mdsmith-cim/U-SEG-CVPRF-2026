import oflibpytorch as of
import numpy as np
import torch.nn.functional as f
from typing import Union, Tuple
from oflibpytorch.flow_class import FlowAlias, Flow
import torch
from oflibpytorch.utils import get_valid_vecs

def loadOpticalFlow(file_fw, file_bw, device=None):
    """
    Load optical flow data from disk and collate into oflibpytorch object that takes occlusion into account.
    Note: both flows are reversed and multiplied by -1 to change reference frame, avoiding expensive interpolation computation. Thus,
    the backward flow is what is actually used for the forward flow and vice versa.
    :param file_fw: Forward flow filename.
    :param file_bw: Backward flow filename.
    :param device: torch device to load data to.
    :return:
    """
    # Read from file
    opt_flow_data = np.load(file_bw, allow_pickle=True, fix_imports=False)
    u, v = opt_flow_data['u'].astype(np.float32), opt_flow_data['v'].astype(np.float32)

    max_img_dim = np.array((u.shape, v.shape)).max()

    # Some values are junk; mask out those areas
    mask = np.isfinite(u) & np.isfinite(v)
    # Apply initial mask if needed from past warps
    # Convert to needed format for oflibpytorch
    # Re: * -1: we load the reverse flow and * by -1 to change the reference such that we don't have to use expensive interpolation when applying the flow
    opt_flow_stacked = np.stack((u, v), axis=-1) * -1
    np.nan_to_num(opt_flow_stacked, copy=False)
    # Turns out we also need to handle cases where a few pixels have finite values equal to ~max IEEE 32 bit floating point numbers
    # If you try and warp those -> NaN
    # So we remove anything > 10x maximum image dimension in pixels
    ok_size = np.abs(opt_flow_stacked) < 10 * max_img_dim
    ok_size = ok_size[..., 0] & ok_size[..., 1]
    opt_flow_stacked[~ok_size, :] = 0
    mask &= ok_size

    # If we weren't loading the reverse flow and * by -1, we would need to use ref='s'
    flow = of.Flow(opt_flow_stacked, ref='t', mask=mask, device=device)

    # Now same for backwards flow
    opt_flow_data_reverse = np.load(file_fw, allow_pickle=True, fix_imports=False)
    u_reverse, v_reverse = opt_flow_data_reverse['u'].astype(np.float32), opt_flow_data_reverse['v'].astype(np.float32)
    mask_reverse = np.isfinite(u_reverse) & np.isfinite(v_reverse)

    opt_flow_stacked_reverse = np.stack((u_reverse,v_reverse), axis=-1) * -1
    np.nan_to_num(opt_flow_stacked_reverse, copy=False)

    # Should be same as before but just in case...
    max_img_dim_rev = np.array((u_reverse.shape, v_reverse.shape)).max()

    ok_size_rev = np.abs(opt_flow_stacked_reverse) < 10 * max_img_dim_rev
    ok_size_rev = ok_size_rev[..., 0] & ok_size_rev[..., 1]
    opt_flow_stacked_reverse[~ok_size_rev, :] = 0
    mask_reverse &= ok_size_rev

    flow_reverse = of.Flow(opt_flow_stacked_reverse, ref='t', mask=mask_reverse, device=device)

    # Occlusion handling when ground truth available for both forward and backward flow
    # Narayanan Sundaram, Thomas Brox, and Kurt Keutzer. Dense point trajectories by gpu-accelerated large displacement optical flow. In ECCV, 2010.
    combined_flow = flow_reverse.combine_with(flow, mode=3)
    occluded = ((combined_flow.vecs**2).sum(axis=1) > 0.01 * ((flow.vecs**2).sum(axis=1) + (flow_reverse.vecs**2).sum(axis=1)) + 0.5)

    # Mask out occluded areas - the sole purpose of loading the backwards flow
    flow.mask &= ~occluded

    return flow

# Redefining some oflibpytorch function to take explicit pixel sizes rather than scale
def resize_flow(flow: Union[np.ndarray, torch.Tensor], new_size: Union[int, list, tuple]) -> torch.Tensor:
    """Resize a flow field numpy array or torch tensor, scaling the flow vectors values accordingly.
    Adapted from oflibpytorch.utils.resize_flow

    The output flow field is differentiable with respect to the input flow field, if given as a torch tensor.

    :param flow: Flow field as a numpy array or torch tensor, shape :math:`(2, H, W)`, :math:`(H, W, 2)`,
        :math:`(N, 2, H, W)`, or :math:`(N, H, W, 2)`
    :param new_size: Size used for resizing, options:

        - Integer specifying size applied both vertically and horizontally
        - List or tuple of shape :math:`(2)` with values ``[vertical size, horizontal size]``
    :return: Resized flow field as a torch tensor, shape :math:`(2, H, W)` or :math:`(N, 2, H, W)`, depending on input
    """

    # Check validity
    valid_flow = get_valid_vecs(flow, error_string="Error resizing flow: ")
    if isinstance(new_size, (int)):
        new_size = [new_size, new_size]
    elif isinstance(new_size, (tuple, list)):
        if len(new_size) != 2:
            raise ValueError("Error resizing flow: New size {} must have a length of 2".format(type(new_size)))
        if not all(isinstance(item, int) for item in new_size):
            raise ValueError("Error resizing flow: New size {} items must be integers or floats".format(type(new_size)))
    else:
        raise TypeError("Error resizing flow: "
                        "Scale must be an integer, or list, or tuple of integers or floats")
    if any(s <= 0 for s in new_size):
        raise ValueError("Error resizing flow: New size values must be larger than 0")

    # Resize and adjust values
    resized = f.interpolate(valid_flow, size=new_size, mode='bilinear', align_corners=False)
    scale = tuple(np.array(new_size) / np.array(valid_flow.shape[-2:]))
    resized[:, 0] *= scale[1]
    resized[:, 1] *= scale[0]

    # Get rid of first dim if input was only 3-dimensional
    if len(flow.shape) == 3:
        resized = resized.squeeze(0)

    return resized

def resize(flow_obj, new_size: Union[int, list, tuple]) -> FlowAlias:
    """Resize a flow field, scaling the flow vectors values :attr:`vecs` accordingly.
    Adapted from oflibpytorch.flow_class.Flow.resize

    The output flow vectors are differentiable with respect to the input flow vectors.

    :param flow_obj: oflibpytorch Flow object to resize
    :param size: Size to resize to, options:

        - Integer applied both vertically and horizontally
        - List or tuple of shape :math:`(2)` with values ``[vertical size, horizontal size]``
    :return: New flow object resized as desired
    """

    resized_flow = resize_flow(flow_obj._vecs, new_size)
    if isinstance(new_size, int):
        scale = [new_size, new_size]
    mask_to_resize = flow_obj._mask.float().unsqueeze(1)
    resized_mask = f.interpolate(mask_to_resize, size=new_size,
                                 mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    # Note: scale can be used with no validity checks because already validated in resize_flows
    return Flow(resized_flow, flow_obj._ref, torch.round(resized_mask), device=flow_obj._device)