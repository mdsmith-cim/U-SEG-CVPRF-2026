import torch
from mask2former.utils.memory import retry_if_cuda_oom

def entropy(pk, dim=0):
    '''
    Adapted from scipy https://github.com/scipy/scipy/blob/v1.9.1/scipy/stats/_entropy.py#L17-L88
    Simplified for my uses, and also adapted to use function calls on tensors for CUDA memory efficiency in PyTorch.
    This routine will normalize `pk` if it doesn't sum to 1.
    Entropy calculated as ``S = -sum(pk * log(pk), axis=axis)``.
    :param pk: array_like
        Defines the (discrete) distribution. Along each axis-slice of ``pk``,
        element ``i`` is the  (possibly unnormalized) probability of event
        ``i``.
    :param dim: int, optional
        The axis along which the entropy is calculated. Default is 0.
    :return: S : {float, array_like}
        The calculated entropy.
    '''
    pk = 1.0 * pk / pk.sum(dim=dim, keepdim=True)
    return (-pk * pk.log()).sum(dim=dim)


def calculate_uncertainty(softmax_preds: torch.Tensor):
    """
    Calculate uncertainty metrics from softmax predictions.

    This function computes various uncertainty metrics such as predictive entropy,
    aleatoric uncertainty, and epistemic uncertainty from the given softmax predictions.

    Args:
        softmax_preds (torch.Tensor): A tensor of softmax predictions with shape
                                      (num_samples, num_classes, height, width).

    Returns:
        dict: A dictionary containing the following keys:
            - "pred_entropy": Predictive entropy.
            - "aleatoric_uncertainty": Aleatoric uncertainty.
            - "epistemic_uncertainty": Epistemic uncertainty.
    """
    uncertainty_dict = {}

    assert softmax_preds.ndim == 4, "Softmax predictions must have shape (num_samples, num_classes, height, width)"

    original_device = softmax_preds.device

    mean_softmax = retry_if_cuda_oom(torch.nanmean, device=original_device)(softmax_preds, dim=0)
    predictive_entropy = retry_if_cuda_oom(entropy, device=original_device)(mean_softmax, dim=0)

    expected_entropy = retry_if_cuda_oom(entropy, device=original_device)(softmax_preds, dim=1).nanmean(dim=0)

    # Mutual information can be calculated from pred - expected
    uncertainty_dict["pred_entropy_class"] = predictive_entropy
    uncertainty_dict["expected_entropy_class"] = expected_entropy

    return uncertainty_dict
