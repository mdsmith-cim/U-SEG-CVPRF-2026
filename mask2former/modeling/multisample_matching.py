# From mask2former/matcher.py
# Modified by Michael Smith, McGill University, to do matching between the outputs of two models/samples
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Computes matches between two models/samples with mask and logits following the linear sum assignment problem
aka this is effectively the hungarian algorithm
As of 2024-12- this is not used due to being extremely extremely slow
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.amp import autocast
from torch.nn.functional import kl_div, binary_cross_entropy_with_logits

@torch.jit.script
def get_cost(out_prob: torch.Tensor, out_prob2: torch.Tensor, out_mask: torch.Tensor, out_mask2: torch.Tensor, cost_mask: float, cost_class: float):
    """Compute the cost matrix for the assignment problem"""
    num_queries = out_prob.shape[0]
    num_queries2 = out_prob2.shape[0]
    cost = torch.zeros((num_queries, num_queries2), device=out_prob.device, dtype=out_prob.dtype)
    for i in range(num_queries):
        for j in range(num_queries2):
            # Do KL-divergence classification cost and cross entropy mask cost
            cost[i, j] = (cost_class * kl_div(out_prob[i], out_prob2[j], reduction='sum')) + (cost_mask * binary_cross_entropy_with_logits(out_mask[i], out_mask2[j], reduction='sum'))
    return cost


class HungarianMatcher(nn.Module):
    """This class computes an assignment between two samples of predictions from the network
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the cross entropy loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask

        assert cost_class != 0 or cost_mask != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, outputs2):
        """More memory-friendly matching"""
        num_queries = outputs["unprocessed_logits"].shape[0]
        num_queries2 = outputs2["unprocessed_logits"].shape[0]

        out_prob = outputs["unprocessed_logits"].softmax(-1)  # [num_queries, num_classes]
        out_prob2 = outputs2["unprocessed_logits"].softmax(-1)  # [num_queries2, num_classes]

        # Get relevant masks and sample with select # of points
        out_mask = outputs['unprocessed_masks'] # [num_queries, H, W] various random logit-type values
        out_mask2 = outputs2['unprocessed_masks'] # [num_queries2, H, W]

        with autocast('cuda', enabled=False):
            out_mask = out_mask.float()
            out_mask2 = out_mask2.float().sigmoid()
            cost = get_cost(out_prob, out_prob2, out_mask, out_mask2, self.cost_mask, self.cost_class)

        # Final cost matrix
        assignment = linear_sum_assignment(cost.cpu())

        return torch.as_tensor(assignment[0], dtype=torch.int64), torch.as_tensor(assignment[1], dtype=torch.int64)


    @torch.no_grad()
    def forward(self, outputs, outputs2):
        """Performs the matching between outputs and outputs2

        Params:
            outputs: This is a dict that contains at least these entries:
                 "unprocessed_logits": Tensor of dim [num_queries, num_classes] with the classification logits
                 "unprocessed_masks": Tensor of dim [num_queries, H_pred, W_pred] with the predicted masks

            outputs2: This is a dict with the same format as above i.e. it contains at least these entries:
                 "unprocessed_logits": Tensor of dim [num_queries, num_classes] with the classification logits
                 "unprocessed_masks": Tensor of dim [num_queries, H_pred, W_pred] with the predicted masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the first predictions (in order)
                - index_j is the indices of the corresponding second predictions (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries1, num_queries2)
        """
        return self.memory_efficient_forward(outputs, outputs2)

    def __repr__(self, _repr_indent=4):
        head = "Hungarian Multisample Matching " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask)
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
