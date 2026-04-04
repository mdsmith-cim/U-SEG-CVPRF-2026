# This class exists as a convenient way to store calibration related data, used for calculating
# the Expected Calibration Error among others
# During semantic/panoptic eval use update_sem or update_panoptic to update the per-bin accuracy/confidence for each image
# Somewhat based on https://github.com/google/uncertainty-baselines/blob/15b6fcdeab4a7666948a476f65eebdd3a7a572fa/baselines/mnist/utils.py#
# Modified to fit in this evaluator paradigm and to accumulate results on a per-image basis as they come in
# without storing the results for every image

from typing import Union

import numpy as np
import torch
from typing_extensions import TypeAlias

DeviceLikeType: TypeAlias = Union[str, torch.device, int]

class CalibTracker:

    def __init__(self, num_bins: int = 15, device: torch.device = 'cpu'):
        """
        Initialize tracker.
        :param num_bins: Number of confidence bins.
        :param device: torch device to use.
        """
        # BRAVO challenge uses 15 bins; most papers seem to use between 10 and 20
        self.num_bins = num_bins
        self.device = device
        self.tau_tab = torch.linspace(0, 1, self.num_bins + 1, device=self.device)  # confidence bins

        self.acc_tab = torch.zeros(self.num_bins, dtype=torch.float64,
                                   device=self.device)  # empirical (true) confidence
        self.mean_conf = torch.zeros(self.num_bins, dtype=torch.float64, device=self.device)  # predicted confidence
        self.nb_items_bin = torch.zeros(self.num_bins, dtype=torch.int64,
                                        device=self.device)  # number of items in the confidence bins

    def to(self, device: DeviceLikeType):
        self.device = device
        self.tau_tab = self.tau_tab.to(device=device)
        self.acc_tab = self.acc_tab.to(device=device)
        self.mean_conf = self.mean_conf.to(device=device)
        self.nb_items_bin = self.nb_items_bin.to(device=device)

    def __iadd__(self, other):
        assert self.num_bins == other.num_bins, f'{self.num_bins} bins != {other.num_bins} bins'
        assert self.device == other.device, f'Device {self.device} != Device {self.device}'
        self.acc_tab += other.acc_tab
        self.mean_conf += other.mean_conf
        self.nb_items_bin += other.nb_items_bin
        return self

    def reset(self):
        self.acc_tab = torch.zeros(self.num_bins, dtype=torch.float64,
                                   device=self.device)  # empirical (true) confidence
        self.mean_conf = torch.zeros(self.num_bins, dtype=torch.float64, device=self.device)  # predicted confidence
        self.nb_items_bin = torch.zeros(self.num_bins, dtype=torch.int64,
                                        device=self.device)  # number of items in the confidence bins

    def update_sem(self, confidence: torch.Tensor, predicted_classes: torch.Tensor, gt_classes: torch.Tensor,
                   not_ignore_region=None):
        """
        Update calibration data with semantic segmentation data.
        :param confidence: Confidence at each pixel
        :param predicted_classes: Predicted class label at each pixel.
        :param gt_classes: Ground truth class labels at each pixel.
        :param not_ignore_region: True for all pixels that have valid ground truth.
        """
        if not_ignore_region is None:
            not_ignore_region = True

        for i in torch.arange(self.num_bins):
            # select the items where the predicted max probability falls in the bin
            # [tau_tab[i], tau_tab[i + 1)]
            # Note: we ignore anything where ground truth is set to ignore
            sec = (self.tau_tab[i + 1] > confidence) & (confidence >= self.tau_tab[i]) & not_ignore_region
            # Number of items in this bin, in this image
            nb_items_cur = torch.sum(sec)
            # Skip if nothing to do
            if nb_items_cur == 0:
                continue
            # Update running bin counter
            self.nb_items_bin[i] += nb_items_cur  # Number of items in the bin
            # select the predicted classes, and the true classes
            class_pred_sec, y_sec = predicted_classes[sec], gt_classes[sec]
            # average of the predicted max probabilities
            self.mean_conf[i] += confidence[sec].sum()
            # compute the empirical confidence
            self.acc_tab[i] += (class_pred_sec == y_sec).sum()

    def update_panoptic(self, confidence: torch.Tensor, accuracy_map: torch.Tensor, not_ignore_region=None):
        """
        Update calibration with panoptic data.
        :param confidence: Per pixel confidence value.
        :param accuracy_map: Accuracy at each pixel (bool array, True=correct)
        :param not_ignore_region: Pixels that have valid ground truth.
        """

        if not_ignore_region is None:
            not_ignore_region = True

        for i in torch.arange(self.num_bins):
            # select the items where the predicted max probability falls in the bin
            # [tau_tab[i], tau_tab[i + 1)]
            # Note: we ignore anything where ground truth is set to ignore
            sec = (self.tau_tab[i + 1] > confidence) & (confidence >= self.tau_tab[i]) & not_ignore_region
            # Number of items in this bin, in this image
            nb_items_cur = torch.sum(sec)
            # Skip if nothing to do
            if nb_items_cur == 0:
                continue
            # Update running bin counter
            self.nb_items_bin[i] += nb_items_cur  # Number of items in the bin
            # average of the predicted max probabilities
            self.mean_conf[i] += confidence[sec].sum()
            # compute the empirical confidence where predicted class and instance IDs were found to be correct
            self.acc_tab[i] += accuracy_map[sec].sum()

    def evaluate(self, save_file=None) -> dict:
        """
        Calculate calibration statistics (ECE, ACE, MCE) and save raw per-bin data to disk if save_file is specified.
        :param save_file: Filename to save data to.
        :return: Dict of results, with key calibration containing ECE, MCE and ACE results.
        """
        # Calculate mean
        mean_conf = self.mean_conf / self.nb_items_bin
        acc_tab = self.acc_tab / self.nb_items_bin

        # Remove any empty bins
        non_empty_bins = self.nb_items_bin > 0
        mean_conf = mean_conf[non_empty_bins]
        acc_tab = acc_tab[non_empty_bins]
        nb_items_bin = self.nb_items_bin[non_empty_bins]

        # Convert to numpy for existence of np.average as quick fix
        # Speed not critical here will run only once per dataset
        mean_conf = mean_conf.cpu().numpy()
        acc_tab = acc_tab.cpu().numpy()
        nb_items_bin = nb_items_bin.cpu().numpy()

        # Expected Calibration Error
        if len(nb_items_bin) > 0:
            ece = np.average(np.absolute(mean_conf - acc_tab),
                             weights=nb_items_bin.astype(float) / np.sum(nb_items_bin))
            # Maximum Calibration Error
            mce = np.max(np.absolute(mean_conf - acc_tab))
            # Average Calibration Error
            ace = np.average(np.absolute(mean_conf - acc_tab))
        else:
            ece, mce, ace = np.full(1, np.nan), np.full(1, np.nan), np.full(1, np.nan)

        if save_file is not None:
            np.savez(save_file, mean_confidence=mean_conf, mean_accuracy=acc_tab, number_items_per_bin=nb_items_bin,
                     num_bins=self.num_bins, bins=self.tau_tab.cpu().numpy(), ece=ece, mce=mce, ace=ace)

        results = {'Calibration': {"ECE": ece.item(), "MCE": mce.item(), "ACE": ace.item()}}

        return results

    def __str__(self):
        return f'Calibration Data: {self.num_bins} bins, device: {self.device}, acc: {self.acc_tab / self.nb_items_bin}, conf: {self.mean_conf / self.nb_items_bin}, nb_items_bin: {self.nb_items_bin}'

    def __repr__(self):
        return f'CalibData: {self.num_bins} bins, device: {self.device}, acc raw: {self.acc_tab}, conf raw: {self.mean_conf}, nb_items_bin: {self.nb_items_bin}'
