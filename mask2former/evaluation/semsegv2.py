import logging
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from tabulate import tabulate

from mask2former.evaluation.calibTracker import CalibTracker, DeviceLikeType


def load_image_into_torch_tensor(filename: str, dtype = None, device: DeviceLikeType = None) -> torch.Tensor:
    with PathManager.open(filename, "rb") as f:
        array = torch.tensor(np.asarray(Image.open(f), dtype=dtype), device=device)
    return array


class SemSegEvaluatorV2(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics, including DICE. Computed per-class, globally, and per-image without consideration for classes.
    See https://arxiv.org/pdf/1605.06211 for original paper that introduced most of these metrics.
    """

    def __init__(
            self,
            dataset_name,
            distributed=True,
            output_dir=None,
            *,
            sem_seg_loading_fn=load_image_into_torch_tensor,
            device=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
        """
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self.device = device

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        self._class_names = meta.stuff_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._num_classes = len(meta.stuff_classes)
        self._ignore_label = meta.ignore_label

        self.calibTracker_classscore = CalibTracker(device=self.device)
        self.calibTracker_combinedscore = CalibTracker(device=self.device)
        self.calibTracker_maskscore = CalibTracker(device=self.device)

        # Detectron default is environment variable - this is constant across all datasets
        self.dataset_root = os.getenv("DETECTRON2_DATASETS", "datasets")

    def reset(self):
        self._conf_matrix = torch.zeros((self._num_classes + 1, self._num_classes + 1), dtype=torch.int64,
                                        device=self.device)
        self.per_img_data = {}

        self.calibTracker_classscore.reset()
        self.calibTracker_combinedscore.reset()
        self.calibTracker_maskscore.reset()

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            pred_labels = output["sem_seg"].argmax(dim=0)
            combined_score = output.get("class_mask_combined_score", None)
            mask_score = output.get('norm_sigmoid_mask_score', None)
            class_score = output.get('softmax_cls_score', None)

            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int, device=self.device)

            not_ignore_region = gt != self._ignore_label

            # Calibration error
            if class_score is not None:
                self.calibTracker_classscore.update_sem(class_score, pred_labels, gt, not_ignore_region)
            if mask_score is not None:
                self.calibTracker_maskscore.update_sem(mask_score, pred_labels, gt, not_ignore_region)
            if combined_score is not None:
                self.calibTracker_combinedscore.update_sem(combined_score, pred_labels, gt, not_ignore_region)

            # Dump ignore region into a placeholder class
            gt[gt == self._ignore_label] = self._num_classes

            # Generate confusion matrix
            conf_matrix = torch.bincount((self._num_classes + 1) * pred_labels.reshape(-1) + gt.reshape(-1), minlength=self._conf_matrix.numel()).reshape(self._conf_matrix.shape)

            fn = Path(input['file_name']).relative_to(self.dataset_root).with_suffix('')

            # Per-image we calculate DICE/IoU on all data, without class differentiation
            tp = conf_matrix.diagonal()[:-1].to(torch.float64).sum()
            pos_gt = torch.sum(conf_matrix[:-1, :-1], dim=0).to(torch.float64).sum()
            pos_pred = torch.sum(conf_matrix[:-1, :-1], dim=1).to(torch.float64).sum()
            per_img_iou = tp / (pos_gt + pos_pred - tp)  # IoU = DICE / (2 - DICE)
            per_img_dice = (2 * tp) / (pos_gt + pos_pred)

            self.per_img_data[str(fn)] = {'IoU': per_img_iou.item() * 100, 'DICE': per_img_dice.item() * 100}

            # For all predictions
            self._conf_matrix += conf_matrix


    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            per_img_data_list = all_gather(self.per_img_data)
            self.per_img_data = {}
            for s in per_img_data_list:
                self.per_img_data.update(s)

            calib_tracker_class_list = all_gather(self.calibTracker_classscore)
            calib_tracker_mask_list = all_gather(self.calibTracker_maskscore)
            calib_tracker_combined_list = all_gather(self.calibTracker_combinedscore)

            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix.cpu().numpy())
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix.cpu().numpy()

            self.calibTracker_classscore = CalibTracker(device=self.device)
            self.calibTracker_maskscore = CalibTracker(device=self.device)
            self.calibTracker_combinedscore = CalibTracker(device=self.device)
            for c, m, b in zip(calib_tracker_class_list, calib_tracker_mask_list, calib_tracker_combined_list):
                self.calibTracker_classscore += c
                self.calibTracker_maskscore += m
                self.calibTracker_combinedscore += b

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        dice = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)  # True positives
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)  # Number of GT examples
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)  # Number of predicted examples
        acc_valid = pos_gt > 0  # All classes with at least 1 GT label
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp  # Union = denominator of Jaccard index
        cards = pos_gt + pos_pred  # Set cardinalities = denominator of dice
        iou_valid = np.logical_and(acc_valid, union > 0)
        dice_valid = np.logical_and(acc_valid, cards > 0)
        # This should never trigger unless something goes very wrong
        assert np.all(dice_valid == iou_valid), f'Validity mismatch between IoU and DICE!'
        self._logger.info(f'{(~iou_valid).sum()} invalid classes for per-class semantic segmentation metrics')
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]  # Normal per-class IoU
        # Note: for global metrics we use all available data, so as to not discard misclassifications with a distribution
        # shifted class
        # For per-class we have to throw out tne entire class as DICE/IoU will always be 0 b/c true pos = 0 -> no point
        giou = np.sum(tp) / (np.sum(pos_gt) + np.sum(pos_pred) - np.sum(tp))  # Global IoU
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)  # Mean accuracy
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)  # Mean IoU
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])  # Frequency Weighted IoU
        pacc = np.sum(tp) / np.sum(pos_gt)  # Pixel Accuracy
        dice[iou_valid] = (2 * tp[iou_valid]) / cards[iou_valid]  # DICE (per-class)
        mdice = np.sum(dice[iou_valid]) / np.sum(iou_valid)  # Mean DICE
        gdice = np.sum(2 * tp) / (np.sum(pos_gt) + np.sum(pos_pred))  # Global DICE

        # ** Calibration **
        save_file_class, save_file_mask, save_file_combined = None, None, None
        if self._output_dir:
            save_file_class = os.path.join(self._output_dir, "semseg_calib_class.npz")
            save_file_mask = os.path.join(self._output_dir, "semseg_calib_mask.npz")
            save_file_combined = os.path.join(self._output_dir, "semseg_calib_combined.npz")

        calibResclass = self.calibTracker_classscore.evaluate(save_file_class)
        calibResmask = self.calibTracker_maskscore.evaluate(save_file_mask)
        calibRescombined = self.calibTracker_combinedscore.evaluate(save_file_combined)

        res = {}
        for met in ('ECE', 'MCE', 'ACE'):
            res[met + '-Class'] = 100 * calibResclass['Calibration'][met]
            res[met + '-Mask'] = 100 * calibResmask['Calibration'][met]
            res[met + '-Combined'] = 100 * calibRescombined['Calibration'][met]

        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            res[f"ACC-{name}"] = 100 * acc[i]
            res[f"DICE-{name}"] = 100 * dice[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        res['gIoU'] = 100 * giou
        res['mDICE'] = 100 * mdice
        res['gDICE'] = 100 * gdice

        res['perImage'] = self.per_img_data

        self._print_calibration_results(res)
        self._print_semantic_results(res)

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation_v2.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)

        # No need for per-image data to be dumped will spam the terminal
        res.pop('perImage')
        results = OrderedDict({"sem_seg_v2": res})

        return results

    def _print_calibration_results(self, res):
        headers = ("Source", "ECE", "ACE", "MCE")
        data = []
        for source in ('Mask', 'Class', 'Combined'):
            row = [source] + [res[k + '-' + source] for k in ["ECE", "ACE", "MCE"]]
            data.append(row)

        table = tabulate(data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center")
        self._logger.info("Semantic Calibration Results:\n" + table)

    def _print_semantic_results(self, res):
        headers = ["", "IoU", "Accuracy", "DICE"]
        data = [("Mean", res["mIoU"], res["mACC"], res["mDICE"]), ("Freq. Weighted", res["fwIoU"], None, None),
                ("Global", res["gIoU"], res['pACC'], res['gDICE']), ("Per-class", None, None, None)]
        for name in self._class_names:
            data.append((name, res["IoU-" + name], res["ACC-" + name], res["DICE-" + name]))
        table = tabulate(data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center")
        self._logger.info("Semantic Evaluation Results:\n" + table)