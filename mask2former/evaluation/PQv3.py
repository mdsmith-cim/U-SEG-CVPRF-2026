# Based off original Detectron2 panopticapi wrapper, https://github.com/cocodataset/panopticapi and https://github.com/google/uncertainty-baselines/blob/15b6fcdeab4a7666948a476f65eebdd3a7a572fa/baselines/mnist/utils.py#
import json
import logging
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from panopticapi.evaluation import VOID, OFFSET
from tabulate import tabulate

from mask2former.evaluation.panopticapi_modified import PQStatGlobal, rgb2id
from mask2former.evaluation.calibTracker import CalibTracker


class PanopticQualityV3(DatasetEvaluator):
    """Calculate the PQ score for panoptic segmentation, but in addition to the standard per-class process
    do it on a global (not per-class) basis and produce per-image results. As part of the per-class process,
    also ignore classes set to ignore in the ground truth JSON when under distribution shift."""

    def __init__(self, dataset_name, output_dir, device=None):

        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        self.device = device

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)

        self._logger = logging.getLogger(__name__)
        self.dataset_name = dataset_name

        gt_json_fn = PathManager.get_local_path(self._metadata.panoptic_json)
        assert os.path.isfile(gt_json_fn), f'Expect ground truth JSON file at {gt_json_fn} to exist!'
        with open(gt_json_fn, 'r') as f:
            self.gt_data = json.load(f)

        self.gt_anns = {i['image_id']: i for i in self.gt_data['annotations']}
        self.categories = {el['id']: el for el in self.gt_data['categories']}

        self.gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)
        if not os.path.isdir(self.gt_folder):
            raise Exception("Folder {} with ground truth segmentations doesn't exist".format(self.gt_folder))

        self.calibTracker_classscore = CalibTracker(device=self.device)
        self.calibTracker_combinedscore = CalibTracker(device=self.device)
        self.calibTracker_maskscore = CalibTracker(device=self.device)

    def _convert_category_id(self, segment_info):
        segment_info = segment_info.copy()
        isthing = segment_info.pop("isthing", None)
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def reset(self):
        self.pq_stat = PQStatGlobal()

        self.calibTracker_classscore.reset()
        self.calibTracker_combinedscore.reset()
        self.calibTracker_maskscore.reset()

    def process(self, inputs, outputs):

        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            combined_score = output.get("class_mask_combined_score", None)
            mask_score = output.get('norm_sigmoid_mask_score', None)
            class_score = output.get('softmax_cls_score', None)

            if segments_info is None:
                raise NotImplementedError('Expect panoptic segmentation to provide `segments_info` list')

            # file_name = os.path.basename(input["file_name"])
            image_id = input["image_id"]
            segments_info = [self._convert_category_id(x) for x in segments_info]

            gt_ann = self.gt_anns[image_id]

            # formerly np.uint32
            pan_gt = torch.tensor(np.asarray(Image.open(os.path.join(self.gt_folder, gt_ann['file_name'])), dtype=int), device=self.device)
            pan_gt = rgb2id(pan_gt)

            gt_segms = {el['id']: el for el in gt_ann['segments_info']}
            pred_segms = {el['id']: el for el in segments_info}

            # predicted segments area calculation + prediction sanity checks
            pred_labels_set = set(el['id'] for el in segments_info)
            labels, labels_cnt = torch.unique(panoptic_img, return_counts=True)
            for label, label_cnt in zip(labels, labels_cnt):
                label, label_cnt = label.item(), label_cnt.item()
                if label not in pred_segms:
                    if label == VOID:
                        continue
                    raise KeyError(
                        'In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(
                            gt_ann['image_id'], label))
                pred_segms[label]['area'] = label_cnt
                pred_labels_set.remove(label)
                if pred_segms[label]['category_id'] not in self.categories:
                    raise KeyError(
                        'In the image with ID {} segment with ID {} has unknown category_id {}.'.format(
                            gt_ann['image_id'],
                            label,
                            pred_segms[label][
                                'category_id']))
            if len(pred_labels_set) != 0:
                raise KeyError(
                    'In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(
                        gt_ann['image_id'], list(pred_labels_set)))

            # confusion matrix calculation
            # Formerly uint64type but torch does not handle some basic multiplication operators for uint types
            pan_gt_pred = pan_gt.to(int) * OFFSET + panoptic_img.to(int)
            gt_pred_map = {}
            labels, labels_cnt = torch.unique(pan_gt_pred, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = int(label // OFFSET)
                pred_id = int(label % OFFSET)
                gt_pred_map[(gt_id, pred_id)] = intersection.item()

            # count all matched pairs
            gt_matched = set()
            pred_matched = set()

            correct_map = torch.zeros_like(pan_gt, dtype=torch.bool)

            # Count true positives
            for label_tuple, intersection in gt_pred_map.items():
                gt_label, pred_label = label_tuple
                if gt_label not in gt_segms:
                    continue
                if pred_label not in pred_segms:
                    continue
                if gt_segms[gt_label]['iscrowd'] == 1:
                    continue
                if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                    continue

                union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get(
                    (VOID, pred_label), 0)
                iou = intersection / union
                if iou > 0.5:
                    self.pq_stat[image_id][gt_segms[gt_label]['category_id']].tp += 1
                    self.pq_stat[image_id][gt_segms[gt_label]['category_id']].iou += iou
                    gt_matched.add(gt_label)
                    pred_matched.add(pred_label)
                    intersect_region = (panoptic_img == pred_label) & (pan_gt == gt_label)
                    assert intersect_region.sum().item() == intersection, '# of pixels in intersect region must match!'
                    correct_map[intersect_region] = True

            # count false negatives
            crowd_labels_dict = {}
            for gt_label, gt_info in gt_segms.items():
                if gt_label in gt_matched:
                    continue
                # crowd segments are ignored
                if gt_info['iscrowd'] == 1:
                    crowd_labels_dict[gt_info['category_id']] = gt_label
                    continue
                self.pq_stat[image_id][gt_info['category_id']].fn += 1

            # count false positives
            for pred_label, pred_info in pred_segms.items():
                if pred_label in pred_matched:
                    continue
                # intersection of the segment with VOID
                intersection = gt_pred_map.get((VOID, pred_label), 0)
                # plus intersection with corresponding CROWD region if it exists
                if pred_info['category_id'] in crowd_labels_dict:
                    intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
                # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
                if intersection / pred_info['area'] > 0.5:
                    continue
                self.pq_stat[image_id][pred_info['category_id']].fp += 1

            not_ignore_region = pan_gt != VOID
            if class_score is not None:
                self.calibTracker_classscore.update_panoptic(class_score, correct_map, not_ignore_region)
            if mask_score is not None:
                self.calibTracker_maskscore.update_panoptic(mask_score, correct_map, not_ignore_region)
            if combined_score is not None:
                self.calibTracker_combinedscore.update_panoptic(combined_score, correct_map, not_ignore_region)

    def evaluate(self):

        self._logger.info('Evaluating Panoptic Quality metric (v3)')
        comm.synchronize()

        self.pq_stat = comm.gather(self.pq_stat)
        pq_stat_tmp = PQStatGlobal()
        for i in self.pq_stat:
            pq_stat_tmp += i
        self.pq_stat = pq_stat_tmp

        self.calibTracker_classscore = comm.gather(self.calibTracker_classscore)
        self.calibTracker_maskscore = comm.gather(self.calibTracker_maskscore)
        self.calibTracker_combinedscore = comm.gather(self.calibTracker_combinedscore)

        calibTracker_classscore = CalibTracker(device=self.device)
        calibTracker_maskscore = CalibTracker(device=self.device)
        calibTracker_combinedscore = CalibTracker(device=self.device)
        for c, m, b in zip(self.calibTracker_classscore, self.calibTracker_maskscore, self.calibTracker_combinedscore):
            calibTracker_classscore += c
            calibTracker_maskscore += m
            calibTracker_combinedscore += b

        self.calibTracker_classscore = calibTracker_classscore
        self.calibTracker_maskscore = calibTracker_maskscore
        self.calibTracker_combinedscore = calibTracker_combinedscore

        if not comm.is_main_process():
            return

        ignored_classes = {i: j["name"] for i, j in self.categories.items() if j.get("ignore", False)}
        self._logger.info(f'Ignoring classes: {ignored_classes}')

        # Standard per class calculation
        metrics = [("All", None), ("Things", True), ("Stuff", False)]
        results = {}
        for name, isthing in metrics:
            results[name], per_class_results = self.pq_stat.pq_average_standard(self.categories, isthing=isthing)
            if name == 'All':
                results['per_class'] = per_class_results
        # Global calculation
        results['Global'] = self.pq_stat.pq_average_global()

        # ** Calibration **
        save_file_class, save_file_mask, save_file_combined = None, None, None
        if self._output_dir:
            save_file_class = os.path.join(self._output_dir, "panoptic_calib_class.npz")
            save_file_mask = os.path.join(self._output_dir, "panoptic_calib_mask.npz")
            save_file_combined = os.path.join(self._output_dir, "panoptic_calib_combined.npz")

        calibResclass = self.calibTracker_classscore.evaluate(save_file_class)
        calibResmask = self.calibTracker_maskscore.evaluate(save_file_mask)
        calibRescombined = self.calibTracker_combinedscore.evaluate(save_file_combined)

        results["Calibration-Class"] = calibResclass['Calibration']
        results["Calibration-Mask"] = calibResmask['Calibration']
        results["Calibration-Combined"] = calibRescombined['Calibration']

        # Dump all raw data (per-image per-class results, TP, FP, etc.) to disk with pickle
        # Following proper convention ideally this would be JSON
        output_fn = os.path.join(self._output_dir, "raw_pq_data.pkl")
        self._logger.info("Writing raw panoptic predictions to {} ...".format(output_fn))
        with PathManager.open(output_fn, "wb") as f:
            pickle.dump(self.pq_stat, f)

        output_fn = os.path.join(self._output_dir, "panoptic_results.json")
        self._logger.info('Writing per-class and global panoptic predictions to {} ...'.format(output_fn))
        with PathManager.open(output_fn, "w") as f:
            json.dump(results, f)

        res = {}
        for met in ('ECE', 'MCE', 'ACE'):
            res[met + '-Class'] = 100 * calibResclass['Calibration'][met]
            res[met + '-Mask'] = 100 * calibResmask['Calibration'][met]
            res[met + '-Combined'] = 100 * calibRescombined['Calibration'][met]

        res["PQ"] = 100 * results["All"]["pq"]
        res["SQ"] = 100 * results["All"]["sq"]
        res["RQ"] = 100 * results["All"]["rq"]
        res["PQ_th"] = 100 * results["Things"]["pq"]
        res["SQ_th"] = 100 * results["Things"]["sq"]
        res["RQ_th"] = 100 * results["Things"]["rq"]
        res["PQ_st"] = 100 * results["Stuff"]["pq"]
        res["SQ_st"] = 100 * results["Stuff"]["sq"]
        res["RQ_st"] = 100 * results["Stuff"]["rq"]
        res["PQ_gl"] = 100 * results["Global"]["pq"]
        res["RQ_gl"] = 100 * results["Global"]["rq"]
        res["SQ_gl"] = 100 * results["Global"]["sq"]

        self._print_panoptic_results(results)
        self._print_calibration_results(results)

        return OrderedDict({"panoptic_seg_v3": res})

    def _print_panoptic_results(self, pq_res):
        headers = ["", "PQ", "SQ", "RQ", "#categories"]
        data = []
        for name in ["All", "Things", "Stuff"]:
            row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
            data.append(row)
        name = 'Global'
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + ["N/A"]
        data.append(row)
        table = tabulate(data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center")
        self._logger.info("Panoptic Evaluation Results:\n" + table)

    def _print_calibration_results(self, res):
        headers = ("Source", "ECE", "ACE", "MCE")
        data = []
        for source in ('Mask', 'Class', 'Combined'):
            row = [source] + [res['Calibration-' + source][k] * 100 for k in ["ECE", "ACE", "MCE"]]
            data.append(row)
        table = tabulate(data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center")
        self._logger.info("Panoptic Calibration Results:\n" + table)
