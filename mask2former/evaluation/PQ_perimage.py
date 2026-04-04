import itertools
import json
import logging
import os
import pickle
import tempfile
from collections import OrderedDict
from tabulate import tabulate
from detectron2.evaluation import COCOPanopticEvaluator
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager


class PanopticQualityPerImage(COCOPanopticEvaluator):
    """Calculate the PQ score for panoptic segmentation, but in addition to the standard per-class process
    do it on a global (not per-class) basis and produce per-image results. As part of the per-class process,
    also ignore classes set to ignore in the ground truth JSON when under distribution shift."""

    def __init__(self, dataset_name, output_dir):

        super().__init__(dataset_name, output_dir)
        self._logger = logging.getLogger(__name__)
        self.dataset_name = dataset_name

    def reset(self):
        self._logger.info('Resetting PanopticQualityPerImage')
        super().reset()

    def evaluate(self):

        self._logger.info('Evaluating Panoptic Quality metric, including dist shift support and per-image level results')
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            self._logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from .panopticapi_modified import pq_compute_global

            pq_res, pq_stat = pq_compute_global(
                gt_json,
                PathManager.get_local_path(predictions_json),
                gt_folder=gt_folder,
                pred_folder=pred_dir,
            )

        # Dump all raw data (per-image per-class results, TP, FP, etc.) to disk with pickle
        # Following proper convention this would be JSON but I don't have the time to deal with this now
        output_fn = os.path.join(self._output_dir, "raw_pq_data.pkl")
        self._logger.info("Writing raw panoptic predictions to {} ...".format(output_fn))
        with PathManager.open(output_fn, "wb") as f:
            pickle.dump(pq_stat, f)

        output_fn = os.path.join(self._output_dir, "panoptic_results.json")
        self._logger.info('Writing per-class and global panoptic predictions to {} ...'.format(output_fn))
        with PathManager.open(output_fn, "w") as f:
            json.dump(pq_res, f)

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]
        res["PQ_gl"] = 100 * pq_res["Global"]["pq"]
        res["RQ_gl"] = 100 * pq_res["Global"]["rq"]
        res["SQ_gl"] = 100 * pq_res["Global"]["sq"]

        results = OrderedDict({"panoptic_seg_per_image_impl": res})
        self._print_panoptic_results(pq_res)

        return results

    def _print_panoptic_results(self, pq_res):
        headers = ["", "PQ", "SQ", "RQ", "#categories"]
        data = []
        for name in ["All", "Things", "Stuff"]:
            row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
            data.append(row)
        name = 'Global'
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + ["N/A"]
        data.append(row)
        table = tabulate(
            data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
        )
        self._logger.info("Panoptic Evaluation Results (w/ dist shift support):\n" + table)