import logging
import os
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator

from mask2former.utils.visualizer import Visualizer


class saveVisualizations(DatasetEvaluator):
    """Write segmentation and uncertainty data to disk, implemented as an evaluator to fit into existing framework."""

    def __init__(self, dataset_name, output_dir, vis_cfg):
        """
        Initialize evaluator, preparing output folder structure and collecting necessary metadata.
        @param output_dir: Output directory to save results in.
        """
        super().__init__()
        self._logger = logging.getLogger(__name__)

        self.dataset_name = dataset_name
        self.meta = MetadataCatalog.get(self.dataset_name)
        self.class_names = self.meta.stuff_classes
        self.num_classes = len(self.meta.stuff_classes)
        self.ignore_label = self.meta.ignore_label

        output_folder = Path(output_dir)
        assert output_folder.exists(), f"Output directory '{output_folder}' does not exist!"
        self.output_folder = output_folder / 'visualizations'
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Detectron default is environment variable - this is constant across all datasets
        self.dataset_root = os.getenv("DETECTRON2_DATASETS", "datasets")
        # Set image root if available; if not, fall back to dataset root
        try:
            self.image_root = self.meta.image_root
        except AttributeError:
            self.image_root = self.dataset_root

        self.vis_cfg = vis_cfg

    def process(self, inputs: list, outputs: list):
        """
        Visualize and save all results to disk.
        Parameters inputs and outputs are standardized by detectron2 for any evaluator; see the detectron2 documentation for details.
        @param inputs: List of inputs, each a dict with at least keys 'file_name' and 'image_id'.
        @param outputs: List of outputs, each a dict with keys corresponding to the data types output by the model.
        """

        for input, output in zip(inputs, outputs):

            # Between semantic/panoptic file_name is one of the few constants
            # Alternatives for unique image IDs such as image_id are panoptic only
            fn = Path(input['file_name']).relative_to(self.dataset_root)
            fn_no_ext = fn.with_suffix('')

            output_folder = self.output_folder / fn_no_ext
            output_folder.mkdir(parents=True, exist_ok=True)

            img_rgb = input['image'].permute(1, 2, 0)

            # Copy over RGB
            shutil.copy(input['file_name'], output_folder)

            for dataType, data in output.items():
                viz = Visualizer(img_rgb, metadata=self.meta)
                if dataType == 'sem_seg':
                    scores, labels = data.max(0)
                    viz.draw_sem_seg(labels.cpu(), scores.cpu(), area_threshold=0)

                elif dataType == 'panoptic_seg':
                    pan_id, seg_info = data
                    viz.draw_panoptic_seg(pan_id.cpu(), seg_info, area_threshold=0)

                elif dataType == 'instances':
                    # Not implemented; ignore
                    continue
                # For various confidence/uncertainty measures, show color coded map
                else:
                    # self._logger.debug(f'Data {dataType} - min: {data.min()} | max: {data.max()}')
                    viz_out = viz.get_output()
                    vmin, vmax = self.vis_cfg.UNCERT_LIMITS.get(dataType, (None, None))
                    viz_out.ax.imshow(data.cpu(), vmin=vmin, vmax=vmax, extent=(0, viz_out.width, viz_out.height, 0))

                viz.get_output().save(output_folder / (dataType + '.' + self.vis_cfg.FORMAT), bbox_inches='tight',
                                      dpi=150)

            # Save ground truth panoptic
            if 'segments_info' in input and 'pan_seg_file_name' in input:
                viz = Visualizer(img_rgb, metadata=self.meta)

                with open(input["pan_seg_file_name"], "rb") as f:
                    pan_seg = torch.tensor(np.asarray(Image.open(f)))

                from mask2former.evaluation.panopticapi_modified import rgb2id
                pan_seg = rgb2id(pan_seg)

                # Add is thing info - required
                segments_info = deepcopy(input["segments_info"])
                for s in segments_info:
                    s['isthing'] = s['category_id'] in self.meta.thing_dataset_id_to_contiguous_id.values()
                viz.draw_panoptic_seg(pan_seg, segments_info, area_threshold=0, alpha=0.5)
                viz.get_output().save(output_folder / ('gt_panoptic.' + self.vis_cfg.FORMAT), bbox_inches='tight',
                                      dpi=150)

            # Save ground truth sem seg
            if 'sem_seg' in input:
                viz = Visualizer(img_rgb, metadata=self.meta)

                sem_seg_gt = input['sem_seg']

                viz.draw_sem_seg(sem_seg_gt, area_threshold=0, alpha=0.5)
                viz.get_output().save(output_folder / ('gt_semseg.' + self.vis_cfg.FORMAT), bbox_inches='tight',
                                      dpi=150)
