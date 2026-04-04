import json
import logging
import os
from pathlib import Path

import h5py
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm

class saveResults(DatasetEvaluator):
    """Saves results from model (predictions, uncertainty information) to disk, implemented as an evaluator to fit into existing framework."""

    def __init__(self, output_dir, compression=None):
        """
        Initialize HDF5 file for saving results.
        @param output_dir: Output directory to save results (model_output.hdf5) in.
        @param compression: Passed along to h5py library; specifies type of compression to use. Default of None means no compression.
        """
        super().__init__()
        self._logger = logging.getLogger(__name__)

        if comm.get_rank() > 0:
            raise Exception("saveResults currently does not work with distributed communication")

        self.compression = compression
        self.root_folder = Path(output_dir)
        assert self.root_folder.exists(), f"Output directory '{self.root_folder}' does not exist!"

        # Detectron default is environment variable - this is constant across all datasets
        self.dataset_root = os.getenv("DETECTRON2_DATASETS", "datasets")

        # Don't create h5file just yet
        self.h5file = False

    def _inith5(self):
        fn = self.root_folder / f"model_output.hdf5"
        self._logger.info(f"Saving results to HDF5 file {fn}")
        self.h5file = h5py.File(fn, "w", libver='latest')

    def reset(self):
        """
        Closes the HDF5 file and opens a new one, overwriting the old one.
        """
        if self.h5file:
            self.h5file.close()
        self._inith5()

    def process(self, inputs: list, outputs: list):
        """
        Save all results to the HDF5 file on disk, indexed by the filename as an HDF5 group.
        Parameters inputs and outputs are standardized by detectron2 for any evaluator; see the detectron2 documentation for details.
        @param inputs: List of inputs, each a dict with at least keys 'file_name' and 'image_id'.
        @param outputs: List of outputs, each a dict with keys corresponding to the data types output by the model.
        """
        if not self.h5file:
            raise BrokenPipeError(f"HDF5 file {self.h5file.filename} is closed! Call .reset() first!")

        for input, output in zip(inputs, outputs):

            # Between semantic/panoptic file_name is one of the few constants
            # Alternatives for unique image IDs such as image_id are panoptic only
            fn = Path(input['file_name']).relative_to(self.dataset_root)
            grp = self.h5file.create_group(str(fn.with_suffix('')), track_order=True)
            if 'image_id' in input:
                grp.attrs['image_id'] = input['image_id']
            if 'is_ood' in input:
                grp.attrs['is_ood'] = input['is_ood']

            for dataType, data in output.items():
                if dataType == 'panoptic_seg':
                    grp.create_dataset(dataType + '_id', data=data[0].cpu().numpy(), compression=self.compression)
                    grp.create_dataset(dataType + '_info', data=json.dumps(data[1]), shape=1,
                                       compression=self.compression)
                elif dataType == 'sem_seg':
                    # Saving semantic segmentation scores for all classes is not practical in terms of file sizes;
                    # so we save only the labels as they are all that is needed for evaluation
                    grp.create_dataset(dataType + "_labels", data=data.argmax(dim=0).cpu().numpy(),
                                       compression=self.compression)
                elif dataType == 'instances':
                    raise NotImplementedError("Instance segmentation not supported!")
                else:
                    grp.create_dataset(dataType, data=data.cpu().numpy(), compression=self.compression)

        # Might address some edge case memory usage issues on large scale clusters with Lustre filesystems
        self.h5file.flush()

    def evaluate(self):
        """
        Save results to disk and close file.
        """
        self.h5file.flush()
        self._logger.info(f"Saved results to {self.h5file.filename}")
        self.h5file.close()
