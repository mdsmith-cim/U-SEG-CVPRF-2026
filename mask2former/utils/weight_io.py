# Based off https://github.com/mdsmith-cim/detectron2/blob/main/detectron2/utils/file_io.py
from iopath.common.file_io import PathHandler
from detectron2.utils.file_io import PathManager
import os
import os.path

class WeightFolderHandler(PathHandler):
    """
    Allows the use of the "weightfolder://" prefix when loading models from disk, which refers to an environment variable
    DETECTRON2_WEIGHT_FOLDER with default ~/Mask2FormerModels/ if not set.
    """

    PREFIX = "weightfolder://"

    def __init__(self):
        super(PathHandler, self).__init__()
        base_path = os.getenv("DETECTRON2_WEIGHT_FOLDER", "~/Mask2FormerModels/")
        self.base_path = os.path.expanduser(base_path)

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path, **kwargs):
        path = path[len(self.PREFIX) :]
        return PathManager.get_local_path(os.path.join(self.base_path, path), **kwargs)

    def _open(self, path, mode="r", **kwargs):
        path = path[len(self.PREFIX) :]
        return PathManager.open(os.path.join(self.base_path, path), mode, **kwargs)


PathManager.register_handler(WeightFolderHandler())