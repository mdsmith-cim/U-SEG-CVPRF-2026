from detectron2.data.transforms import Augmentation
from mask2former.data.augmentation.custom_transforms import GaussianNoiseTransform

class GaussianNoiseAugmentation(Augmentation):
    """
    Add gaussian noise to images.
    """

    def __init__(self, mean: float = 0.0, sigma: float = 0.1, clip=True):
        """
        Args:
            mean (float): Mean of the sampled normal distribution. Default is 0.
            sigma (float): Standard deviation of the sampled normal distribution. Default is 0.1.
            clip (bool, optional): Whether to clip the values in ``[0, 1]`` after adding noise. Default is True.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        return GaussianNoiseTransform(self.mean, self.sigma, self.clip)