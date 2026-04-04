import numpy as np
from fvcore.transforms.transform import NoOpTransform, Transform
import torch
from torchvision.transforms.v2 import functional as F2

class GaussianNoiseTransform(Transform):
    """
    This method returns a copy of this image with Gaussian noise applied, with specified mean and std. dev.
    """

    def __init__(self, mean: float = 0.0, sigma: float = 0.1, clip=True):
        """
        Args:
            mean (float): Mean of the sampled normal distribution. Default is 0.
            sigma (float): Standard deviation of the sampled normal distribution. Default is 0.1.
            clip (bool, optional): Whether to clip the values in ``[0, 1]`` after adding noise. Default is True.
        """
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.clip = clip


    def apply_image(self, img):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """

        if len(img) == 0:
            return img

        img = torch.as_tensor(img.copy()) # Copy required b/c of negative strides in original np array
        orig_dtype = img.dtype
        # Convert first to float and range [0,1] as expected by Gaussian
        img = F2.to_dtype(img, dtype=torch.float32, scale=True)
        # Apply Gaussian noise
        img = F2.gaussian_noise(img, mean=self.mean, sigma=self.sigma, clip=self.clip)
        # Undo conversion
        img = F2.to_dtype(img, dtype=orig_dtype, scale=True)
        img = img.numpy()
        return img

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        segmentation (ndarray): of shape HxW. The array should have integer
        or bool dtype.
        Does nothing as Gaussian Noise not applicable to segmentation.
        """
        return segmentation

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        return coords

    def inverse(self):
        return NoOpTransform()