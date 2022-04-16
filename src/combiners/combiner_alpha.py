"""Mask combiner"""
import cv2
import numpy as np

from src.base.base_combiner import Combiner
from src.base.image_array import ImageArray


class AlphaCombiner(Combiner):
    """Combine an image with a given alpha"""

    def __init__(self, alpha: float = 0.5):
        """Initialize the mask combiner"""
        self.alpha = alpha

    def __call__(self, image1: ImageArray, image2: ImageArray) -> ImageArray:
        """Combine two images"""
        return ((1 - self.alpha) * image1 + self.alpha * image2).astype(np.uint8)
