"""Blender combiner."""
from typing import Tuple

import cv2

from src.base.base_combiner import Combiner
from src.base.image_array import ImageArray


class ImageBlenderCombiner(Combiner):
    """Combine an image with a mask"""

    def __init__(
        self,
        weight_image_1: float = 0.7,
        weight_image_2: float = 0.3,
        size: Tuple[int, int] = (256, 256),
    ):
        """Initialize the mask combiner"""
        self.weight_image_1 = weight_image_1
        self.weight_image_2 = weight_image_2
        self.size = size

    def __call__(self, image1: ImageArray, image2: ImageArray) -> ImageArray:
        """Blend two images"""
        image1 = cv2.resize(image1, self.size)
        image2 = cv2.resize(image2, self.size)
        return cv2.addWeighted(image1, 0.3, image2, 0.7, 0)
