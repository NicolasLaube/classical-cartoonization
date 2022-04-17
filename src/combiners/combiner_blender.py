"""Blender combiner."""
import cv2

from src.base.base_combiner import Combiner
from src.base.image_array import ImageArray


class CombinerImageBlender(Combiner):
    """Combine an image with a mask"""

    def __init__(
        self,
        weight_image_1: float = 0.7,
        weight_image_2: float = 0.3,
    ):
        """Initialize the mask combiner"""
        self.weight_image_1 = weight_image_1
        self.weight_image_2 = weight_image_2

    def __call__(self, image1: ImageArray, image2: ImageArray) -> ImageArray:
        """Blend two images"""
        min_shape = (
            min(image1.shape[0], image2.shape[0]),
            min(image1.shape[1], image2.shape[1]),
        )
        image1 = cv2.resize(image1, min_shape)
        image2 = cv2.resize(image2, min_shape)
        return cv2.addWeighted(image1, 0.3, image2, 0.7, 0)
