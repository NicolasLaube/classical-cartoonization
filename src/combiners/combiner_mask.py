"""Mask combiner"""
import cv2
import numpy as np

from src.base.base_combiner import Combiner
from src.base.image_array import ImageArray


class MaskCombiner(Combiner):
    """Combine an image with a mask"""

    def __init__(self, opaqueness: float = 1):
        """Initialize the mask combiner"""
        self.opaqueness = opaqueness

    def __call__(self, image1: ImageArray, image2_with_mask: ImageArray) -> ImageArray:
        """Combine two images"""
        # return cv2.bitwise_and(input_img, input_img, mask=mask)

        mask = image2_with_mask[:, :, -1]
        image2 = image2_with_mask[:, :, :-1]

        # get first masked value (foreground)
        first_mask = cv2.bitwise_or(image2, image2, mask=mask)

        # get second masked value (background) mask must be inverted
        mask = cv2.bitwise_not(mask)
        background_mask = cv2.bitwise_or(image1, image1, mask=mask)

        # combine foreground+background
        combined = cv2.bitwise_or(first_mask, background_mask)

        # Use opaqueness
        return (self.opaqueness * combined + (1 - self.opaqueness) * image1).astype(
            np.uint8
        )
