"""Blur transformations classes."""
from typing import Tuple

import cv2

from src.base.base_transformer import Transformer
from src.base.image_array import ImageArray


class GaussianBlurTransformer(Transformer):
    """Apply gaussian blur to transformer"""

    def __init__(self, kernel: Tuple[int, int] = (5, 5), stdev: int = 0):
        """Initialize the blur parameters"""
        self.kernel = kernel
        self.stdev = stdev

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        return cv2.GaussianBlur(input_img, self.kernel, self.stdev)


class MedianBlurTransformer(Transformer):
    """Apply median blur to transformer"""

    def __init__(self, ksize: int = 5):
        """Initialize the blur parameters"""
        self.ksize = ksize

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        return cv2.medianBlur(input_img, self.ksize)


class BilateralBlurTransformer(Transformer):
    """Apply bilateral blur to transformer"""

    def __init__(
        self, sigma_color: int = 5, sigma_space: int = 80, border_type: int = 80
    ):
        """Initialize the blur parameters"""
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.border_type = border_type

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        return cv2.bilateralFilter(
            input_img, self.sigma_color, self.sigma_space, self.border_type
        )
