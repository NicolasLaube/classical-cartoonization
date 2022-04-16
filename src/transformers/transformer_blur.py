"""Blur transformations classes."""
from typing import Literal, Tuple

import numpy as np
import cv2
from skimage.filters.rank import median
from skimage.morphology import disk, ball

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

    def __init__(
        self, kernel_type: Literal["square", "circle"] = "square", kernel_size: int = 3
    ):
        """Initialize the blur parameters"""
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        if self.kernel_type == "square":
            return cv2.medianBlur(input_img, self.kernel_size)
        if self.kernel_type == "circle":
            return median(input_img, ball(self.kernel_size))


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


class DilateErodeTransformer(Transformer):
    """Delete noise by dilating then eroding"""

    def __init__(
        self, kernel_type: Literal["square", "circle"] = "square", kernel_size: int = 3
    ):
        """Initialize the affine transformation"""
        if kernel_type == "square":
            self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if kernel_type == "circle":
            self.kernel = disk(kernel_size)

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        dilated = cv2.dilate(input_img, self.kernel, iterations=1)
        return cv2.erode(dilated, self.kernel, iterations=1)
