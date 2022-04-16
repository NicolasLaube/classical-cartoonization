"""Color transformers."""
from typing import List, Tuple

import cv2
import numpy as np

from src.base.base_transformer import Transformer
from src.base.image_array import ImageArray, ImageFormat
from src.dataset.formatter import format_image


class HSVAffineTransformer(Transformer):
    """Affine hsv transformation"""

    def __init__(
        self,
        h_a: float = 1,
        h_b: float = 0,
        s_a: float = 1,
        s_b: float = 0,
        v_a: float = 1,
        v_b: float = 0,
    ):
        """Initialize the affine transformation (a*x + b)"""
        self.h_a = h_a
        self.h_b = h_b
        self.s_a = s_a
        self.s_b = s_b
        self.v_a = v_a
        self.v_b = v_b

    def transform(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        hsv = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] * self.h_a + self.h_b) % 256
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.s_a + self.s_b, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * self.v_a + self.v_b, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


class SpecificColorTransformer(Transformer):
    """Modifines specicied colors."""

    def __init__(self, color_params: List[Tuple[List[int], float, float]]):
        """Initialize the specific color transformer"""
        self.color_params = color_params


class ColorTransformer(Transformer):
    """To transform an image of a certain type to another"""

    def __init__(self, to_format: ImageFormat, return_mask: bool = False):
        """Initialize the image converter"""
        self.to_format = to_format
        self.return_mask = return_mask

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        idx = 1 if self.return_mask else 0
        return format_image(input_img, self.to_format)[idx]


class BinsQuantizationTransformer(Transformer):
    """Quantize an image to a certain number of colors using bins"""

    def __init__(self, n_colors: int):
        """Initialize the color quantization transformer"""
        self.bins = 256 // n_colors

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        return input_img // self.bins * self.bins + 128 // self.bins


class TransformerKMeans(Transformer):
    """Kmeans trasnformer"""

    def __init__(self, n_colors: int):
        """Initialize the color quantization transformer"""
        self.n_colors = n_colors

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        reshaped_image = np.float32(input_img.reshape((-1, 3)))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(
            reshaped_image, self.n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        center = np.uint8(center)
        results = center[label.flatten()]  # pylint: disable = E1136
        image_kmeans_colors = results.reshape((input_img.shape))
        return image_kmeans_colors
