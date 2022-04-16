"""Color transformers."""
from dataclasses import dataclass
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


@dataclass
class ColorTransformParams:
    """Parameters for color transformation"""

    color: List[int]
    threshold: float = 0
    a: float = 1  # pylint: disable=invalid-name
    b: float = 0  # pylint: disable=invalid-name


class AffineColorTransformer(Transformer):
    """Do an affine transform on the specified colors"""

    def __init__(self, color_transforms: List[ColorTransformParams]):
        """Initialize the affine color transformer"""
        self.color_transforms = color_transforms

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        for color_transform in self.color_transforms:
            color_weights = input_img.dot(color_transform.color) / np.array(
                color_transform.color
            ).dot(color_transform.color)
            duplicated_color = np.tile(
                color_transform.color, (input_img.shape[0], input_img.shape[1], 1)
            )
            weight_matrix = color_weights * (color_transform.a - 1) + color_transform.b
            mask = np.where(color_weights >= color_transform.threshold, 1, 0)
            weight_matrix *= mask
            input_img = np.clip(
                input_img.astype(np.float64)
                + np.einsum(
                    "kij->ijk",
                    np.multiply(np.einsum("ijk->kij", duplicated_color), weight_matrix),
                ),
                0,
                255,
            ).astype(np.uint8)
        return input_img
