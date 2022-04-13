"""Unskew an image."""
import math
from typing import Tuple

import cv2
import numpy as np

from src.base.base_transformer import Transformer
from src.base.image_array import ImageArray


class TransformerUnskew(Transformer):
    """Unskew an image."""

    def __init__(self, size: Tuple[int, int] = (256, 256)):
        """Initialize the unskew transformer."""
        self.size = size

    @staticmethod
    def convert_line_to_angle(line: np.ndarray) -> float:
        """Convert a line to an angle."""
        try:
            x_1, y_1, x_2, y_2 = tuple(np.split(line, 4, 1))  # pylint: disable=W0632
            return math.degrees(math.atan2(y_2 - y_1, x_2 - x_1))
        except ValueError:
            return 0

    def __call__(self, image_array: ImageArray) -> ImageArray:
        """
        Unskews an image.
        ---
        1/ computes the median image angle
        2/ computes the rotation matrix from the angle
        3/ flips the image thanks to the rotation matrix
        """
        try:
            img_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3, L2gradient=True)
            lines = cv2.HoughLinesP(
                img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5
            )
            angles = np.array([self.convert_line_to_angle(line) for line in lines])
            height, width = image_array.shape[:2]
            center = (width / 2, height / 2)
            rotation_mat = cv2.getRotationMatrix2D(center, np.median(angles), 1.0)

            abs_cos = abs(rotation_mat[0, 0])
            abs_sin = abs(rotation_mat[0, 1])

            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            rotation_mat[0, 2] += bound_w / 2 - center[0]
            rotation_mat[1, 2] += bound_h / 2 - center[1]

            img_rotated = cv2.warpAffine(image_array, rotation_mat, (bound_w, bound_h))
            return img_rotated  # np.median(angles)
        except:  # pylint: disable=W0702
            return image_array
