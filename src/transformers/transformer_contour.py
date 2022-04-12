"""Transformers contour"""
import cv2
import numpy as np

from src.base.base_transformer import Transformer
from src.base.image_array import ImageArray


class ContourTransformer(Transformer):
    """Draw contours of an image"""

    def __init__(
        self,
        mode: int = cv2.RETR_TREE,
        method: int = cv2.CHAIN_APPROX_SIMPLE,
        edge_color: int = 0,
        edge_thickness: int = 1,
    ):
        """Initialize the contour transformer"""
        self.mode = mode
        self.method = method
        self.edge_color = edge_color
        self.edge_thickness = edge_thickness

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        contours, _ = cv2.findContours(input_img, self.mode, self.method)
        # sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # create mask for drawing contours
        mask = 255 * np.ones(input_img.shape, dtype=np.uint8)
        # draw contours on mask
        cv2.drawContours(
            mask,
            contours,
            contourIdx=-1,
            color=self.edge_color,
            thickness=self.edge_thickness,
        )

        # format
        image_with_mask = np.zeros(
            (input_img.shape[0], input_img.shape[1], 2), dtype=np.uint8
        )
        image_with_mask[:, :, 0] = mask
        image_with_mask[:, :, 1] = cv2.bitwise_not(mask)
        return image_with_mask


class AdaptiveThresholdContour(Transformer):
    """Draw contours thanks to adaptive threshold"""

    def __init__(self, line_size: int = 3, blur_value: int = 3):
        """Initialize the adaptive threshold contour"""
        self.line_size = line_size
        self.blur_value = blur_value

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        edges = cv2.adaptiveThreshold(
            input_img,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self.line_size,
            self.blur_value,
        )
        image_with_mask = np.zeros(
            (input_img.shape[0], input_img.shape[1], 2), dtype=np.uint8
        )
        image_with_mask[:, :, 0] = edges
        image_with_mask[:, :, 1] = cv2.bitwise_not(edges)
        return image_with_mask
