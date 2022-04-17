"""Corner transformations classes."""
from typing import Optional

import cv2
import numpy as np
from skimage.morphology import disk

from src.base.base_transformer import Transformer
from src.base.image_array import ImageArray
from src.transformers.transformer_blur import (
    TransformerDilateErode,
    TransformerGaussianBlur,
)
from src.transformers.transformers_super_pixels import TransformerRAGSegmentation


class TranformerSmoothCorners(
    Transformer
):  # pylint: disable=too-many-instance-attributes
    """Smooth angles"""

    def __init__(
        self,
        block_size: int = 4,
        k_size: int = 3,
        k: float = 0.2,
        threshold: float = 1e6,
        mask_kernel_size: int = 10,
        preprocessor_corners: Optional[Transformer] = TransformerRAGSegmentation(
            n_segments=1000, compactness=20.0, threshold=20.0
        ),
        smooth_transformer: Transformer = TransformerDilateErode(
            kernel_type="circle", kernel_size=2
        ),
    ):
        """Initialize the affine transformation"""
        self.block_size = block_size
        self.k_size = k_size
        self.k = k
        self.threshold = threshold
        self.mask_kernel_size = mask_kernel_size
        self.preprocessor_corners = preprocessor_corners
        self.smooth_transformer = smooth_transformer
        self.blur_transformer = TransformerGaussianBlur()

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        # First we detect the corners and make up the mask
        preprocessed_img = (
            self.preprocessor_corners(input_img)
            if self.preprocessor_corners is not None
            else input_img
        )
        gray = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, self.block_size, self.k_size, self.k)
        mask = dst.copy()
        mask[dst > self.threshold] = 1
        mask[dst <= self.threshold] = 0
        mask = cv2.dilate(mask, disk(self.mask_kernel_size))
        mask = self.blur_transformer(mask)
        mask = np.repeat(mask[:, :, None], 3, axis=2)

        # Then we smooth the image
        smoothed_img = self.smooth_transformer(input_img)

        # Finally we combine the 2
        output_img = ((1 - mask) * input_img + mask * smoothed_img).astype(np.uint8)
        return output_img

    @staticmethod
    def show():
        """Show"""
