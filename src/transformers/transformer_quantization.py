"""Quantization transformers."""
import cv2
import numpy as np

from src.base.base_transformer import Transformer
from src.base.image_array import ImageArray


class KmeansQuantizationTransformer(Transformer):
    """Quantize an image to a certain number of colors using k-means"""

    def __init__(self, n_colors: int):
        """Initialize the color quantization transformer"""
        self.n_colors = n_colors

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        data = np.float32(input_img).reshape(  # pylint: disable=too-many-function-args
            (-1, 3)
        )

        # Determine criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

        # Implementing K-Means
        _, label, center = cv2.kmeans(
            data, self.n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        center = np.uint8(center)
        result = center[label.flatten()]  # pylint: disable=unsubscriptable-object
        result = result.reshape(input_img.shape)
        return result


class BinsQuantizationTransformer(Transformer):
    """Quantize an image to a certain number of colors using bins"""

    def __init__(self, n_colors: int):
        """Initialize the color quantization transformer"""
        self.discriminator = 256 // n_colors

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        return (
            input_img // self.discriminator * self.discriminator
            + 128 // self.discriminator
        )


class HSVBinsQuantization(Transformer):
    """Quantize some hsv components using bins"""

    def __init__(
        self, hue_bins: int = 0, saturation_bins: int = 0, value_bins: int = 0
    ):
        """Initialize the hsv quantization transformer"""
        self.hue_discriminator = None if hue_bins <= 0 else 256 // hue_bins
        self.saturation_discriminator = (
            None if saturation_bins <= 0 else 256 // saturation_bins
        )
        self.value_discriminator = None if value_bins <= 0 else 256 // value_bins

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        hsv = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
        if self.hue_discriminator is not None:
            hsv[:, :, 0] = (
                hsv[:, :, 0] // self.hue_discriminator * self.hue_discriminator
                + 128 // self.hue_discriminator
            )
        if self.saturation_discriminator is not None:
            hsv[:, :, 1] = (
                hsv[:, :, 1]
                // self.saturation_discriminator
                * self.saturation_discriminator
                + 128 // self.saturation_discriminator
            )
        if self.value_discriminator is not None:
            hsv[:, :, 2] = (
                hsv[:, :, 2] // self.value_discriminator * self.value_discriminator
                + 128 // self.value_discriminator
            )
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
