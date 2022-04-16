"""Histogram transformers."""
import json

import cv2
import numpy as np

from src import config
from src.base.base_transformer import Transformer
from src.base.image_array import ImageArray


class HistogramEqualizationTransformer(Transformer):
    """Equalize the histogram of an image"""

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        return cv2.equalizeHist(input_img)


class HistogramMatchingTransformer(Transformer):
    """Matches an images histogram with a given one"""

    def __init__(
        self,
        histogram_path: str = config.REVERSED_CARTOON_HISTOGRAM_JSON,
        flat_histogram: bool = False,
    ):
        """Initialize the histogram matching transformer"""
        with open(histogram_path, "r", encoding="utf-8") as file:
            self.reverse_hists = json.load(file)
        self.flat_histogram = flat_histogram

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        hsv = np.asarray(cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV))
        if self.flat_histogram:
            direct_hists = {
                # "hue": cv2.calcHist([hsv[:, :, 0]], [0], None, [256], [0, 256]).flatten(),
                "saturation": cv2.calcHist(
                    [hsv[:, :, 1]], [0], None, [256], [0, 256]
                ).flatten(),
                "value": cv2.calcHist(
                    [hsv[:, :, 2]], [0], None, [256], [0, 256]
                ).flatten(),
            }
            final_hists = {}
            for k, hist in direct_hists.items():  # hist
                hists_cum = np.cumsum(hist)
                hists_norm = (hists_cum / hists_cum[-1] * (len(hists_cum) - 1)).astype(
                    np.uint8
                )
                final_hists[k] = np.array(
                    [
                        self.reverse_hists[k][hists_norm[i]]
                        for i in range(len(hists_norm))
                    ]
                )
        else:
            final_hists = {
                k: np.array(self.reverse_hists[k]) for k in self.reverse_hists
            }
        final_img = np.zeros(hsv.shape)
        # final_img[:,:,0] = cv2.LUT(hsv[:,:,0], final_hists["hue"])
        final_img[:, :, 0] = hsv[:, :, 0]
        final_img[:, :, 1] = cv2.LUT(hsv[:, :, 1], final_hists["saturation"])
        final_img[:, :, 2] = cv2.LUT(hsv[:, :, 2], final_hists["value"])
        return cv2.cvtColor(final_img.astype(np.uint8), cv2.COLOR_HSV2RGB)
