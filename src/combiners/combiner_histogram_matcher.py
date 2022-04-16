"""Histogram matcher combiner."""
import logging

import cv2
import matplotlib.pyplot as plt
from skimage import exposure

from src.base.base_combiner import Combiner
from src.base.image_array import ImageArray


class HistogramMatcherCombiner(Combiner):
    """
    Histogram matcher combiner.
    ---
    Beneficial when applying image processing pipelines to images
    captured in different lighting conditions, thereby creating a
    “normalized” representation of images, regardless of the lighting
    conditions they were captured in
    """

    def __init__(self, plot: bool = True):
        """Initialize the histogram matcher combiner."""
        self.plot = plot

    def __call__(
        self,
        input_image: ImageArray,
        reference_image: ImageArray,
    ) -> ImageArray:
        """Match two images."""
        hist_image = cv2.calcHist(
            [input_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        hist_image = cv2.normalize(hist_image, hist_image).flatten()
        hist_reference_image = cv2.calcHist(
            [reference_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        hist_reference_image = cv2.normalize(
            hist_reference_image, hist_reference_image
        ).flatten()

        distance = cv2.compareHist(hist_image, hist_reference_image, cv2.HISTCMP_CORREL)
        print(distance)

        matched = exposure.match_histograms(
            input_image, reference_image, multichannel=bool(input_image.shape[-1] > 1)
        )
        if self.plot:
            self.show(input_image, reference_image, matched)
        return matched

    @staticmethod
    def show(
        input_image: ImageArray,
        reference_image: ImageArray,
        matched_image: ImageArray,
    ):
        """Show the histogram matcher combiner."""
        logging.info("Histogram matcher combiner")
        _, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

        for i, img in enumerate((input_image, reference_image, matched_image)):
            for channel, c_color in enumerate(("red", "green", "blue")):
                img_hist, bins = exposure.histogram(
                    img[..., channel], source_range="dtype"
                )
                axes[channel, i].plot(bins, img_hist / img_hist.max())
                img_cdf, bins = exposure.cumulative_distribution(img[..., channel])
                axes[channel, i].plot(bins, img_cdf)
                axes[channel, 0].set_ylabel(c_color)

        axes[0, 0].set_title("Source")
        axes[0, 1].set_title("Reference")
        axes[0, 2].set_title("Matched")

        plt.tight_layout()
        plt.show()
