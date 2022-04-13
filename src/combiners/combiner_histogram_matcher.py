"""Histogram matcher combiner."""
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

    def __init__(
        self,
    ):
        """Initialize the histogram matcher combiner."""

    def __call__(
        self,
        input_image: ImageArray,
        reference_image: ImageArray,
    ) -> ImageArray:
        """Match two images."""
        return exposure.match_histograms(
            input_image, reference_image, multichannel=bool(input_image.shape[-1] > 1)
        )
