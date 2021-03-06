"""Transformers super pixels."""
from enum import Enum

from skimage.color import label2rgb
from skimage.future.graph import cut_threshold, rag_mean_color
from skimage.segmentation import felzenszwalb, quickshift, slic, watershed

from src.base.base_transformer import Transformer
from src.base.image_array import ImageArray


class SuperPixelMode(Enum):
    """Super pixel mode."""

    FELZENSZWALB = "felzenszwalb"
    SLIC = "slic"
    QUICKSHIFT = "quickshift"
    WATERSHED = "watershed"


class TransformerSuperPixel(Transformer):
    """Super pixel transformer."""

    def __init__(
        self, super_pixel_mode: SuperPixelMode, add_rag_thresholding: bool = True
    ) -> None:
        self.super_pixel_mode = super_pixel_mode
        self.add_rag_thresholding = add_rag_thresholding

    def __call__(self, image: ImageArray) -> ImageArray:
        """Apply super pixels."""
        if self.super_pixel_mode == SuperPixelMode.FELZENSZWALB:
            sigma = 0.5
            min_size = 50
            scale = 200

            felzenszwalb_image = felzenszwalb(
                image, scale=scale, sigma=sigma, min_size=min_size
            )

            return label2rgb(
                felzenszwalb_image,
                image,
                kind="avg",
            )

        if self.super_pixel_mode == SuperPixelMode.SLIC:
            num_seg = 500
            compactness = 10
            sigma = 3
            slic_image = slic(
                image, n_segments=num_seg, compactness=compactness, sigma=sigma
            )

            if not self.add_rag_thresholding:

                return label2rgb(slic_image, image, kind="avg")
            return self.rag_thresholding(image, slic_image)

        if self.super_pixel_mode == SuperPixelMode.QUICKSHIFT:
            quickshifted_image = quickshift(image, kernel_size=3, max_dist=6, ratio=0.7)

            if not self.add_rag_thresholding:
                return label2rgb(quickshifted_image, image, kind="avg")
            return self.rag_thresholding(image, quickshifted_image)

        if self.super_pixel_mode == SuperPixelMode.WATERSHED:
            watersheded_image = watershed(image, markers=200, compactness=0.001)

            return label2rgb(watersheded_image, image, kind="avg")

        raise ValueError(f"Unknown super pixel mode: {self.super_pixel_mode}")

    @staticmethod
    def rag_thresholding(image: ImageArray, super_pixels) -> ImageArray:
        """Fill super pixels with avearge value"""
        graph = rag_mean_color(image, super_pixels)
        labels2 = cut_threshold(super_pixels, graph, 29)
        return label2rgb(labels2, image, kind="avg", bg_label=0)

    @staticmethod
    def show():
        """Show"""

    # def fill_average(self, image: ImageArray, markers) -> ImageArray:
    #     """Fill the average of the super pixels."""
    #     final_mask = np.repeat(
    #         np.expand_dims(np.zeros(markers.shape), axis=-1), 3, axis=-1
    #     )
    #     for i in range(1, np.max(markers) + 1):
    #         markers_copy = np.repeat(
    #             np.expand_dims(np.copy(markers), axis=-1), 3, axis=-1
    #         )

    #         markers_copy[markers_copy != i] = 0
    #         markers_copy[markers_copy == i] = 1

    #         avg_val = np.average(image * markers_copy, axis=(0, 1))

    #         final_mask += avg_val * markers_copy

    #     return np.float32(final_mask)


class TransformerRAGSegmentation(Transformer):
    """Segment image using slic and RAG"""

    def __init__(
        self, n_segments: int = 1000, compactness: float = 10.0, threshold: float = 20.0
    ):
        """Initialize the hsv quantization transformer"""
        self.n_segments = n_segments
        self.compactness = compactness
        self.threshold = threshold

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        labels1 = slic(
            input_img,
            compactness=self.compactness,
            n_segments=self.n_segments,
            start_label=1,
        )
        graph_res = rag_mean_color(input_img, labels1)
        labels2 = cut_threshold(labels1, graph_res, self.threshold)
        return label2rgb(labels2, input_img, kind="avg", bg_label=None)

    @staticmethod
    def show():
        """Show"""
