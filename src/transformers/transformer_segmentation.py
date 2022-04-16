"""Quantization transformers."""
from skimage import segmentation, color
from skimage.future import graph

from src.base.base_transformer import Transformer
from src.base.image_array import ImageArray


class RAGSegmentationTransformer(Transformer):
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
        labels1 = segmentation.slic(
            input_img,
            compactness=self.compactness,
            n_segments=self.n_segments,
            start_label=1,
        )
        graph_res = graph.rag_mean_color(input_img, labels1)
        labels2 = graph.cut_threshold(labels1, graph_res, self.threshold)
        return color.label2rgb(labels2, input_img, kind="avg", bg_label=None)
