"""Edge transformers."""
import cv2

from src.base.base_transformer import Transformer
from src.base.image_array import ImageArray


class TransformerCannyEdge(Transformer):
    """Apply canny edge detection"""

    def __init__(self, th_min: int = 30, th_max: int = 150):
        """Initialize the canny edge detector"""
        self.th_min = th_min
        self.th_max = th_max

    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""
        return cv2.Canny(input_img, self.th_min, self.th_max)
