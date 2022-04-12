"""Base combiner"""
from abc import ABC, abstractmethod

from src.base.image_array import ImageArray


class Combiner(ABC):
    """Generic image combiner"""

    @abstractmethod
    def __call__(self, input_img1: ImageArray, input_img2: ImageArray) -> ImageArray:
        """Combine two images"""
