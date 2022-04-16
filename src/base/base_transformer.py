"""Base transformer class."""
from abc import ABC, abstractmethod

from src.base.image_array import ImageArray


class Transformer(ABC):
    """Generic image transformation"""

    @abstractmethod
    def __call__(self, input_img: ImageArray) -> ImageArray:
        """Applies transform to an image"""

    @staticmethod
    @abstractmethod
    def show(**kwargs):
        """Plot trasnformation"""
