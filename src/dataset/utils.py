"""Dataset utils."""
import cv2
import numpy as np


def read_image(image_path: str) -> np.ndarray:
    """Read image from path"""
    image = cv2.imread(image_path)
    assert image is not None, "Image is None"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
