"""Dataset utils."""
import os

import cv2
import numpy as np

from src import config


def read_image(image_path: str, add_base: bool = True) -> np.ndarray:
    """Read image from path"""
    if add_base:
        image_path = os.path.join(config.BASE_DATA_PATH, image_path)
    image = cv2.imread(image_path)
    assert image is not None, "Image is None"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def show_image(image: np.ndarray) -> None:
    """Show image"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
