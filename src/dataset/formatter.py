"""Image formatter"""
# pylint: disable=R0911,R0912
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from nptyping import NDArray

from src.base.image_array import ImageArray, ImageFormat


def format_image(
    image: ImageArray, to_format
) -> Tuple[ImageArray, Optional[NDArray[(Any, Any), np.uint8]]]:
    """To format a nD image into a mD one"""
    if to_format == ImageFormat.BLACK_AND_WHITE:
        if image.ndim == 2:
            return (image, None)
        if image.ndim == 3 and image.shape[-1] == 2:
            mask = image[:, :, 1]
            image = image[:, :, 0]
            return image, mask
        if image.ndim == 3 and image.shape[-1] == 3:
            return (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
        if image.ndim == 3 and image.shape[-1] == 4:
            mask = (image[:, :, 3],)
            image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
            return image, mask
    if to_format == ImageFormat.BLACK_AND_WHITE_WITH_TRANSPARENCY:
        if image.ndim == 2:
            image_with_transparency = np.zeros(
                (image.shape[0], image.shape[1], 2), dtype=np.uint8
            )
            mask = 255 * np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
            image_with_transparency[:, :, 0] = image
            image_with_transparency[:, :, 1] = mask
            return image, mask
        if image.ndim == 3 and image.shape[-1] == 2:
            return image, image[:, :, 3]
        if image.ndim == 3 and image.shape[-1] == 3:
            image_with_transparency = np.zeros(
                (image.shape[0], image.shape[1], 2), dtype=np.uint8
            )
            mask = 255 * np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
            image_with_transparency[:, :, 0] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image_with_transparency[:, :, 1] = mask
            return image_with_transparency, mask
        if image.ndim == 3 and image.shape[-1] == 4:
            image_with_transparency = np.zeros(
                (image.shape[0], image.shape[1], 2), dtype=np.uint8
            )
            mask = (image[:, :, 3],)
            image_with_transparency[:, :, 0] = cv2.cvtColor(
                image[:, :, :3], cv2.COLOR_RGB2GRAY
            )
            image_with_transparency[:, :, 1] = mask
            return image, mask
    if to_format == ImageFormat.COLORED:
        if image.ndim == 2:

            return (cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), None)
        if image.ndim == 3 and image.shape[-1] == 2:
            mask = image[:, :, 1]
            image = image[:, :, 0]

            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), mask
        if image.ndim == 3 and image.shape[-1] == 3:
            return (image, None)

        mask = (image[:, :, 3],)
        image = image[:, :, :3]
        return image, mask

    if image.ndim == 2:
        image_with_transparency = np.zeros(
            (image.shape[0], image.shape[1], 4), dtype=np.uint8
        )
        mask = 255 * np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
        image_with_transparency[:, :, :3] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_with_transparency[:, :, 3] = mask
        return image_with_transparency, mask
    if image.ndim == 3 and image.shape[-1] == 2:
        image_with_transparency = np.zeros(
            (image.shape[0], image.shape[1], 4), dtype=np.uint8
        )
        mask = image[:, :, 1]
        image_with_transparency[:, :, :3] = cv2.cvtColor(
            image[:, :, 0], cv2.COLOR_GRAY2RGB
        )
        image_with_transparency[:, :, 3] = mask
        return image_with_transparency, mask
    if image.ndim == 3 and image.shape[-1] == 3:
        image_with_transparency = np.zeros(
            (image.shape[0], image.shape[1], 4), dtype=np.uint8
        )
        mask = 255 * np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
        image_with_transparency[:, :, :3] = image
        image_with_transparency[:, :, 3] = mask
        return image_with_transparency, mask

    return image, image[:, :, 3]
