"""Image array type."""
from enum import Enum
from typing import Any, Union

import numpy as np
from nptyping import NDArray

ImageArray = Union[  # type: ignore
    NDArray[(Any, Any), np.uint8],
    NDArray[(Any, Any, 2), np.uint8],
    NDArray[(Any, Any, 3), np.uint8],
    NDArray[(Any, Any, 4), np.uint8],
]


class ImageFormat(Enum):
    """Image format"""

    BLACK_AND_WHITE = "BLACK_AND_WHITE"
    BLACK_AND_WHITE_WITH_TRANSPARENCY = "BLACK_AND_WHITE_WITH_TRANSPARENCY"
    COLORED = "COLORED"
    COLORED_WITH_TRANSPARENCY = "COLORED_WITH_TRANSPARENCY"
