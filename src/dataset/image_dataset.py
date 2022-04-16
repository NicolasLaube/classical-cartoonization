"""Image dataset."""
from enum import Enum
from typing import Optional

import pandas as pd

from src import config
from src.base.image_array import ImageArray
from src.dataset.utils import read_image


class DatasetMode(Enum):
    """Dataset type"""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DatasetType(Enum):
    """Dataset type"""

    CARTOONS = "cartoons"
    LANDSCAPES = "landscapes"
    FLICKR = "flickr"


class Dataset:
    """Dataset"""

    def __init__(
        self,
        dataset_type: DatasetType = DatasetType.FLICKR,
        mode: DatasetMode = DatasetMode.TRAIN,
        size: Optional[int] = None,
    ) -> None:
        """Initialize the image dataset."""
        self.df = pd.read_csv(config.DATASETS_CSV_PATH[mode.value][dataset_type.value])
        if size is not None:
            self.df = self.df.sample(size, random_state=42)

    def __len__(self) -> int:
        """Get the number of images in the dataset."""
        return len(self.df)

    def __item__(self, index: int) -> ImageArray:
        """Get an image from the dataset."""
        return read_image(self.df.iloc[index]["path"])

    def __getitem__(self, index: int) -> ImageArray:
        """Get an image from the dataset."""
        return self.__item__(index)
